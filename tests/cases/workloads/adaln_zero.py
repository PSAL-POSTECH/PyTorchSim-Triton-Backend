"""DiT's adaLN-Zero -- six modulation vectors per block, and why one is a GATE.

    shift1, scale1, gate1, shift2, scale2, gate2 = MLP(c).chunk(6)
    h = x + gate1 * attn(LN(x) * (1 + scale1) + shift1)
    y = h + gate2 * mlp(LN(h) * (1 + scale2) + shift2)

CONDITIONING BY MODULATION, TAKEN FURTHER THAN FiLM. `groupnorm_film` has a
scale and a shift; DiT has three per sub-block, and the third is what the paper
is named after:

    shift, scale   FiLM on the normalised activations -- adaptive LayerNorm
    gate           multiplies the whole residual BRANCH

**gate is initialised to ZERO**, so a fresh block is exactly the identity and the
residual stream passes through untouched. Peebles & Xie (2023) measure this as
the single largest win in the paper -- larger than the conditioning mechanism
itself -- because it lets a deep stack of transformer blocks start as a deep
stack of identity functions and grow into the task. It is the same idea as
ControlNet's zero-convs and LayerScale's 1e-6 init (`convnext_block`).

  Why DiT matters: it replaced the U-Net with a plain transformer over patches
  and showed diffusion quality scales with transformer FLOPs the way language
  models do. SD3 and Flux are its descendants.

PATCHIFY IS A RESHAPE, AND THAT IS THE POINT. A `[C, P, P]` patch becomes one
token of `C*P*P` features, so the spatial axes leave the tile entirely and what
follows is a sequence model. Here it is the reverse of `channel_shuffle`'s
factored transpose -- axes are merged rather than permuted -- and it costs
nothing when the merged axes are already adjacent in memory.

WHAT IT ASKS OF THIS MACHINE. Six per-channel vectors read against one tile,
which is `bn_relu_add`'s shape six times over; the interesting part is that
three of them are consumed at DIFFERENT points of the block, so the same tile is
modulated, used, and modulated again. Nothing crosses lanes.

torch equivalent:
    s1, c1, g1, s2, c2, g2 = mlp(cond).chunk(6, dim=-1)
    h = x + g1 * (ln(x) * (1 + c1) + s1) @ w1
    y = h + g2 * (ln(h) * (1 + c2) + s2) @ w2

DiT: six modulation vectors against one tile, two of them GATES that start at zero so a fresh block is the identity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, C = 64, 32              # tokens (patchified), channels per token
EPS = 1e-6


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def adaln_zero_kernel(x_ptr, m_ptr, w1_ptr, w2_ptr, o_ptr,
                          T: tl.constexpr, C: tl.constexpr,
                          EPS: tl.constexpr):
        t = tl.arange(0, T)
        c = tl.arange(0, C)
        idx = t[:, None] * C + c[None, :]
        x = tl.load(x_ptr + idx)

        # THE SIX, read as six rows of one [6, C] table -- which is what
        # `.chunk(6)` of the MLP's output is once it is in memory.
        s1 = tl.load(m_ptr + 0 * C + c)
        c1 = tl.load(m_ptr + 1 * C + c)
        g1 = tl.load(m_ptr + 2 * C + c)
        s2 = tl.load(m_ptr + 3 * C + c)
        c2 = tl.load(m_ptr + 4 * C + c)
        g2 = tl.load(m_ptr + 5 * C + c)

        # SUB-BLOCK ONE: normalise, modulate, transform, GATE, add.
        mu = tl.sum(x, axis=1) / C
        d = x - mu[:, None]
        var = tl.sum(d * d, axis=1) / C
        n = d * tl.exp(-0.5 * tl.log(var + EPS))[:, None]
        n = n * (1.0 + c1[None, :]) + s1[None, :]
        h = x + g1[None, :] * (n * tl.load(w1_ptr + c)[None, :])

        # SUB-BLOCK TWO: the same tile, normalised and modulated again with the
        # OTHER three. That the two halves share a tile is the shape here.
        mu = tl.sum(h, axis=1) / C
        d = h - mu[:, None]
        var = tl.sum(d * d, axis=1) / C
        n = d * tl.exp(-0.5 * tl.log(var + EPS))[:, None]
        n = n * (1.0 + c2[None, :]) + s2[None, :]
        y = h + g2[None, :] * (n * tl.load(w2_ptr + c)[None, :])

        tl.store(o_ptr + idx, y)

    return adaln_zero_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(T * C, generator=g, dtype=torch.float32),
            "m": torch.randn(6 * C, generator=g, dtype=torch.float32) * 0.3,
            "w1": torch.randn(C, generator=g, dtype=torch.float32),
            "w2": torch.randn(C, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(T, C)
    m = inputs["m"].reshape(6, C)
    s1, c1, g1, s2, c2, g2 = (m[k] for k in range(6))

    def ln(v):
        mu = v.mean(1, keepdim=True)
        d = v - mu
        return d * torch.rsqrt((d * d).mean(1, keepdim=True) + EPS)

    n = ln(x) * (1.0 + c1[None, :]) + s1[None, :]
    h = x + g1[None, :] * (n * inputs["w1"][None, :])
    n = ln(h) * (1.0 + c2[None, :]) + s2[None, :]
    y = h + g2[None, :] * (n * inputs["w2"][None, :])
    return {"o": y.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="adaln_zero_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "m_ptr": "*fp32", "w1_ptr": "*fp32",
               "w2_ptr": "*fp32", "o_ptr": "*fp32",
               "T": "constexpr", "C": "constexpr", "EPS": "constexpr"},
    constexprs={"T": T, "C": C, "EPS": EPS},
    args=[Arg("x", IN, "float32", (T * C,)),
          Arg("m", IN, "float32", (6 * C,)),
          Arg("w1", IN, "float32", (C,)),
          Arg("w2", IN, "float32", (C,)),
          Arg("o", OUT, "float32", (T * C,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
