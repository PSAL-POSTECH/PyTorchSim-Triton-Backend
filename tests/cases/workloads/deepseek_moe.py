"""DeepSeek-V3's MoE -- shared experts, sigmoid routing, and a balancing bias.

    s      = sigmoid(x @ Wg.T)                 per-expert affinity, NOT a softmax
    pick   = top-k of (s + bias)               bias balances load, and is NOT
    g      = s[pick] / sum(s[pick])            in the weight
    out    = shared(x) + sum_k g_k * E_k(x)

THREE CHANGES FROM MIXTRAL, and each is a different lesson about what a router
is for.

**SHARED EXPERTS ARE ALWAYS ON.** One (or a few) experts every token uses, plus
the routed ones. The argument is redundancy: if every expert must handle the
common case, every expert spends capacity on it. Factor that out and the routed
experts specialise. DeepSeek-V3 has 1 shared and 8 of 256 routed per token.

**SIGMOID, NOT SOFTMAX.** A softmax over experts makes the affinities compete --
raising one lowers the rest, even when they are unrelated. A sigmoid scores each
expert on its own, and the normalisation happens AFTER the top-k over just the
survivors. With 256 experts the softmax's coupling is most of what it computes.

**THE BALANCING BIAS IS IN THE CHOICE, NOT IN THE WEIGHT** -- and that is the
"auxiliary-loss-free" part. Earlier MoEs added a load-balancing term to the LOSS,
which fights the language objective. V3 instead keeps a per-expert bias, adds it
only when ranking, and nudges it outside the forward pass: an overloaded expert's
bias goes down so it is picked less, while the WEIGHT it gets when picked is
still the raw affinity. Balancing therefore costs the model nothing in quality,
because it never touches the value.

    pick by   s + bias        <- the bias is here
    weight by s               <- and not here

WHAT IT ASKS OF THIS MACHINE. `moe_router` already established the shape that
matters: a reduction folding to a POSITION, and that position becoming an
address. What is new is that the ranked quantity and the weighted quantity are
DIFFERENT tensors, so the top-k's argmax runs over one and its result indexes
the other -- two rank-1 vectors that have to stay aligned through the same
selection.

torch equivalent:
    s = torch.sigmoid(x @ wg.T)
    idx = (s + bias).topk(2, -1).indices
    g = s.gather(-1, idx);  g = g / g.sum(-1, keepdim=True)
    out = x * shared + g[:, 0:1] * w[idx[:, 0]] * x + g[:, 1:2] * w[idx[:, 1]] * x

DeepSeek-V3: shared experts, sigmoid affinities, and a balancing bias that enters the CHOICE but never the weight.

LAYOUT PIN. `expect_ir` holds `torchsim.copy` in 08-passes/04-torchsim-materialize-layouts.mlir at 0: the layouts come down decided, so materialize-layouts writes no copy of its own.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, E, D = 64, 8, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def deepseek_moe_kernel(x_ptr, wg_ptr, b_ptr, sh_ptr, w_ptr, o_ptr,
                            T: tl.constexpr, E: tl.constexpr,
                            D: tl.constexpr):
        t = tl.arange(0, T)
        e = tl.arange(0, E)
        d = tl.arange(0, D)
        x = tl.load(x_ptr + t[:, None] * D + d[None, :])

        # AFFINITIES, one expert at a time so every tile stays rank 2.
        s = tl.zeros((T, E), dtype=tl.float32)
        for jx in tl.static_range(E):
            wj = tl.load(wg_ptr + jx * D + d)
            lj = tl.sum(x * wj[None, :], axis=1)
            s = tl.where(e[None, :] == jx, lj[:, None], s)
        s = tl.sigmoid(s)              # per expert, not against the others

        # RANK BY s + bias, WEIGHT BY s. The bias only ever enters the choice.
        r = s + tl.load(b_ptr + e)[None, :]
        r1 = tl.max(r, axis=1)
        i1 = tl.argmax(r, axis=1)
        r_2 = tl.where(e[None, :] == i1[:, None], -1e30, r)
        i2 = tl.argmax(r_2, axis=1)

        # The weights come from `s` at the positions `r` chose -- two rank-1
        # vectors kept aligned through one selection.
        g1 = tl.sum(tl.where(e[None, :] == i1[:, None], s, 0.0), axis=1)
        g2 = tl.sum(tl.where(e[None, :] == i2[:, None], s, 0.0), axis=1)
        tot = g1 + g2
        g1 = g1 / tot
        g2 = g2 / tot

        # SHARED first, then the two routed ones. The shared expert needs no
        # index at all -- every token uses it.
        out = x * tl.load(sh_ptr + d)[None, :]
        out += g1[:, None] * tl.load(w_ptr + i1[:, None] * D + d[None, :]) * x
        out += g2[:, None] * tl.load(w_ptr + i2[:, None] * D + d[None, :]) * x
        tl.store(o_ptr + t[:, None] * D + d[None, :], out + r1[:, None] * 0.0)

    return deepseek_moe_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(T * D, generator=g, dtype=torch.float32),
            "wg": torch.randn(E * D, generator=g, dtype=torch.float32),
            "b": torch.randn(E, generator=g, dtype=torch.float32) * 0.2,
            "sh": torch.randn(D, generator=g, dtype=torch.float32),
            "w": torch.randn(E * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(T, D)
    s = torch.sigmoid(x @ inputs["wg"].reshape(E, D).T)
    r = s + inputs["b"][None, :]
    idx = r.topk(2, dim=1).indices
    gv = s.gather(1, idx)
    gv = gv / gv.sum(1, keepdim=True)
    w = inputs["w"].reshape(E, D)
    out = x * inputs["sh"][None, :]
    out = out + gv[:, 0:1] * w[idx[:, 0]] * x + gv[:, 1:2] * w[idx[:, 1]] * x
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="deepseek_moe_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "wg_ptr": "*fp32", "b_ptr": "*fp32",
               "sh_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "T": "constexpr", "E": "constexpr", "D": "constexpr"},
    constexprs={"T": T, "E": E, "D": D},
    args=[Arg("x", IN, "float32", (T * D,)),
          Arg("wg", IN, "float32", (E * D,)),
          Arg("b", IN, "float32", (E,)),
          Arg("sh", IN, "float32", (D,)),
          Arg("w", IN, "float32", (E * D,)),
          Arg("o", OUT, "float32", (T * D,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"08-passes/04-torchsim-materialize-layouts.mlir": {"torchsim.copy": 0}},
)
