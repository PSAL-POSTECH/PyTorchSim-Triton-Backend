"""Mixtral's MoE router -- top-k, and a reduction that returns an INDEX.

    logits = x @ Wg.T                 [T, E]   one score per expert per token
    p      = softmax(logits)                   over the EXPERTS
    i1, i2 = the two largest                   top-k, k = 2
    out    = g1 * E_{i1}(x) + g2 * E_{i2}(x)   g = the two probs, renormalised

EIGHT EXPERTS, TWO USED. Mixtral-8x7B has 47B parameters and spends 13B per
token: the FFN is eight copies and the router picks two per token per layer. The
argument is that capacity and compute stop being the same number -- a model can
be large in what it knows and small in what it runs.

TOP-K IS THE OP THIS SUITE DID NOT HAVE. Every reduction here folds to a VALUE:
a sum, a max, a mean. This one folds to a POSITION, and the position is then
used as an address -- so a reduction is immediately followed by a gather that
depends on its result. That pairing is what makes routing awkward everywhere,
not just here.

  k = 2 WITHOUT A SORT. Take the argmax, mask that expert to -inf, take the
  argmax again. Two passes over eight elements beats sorting them, and it is
  what the reference implementations do for small k.

RENORMALISED OVER THE CHOSEN TWO. `g1 + g2 == 1` after the division, not after
the softmax -- the six unchosen experts' probability is discarded rather than
spread. Mixtral does it this way (`softmax` then `topk` then divide); some other
MoEs take the topk first and softmax only the survivors, which is not the same
function.

THE EXPERT IS A PER-CHANNEL SCALE HERE, not a pair of matmuls. A real expert is
SwiGLU -- `swiglu.py` -- and what this kernel is for is the ROUTING: the index
that comes out of a reduction, the row gather that reads a table with it, and
the weighted combine that puts two of them back together. Making the expert a
[E, D] table keeps that visible instead of burying it in two GEMMs.

WHAT IT ASKS OF THIS MACHINE. The gather is `s[i[t], :]` -- one index per TOKEN,
addressing a row. With the tokens on the lanes that is one index per lane, which
is the cheap form the DMA wants (see convert_accesses_to_transfer: the gathered axis
being the lane axis means one index per row rather than one per element).

torch equivalent:
    p = softmax(x @ wg.T, dim=1)
    g, i = p.topk(2, dim=1);  g = g / g.sum(1, keepdim=True)
    out = g[:, 0:1] * s[i[:, 0]] * x + g[:, 1:2] * s[i[:, 1]] * x

Mixtral: top-k routing -- a reduction that returns an INDEX, and a row gather addressed by it, then a weighted combine of two experts.

LAYOUT PIN. `expect_ir` holds `torchsim.copy` in 04-passes/04-torchsim-materialize-layouts.mlir at 0: the layouts come down decided, so materialize-layouts writes no copy of its own.
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
    def moe_router_kernel(x_ptr, wg_ptr, s_ptr, o_ptr,
                          T: tl.constexpr, E: tl.constexpr, D: tl.constexpr):
        t = tl.arange(0, T)
        e = tl.arange(0, E)
        d = tl.arange(0, D)

        x = tl.load(x_ptr + t[:, None] * D + d[None, :])          # [T, D]

        # ROUTER: one score per (token, expert), built a COLUMN AT A TIME.
        # `x[:, None, :] * wg[None, :, :]` is the same map and materialises a
        # [T, E, D] tile -- rank 3 with a broadcast on each of two axes, which
        # is both E times the memory and a staging vectorize refuses.
        # E is 8 and static, so the loop is free and every tile stays rank 2.
        logits = tl.zeros((T, E), dtype=tl.float32)
        for j in tl.static_range(E):
            wj = tl.load(wg_ptr + j * D + d)                      # [D]
            lj = tl.sum(x * wj[None, :], axis=1)                  # [T]
            logits = tl.where(e[None, :] == j, lj[:, None], logits)

        # SOFTMAX OVER THE EXPERTS, then the top two of it.
        m = tl.max(logits, axis=1)
        ex = tl.exp(logits - m[:, None])
        p = ex / tl.sum(ex, axis=1)[:, None]

        # TOP-2 WITHOUT A SORT: argmax, mask, argmax again.
        p1 = tl.max(p, axis=1)
        i1 = tl.argmax(p, axis=1)
        p_2 = tl.where(e[None, :] == i1[:, None], -1.0, p)
        p2 = tl.max(p_2, axis=1)
        i2 = tl.argmax(p_2, axis=1)

        # RENORMALISE over the two that were kept.
        tot = p1 + p2
        g1 = p1 / tot
        g2 = p2 / tot

        # DISPATCH: a row gather whose index came out of the reduction above.
        s1 = tl.load(s_ptr + i1[:, None] * D + d[None, :])        # [T, D]
        s2 = tl.load(s_ptr + i2[:, None] * D + d[None, :])

        out = g1[:, None] * s1 * x + g2[:, None] * s2 * x
        tl.store(o_ptr + t[:, None] * D + d[None, :], out)

    return moe_router_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(T * D, generator=g, dtype=torch.float32),
            "wg": torch.randn(E * D, generator=g, dtype=torch.float32),
            "s": torch.randn(E * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(T, D)
    wg = inputs["wg"].reshape(E, D)
    s = inputs["s"].reshape(E, D)
    p = torch.softmax(x @ wg.T, dim=1)
    g, i = p.topk(2, dim=1)
    g = g / g.sum(1, keepdim=True)
    out = g[:, 0:1] * s[i[:, 0]] * x + g[:, 1:2] * s[i[:, 1]] * x
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="moe_router_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "wg_ptr": "*fp32", "s_ptr": "*fp32",
               "o_ptr": "*fp32", "T": "constexpr", "E": "constexpr",
               "D": "constexpr"},
    constexprs={"T": T, "E": E, "D": D},
    args=[Arg("x", IN, "float32", (T * D,)),
          Arg("wg", IN, "float32", (E * D,)),
          Arg("s", IN, "float32", (E * D,)),
          Arg("o", OUT, "float32", (T * D,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"04-passes/04-torchsim-materialize-layouts.mlir": {"torchsim.copy": 0}},
)
