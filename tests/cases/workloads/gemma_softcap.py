"""Gemma 2's logit soft-capping, and its alternating local / global attention.

    s = tanh(s / cap) * cap          bounded, smooth, and NOT a clamp
    layer even -> sliding window of W;  layer odd -> full causal

TWO THINGS THAT LOOK UNRELATED AND ARE THE SAME BET: bound what the model can
do so it does not have to be trusted not to.

**SOFT-CAPPING IS A CLAMP THAT HAS A GRADIENT.** Attention logits and final
logits both drift up during training and both can blow the softmax out.
`min(x, c)` would bound them and kill the gradient above the cap; `tanh(x/c)*c`
bounds them at the same place, stays monotone, and keeps a gradient everywhere.
Gemma 2 caps attention logits at 50 and output logits at 30.

  Gemma 3 REMOVED it and adopted QK-norm instead (`qk_norm.py`) -- two answers
  to one problem, and the later one attacks the inputs rather than the output.
  Worth having both: they are the same worry a version apart.

**ALTERNATING ATTENTION IS A CACHE ARGUMENT, NOT A QUALITY ONE.** Every second
layer sees only a 4096-token window, so its KV entries can be dropped once they
leave the window; only the global layers pay for the full context. Gemma 2 goes
1:1, Gemma 3 goes 5:1 -- five local layers per global one, which cuts the KV
cache by most of it while a global layer every sixth keeps long-range reach.

  It is the same trade Mistral makes with a pure sliding window
  (`sliding_window.py`), except a pure window can only reach far through DEPTH,
  and here one layer in six reaches directly.

WHAT IT ASKS OF THIS MACHINE. The cap is elementwise -- `tanh` is native here,
which is the whole cost. The alternation matters more than it looks: the mask
differs BY LAYER, so a stack is two kernels called in turn rather than one
kernel with a parameter, and the two have different KV lifetimes. This kernel
runs both masks so the pair can be read side by side.

torch equivalent:
    s = torch.tanh(s / cap) * cap
    s = s.masked_fill(band_or_causal, -inf)
    p = s.softmax(-1)

Gemma 2: a clamp that keeps its gradient, and the two masks a stack alternates between.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, W = 64, 16
CAP = 8.0


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemma_softcap_kernel(sc_ptr, lo_ptr, gl_ptr,
                             S: tl.constexpr, W: tl.constexpr,
                             CAP: tl.constexpr):
        i = tl.arange(0, S)
        j = tl.arange(0, S)
        dist = i[:, None] - j[None, :]

        sc = tl.load(sc_ptr + i[:, None] * S + j[None, :])
        # SOFT-CAP. tanh as `2*sigmoid(2u) - 1`, the form activations.py uses --
        # the (e-1)/(e+1) spelling overflows for a logit the cap exists to bound.
        u = sc / CAP
        capped = (2.0 * tl.sigmoid(2.0 * u) - 1.0) * CAP

        # LOCAL LAYER: a band, the shape sliding_window.py has.
        keep = (dist >= 0) & (dist < W)
        b = tl.where(keep, capped, float("-inf"))
        m = tl.max(b, axis=1)
        ex = tl.where(keep, tl.exp(b - m[:, None]), 0.0)
        tl.store(lo_ptr + i[:, None] * S + j[None, :],
                 ex / tl.sum(ex, axis=1)[:, None])

        # GLOBAL LAYER: the same capped scores, the half-plane instead.
        keep = dist >= 0
        b = tl.where(keep, capped, float("-inf"))
        m = tl.max(b, axis=1)
        ex = tl.where(keep, tl.exp(b - m[:, None]), 0.0)
        tl.store(gl_ptr + i[:, None] * S + j[None, :],
                 ex / tl.sum(ex, axis=1)[:, None])

    return gemma_softcap_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"sc": torch.randn(S * S, generator=g, dtype=torch.float32) * 20}


def reference(inputs):
    import torch
    sc = inputs["sc"].reshape(S, S)
    capped = torch.tanh(sc / CAP) * CAP
    i = torch.arange(S)[:, None]
    j = torch.arange(S)[None, :]
    d = i - j
    out = {}
    for name, keep in (("lo", (d >= 0) & (d < W)), ("gl", d >= 0)):
        b = capped.masked_fill(~keep, float("-inf"))
        out[name] = torch.softmax(b, dim=-1).reshape(-1).contiguous()
    return out


SPEC = KernelSpec(
    name="gemma_softcap_kernel",
    kernel=kernel,
    signature={"sc_ptr": "*fp32", "lo_ptr": "*fp32", "gl_ptr": "*fp32",
               "S": "constexpr", "W": "constexpr", "CAP": "constexpr"},
    constexprs={"S": S, "W": W, "CAP": CAP},
    args=[Arg("sc", IN, "float32", (S * S,)),
          Arg("lo", OUT, "float32", (S * S,)),
          Arg("gl", OUT, "float32", (S * S,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
