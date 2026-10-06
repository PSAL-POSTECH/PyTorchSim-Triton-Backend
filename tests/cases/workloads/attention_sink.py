"""Attention sinks -- a term in the softmax denominator that is not a key.

    p_j = exp(s_j) / (exp(sink_h) + sum_k exp(s_k))

    sink_h is a learned scalar, one per head, and there is no j for it.

THE PROBABILITIES NO LONGER SUM TO ONE, and that is the entire point. A softmax
is forced to spend all of its mass somewhere; when a head has nothing it wants
to attend to, it has to pick something anyway. Xiao et al. (2023) observed that
trained models solve this by dumping the surplus on the FIRST token -- position
0 gets a huge weight in most heads regardless of what it is -- and named it an
attention sink. Evicting it from a streaming KV cache destroyed the model, which
is how the phenomenon was found.

A sink term makes the escape valve explicit: a denominator entry with no key
behind it, so a head that wants to attend to nothing attends to nothing and the
row simply sums to less than one. GPT-OSS ships them as a learned per-head
parameter; StreamingLLM keeps the first few tokens instead.

  Both are the same fix for the same thing. The learned version is cheaper --
  no cache slots -- and does not depend on whatever happened to be at position 0.

NUMERICALLY THE SINK JOINS THE MAX, not just the sum. The stable softmax
subtracts a row maximum before exponentiating, and if the sink is larger than
every score then leaving it out of that maximum overflows the one term that was
supposed to be the safe one. So `m = max(max_j s_j, sink)` and both the row and
the sink are shifted by it.

WHAT IT ASKS OF THIS MACHINE. The sink is one scalar per head, added to a rank-1
denominator -- so it is a scalar meeting a fold's result, which is the shape
`_replicate_captured_scalars` stages. The heads are a static loop for the usual
reason: a head axis in the tile would put the per-head scalar on an axis that is
neither the lane nor the vector one.

torch equivalent:
    m = torch.maximum(s.max(-1, keepdim=True).values, sink)
    e = (s - m).exp()
    p = e / (e.sum(-1, keepdim=True) + (sink - m).exp())

GPT-OSS: a denominator entry with no key behind it, so a row may sum to less than one -- and it has to join the max, not just the sum.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, H = 64, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def attention_sink_kernel(sc_ptr, sink_ptr, o_ptr,
                              S: tl.constexpr, H: tl.constexpr):
        i = tl.arange(0, S)
        j = tl.arange(0, S)
        causal = (i[:, None] - j[None, :]) >= 0

        for h in tl.static_range(H):
            sk = tl.load(sink_ptr + h)
            sc = tl.load(sc_ptr + h * S * S + i[:, None] * S + j[None, :])
            b = tl.where(causal, sc, float("-inf"))

            # THE SINK JOINS THE MAX. Leaving it out overflows exactly the term
            # that was meant to be safe, whenever the sink is the largest thing
            # in the row.
            m = tl.maximum(tl.max(b, axis=1), sk)

            ex = tl.where(causal, tl.exp(b - m[:, None]), 0.0)
            # ...and the denominator carries one entry with no key behind it.
            den = tl.sum(ex, axis=1) + tl.exp(sk - m)
            p = ex / den[:, None]
            tl.store(o_ptr + h * S * S + i[:, None] * S + j[None, :], p)

    return attention_sink_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"sc": torch.randn(H * S * S, generator=g, dtype=torch.float32) * 2,
            "sink": torch.randn(H, generator=g, dtype=torch.float32) * 2}


def reference(inputs):
    import torch
    sc = inputs["sc"].reshape(H, S, S)
    sink = inputs["sink"]
    i = torch.arange(S)[:, None]
    j = torch.arange(S)[None, :]
    causal = (i - j) >= 0
    out = torch.empty_like(sc)
    for h in range(H):
        b = sc[h].masked_fill(~causal, float("-inf"))
        m = torch.maximum(b.max(-1).values, sink[h])
        e = torch.where(causal, (b - m[:, None]).exp(), torch.zeros_like(b))
        out[h] = e / (e.sum(-1) + (sink[h] - m).exp())[:, None]
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="attention_sink_kernel",
    kernel=kernel,
    signature={"sc_ptr": "*fp32", "sink_ptr": "*fp32", "o_ptr": "*fp32",
               "S": "constexpr", "H": "constexpr"},
    constexprs={"S": S, "H": H},
    args=[Arg("sc", IN, "float32", (H * S * S,)),
          Arg("sink", IN, "float32", (H,)),
          Arg("o", OUT, "float32", (H * S * S,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
