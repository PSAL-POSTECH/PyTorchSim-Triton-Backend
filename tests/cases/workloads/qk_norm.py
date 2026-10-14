"""QK-norm -- Qwen3 and Gemma 3 normalise q and k before the dot product.

    qn = q * rsqrt(mean(q^2, -1) + eps) * gq
    kn = k * rsqrt(mean(k^2, -1) + eps) * gk
    scores = qn @ kn.T * scale

WHY NORMALISE BEFORE THE DOT. Attention logits are a dot product of two learned
vectors, and nothing bounds them: as a model gets deeper or wider the logits
drift up, the softmax saturates, and the gradient through it vanishes. Training
runs diverge from exactly this, late and expensively. Normalising q and k to
unit RMS first makes the logit a scaled COSINE rather than an inner product, so
its range is fixed by construction and the temperature is the only thing that
sets it.

  Gemma 2 attacked the same problem from the other end with `tanh` soft-capping
  on the logits, then Gemma 3 dropped the cap and adopted QK-norm instead.
  Qwen3 has it too. It is one of the few recent changes that is about TRAINING
  stability and has no inference-quality story at all.

  The scale is per HEAD DIM, not per head -- `gq` and `gk` are length D, the
  same shape RMSNorm's weight always has.

WHAT IT ASKS OF THIS MACHINE, and why this kernel exists. The normalisation
folds the head-dim axis; the matmul REDUCES over that same axis and pins its
own last axis to the lanes. So one buffer carries a reduction's "not this axis"
and the array's "this axis" at once, which is the shape `dl/attention` needed
five separate fixes to get through. The difference here is that the two demands
are about the INPUT rather than about the score matrix.

  The softmax is deliberately not here. A score matrix with no loop around it
  is refused (see the rulebook), so including it would test that refusal
  instead of this chain -- `softmax.py` has the fold and `attention.py` has the
  looped version.

torch equivalent:
    qn = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps) * gq
    kn = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps) * gk
    scores = qn @ kn.T * scale

GREEN SINCE 2026-09-04. vectorize refused this with "operands of one
elementwise op are banked on different iteration dims": one buffer a matmul pins
on one axis sits in an all-parallel op beside buffers that follow the op onto the
other. select_lane_axis owned that decision and did not know it -- its
`_is_one_iteration_dim` reported only the disagreements a widen could already
fix, so the ladder never reached step 4 and the refusal fell out a pass later.

The pinned operand here is the op's `outs`, and a relayout mints a new value
while the op already produces that one. So the cut goes on the CONSUMER instead:
the matmul that pins it reads a copy banked its way, and the producing op is free
to follow its own operands. Same trip, no new rewrite. Costs 1 round trip, which is the cheapest plan the cost table offered.

Qwen3 / Gemma 3: a reduction over the head dim feeding a matmul that reduces over the SAME axis -- one buffer carrying a NOT and a MUST.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, D = 64, 32
BK = 16
EPS = 1e-6
SCALE = 1.0 / (D ** 0.5)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def qk_norm_kernel(q_ptr, k_ptr, gq_ptr, gk_ptr, o_ptr,
                       S: tl.constexpr, D: tl.constexpr, BK: tl.constexpr,
                       EPS: tl.constexpr, SCALE: tl.constexpr):
        i = tl.arange(0, S)
        j = tl.arange(0, S)
        d = tl.arange(0, D)
        kk = tl.arange(0, BK)

        # RMSNorm on each ROW of q and of k. The fold is over the head dim --
        # the same axis the matmul below reduces over.
        q = tl.load(q_ptr + i[:, None] * D + d[None, :])
        qn = q * tl.exp(-0.5 * tl.log(tl.sum(q * q, axis=1) / D + EPS))[:, None]
        qn = qn * tl.load(gq_ptr + d)[None, :]

        k = tl.load(k_ptr + j[:, None] * D + d[None, :])
        kn = k * tl.exp(-0.5 * tl.log(tl.sum(k * k, axis=1) / D + EPS))[:, None]
        kn = kn * tl.load(gk_ptr + d)[None, :]

        # The scores. k is transposed by its own addressing, the way
        # dl/attention reads it, so PtrAnalysis stays affine -- but kn is a
        # VALUE now, so the transpose has to happen on the way in and the K
        # loop reads slices of the already-normalised tiles. Triton cannot
        # slice a value, so the whole thing is one dot: D is the reduced
        # extent and it is 32, which fits.
        tl.store(o_ptr + i[:, None] * S + j[None, :],
                 tl.dot(qn, tl.trans(kn)) * SCALE)

    return qk_norm_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"q": torch.randn(S * D, generator=g, dtype=torch.float32),
            "k": torch.randn(S * D, generator=g, dtype=torch.float32),
            "gq": torch.randn(D, generator=g, dtype=torch.float32),
            "gk": torch.randn(D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    q = inputs["q"].reshape(S, D)
    k = inputs["k"].reshape(S, D)
    qn = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + EPS) * inputs["gq"]
    kn = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + EPS) * inputs["gk"]
    return {"o": (qn @ kn.T * SCALE).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="qk_norm_kernel",
    kernel=kernel,
    signature={"q_ptr": "*fp32", "k_ptr": "*fp32", "gq_ptr": "*fp32",
               "gk_ptr": "*fp32", "o_ptr": "*fp32",
               "S": "constexpr", "D": "constexpr", "BK": "constexpr",
               "EPS": "constexpr", "SCALE": "constexpr"},
    constexprs={"S": S, "D": D, "BK": BK, "EPS": EPS, "SCALE": SCALE},
    args=[Arg("q", IN, "float32", (S * D,)),
          Arg("k", IN, "float32", (S * D,)),
          Arg("gq", IN, "float32", (D,)),
          Arg("gk", IN, "float32", (D,)),
          Arg("o", OUT, "float32", (S * S,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
    expect="blocked",
    expect_why="matmul K=32 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
