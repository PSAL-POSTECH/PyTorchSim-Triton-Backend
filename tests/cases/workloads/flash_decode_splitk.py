"""Flash-decoding -- split the KV, softmax each half, then MERGE the halves.

torch equivalent:
    o = softmax(q @ k.T) @ v          # but computed as two independent halves

WHY DECODING SPLITS THE KV AT ALL
  One decoded token is one query against a KV cache thousands long. Tiling the
  query axis gives one tile, so the machine sits idle; the parallelism has to
  come from the KV axis instead. Every chunk computes a PARTIAL softmax over its
  own keys, and the chunks are combined afterwards.

THE MERGE IS THE WHOLE POINT, AND IT IS NOT AN ADD
  A softmax is not decomposable by concatenation: each half normalised by its
  own max and its own denominator. Rescaling both to the joint max is what makes
  them addable:

      m = max(m1, m2)
      o = (o1 * l1 * e^(m1-m) + o2 * l2 * e^(m2-m)) / (l1 * e^(m1-m) + l2 * e^(m2-m))

  So each half has to carry THREE things out -- its output, its max, and its
  denominator -- and the combine multiplies by an exponential of a difference of
  maxima. That is the same algebra dl/attention.py runs inside its loop; here it
  is between two tiles, which is where a backend can get it wrong without the
  loop hiding it.

WHY THIS AND NOT JUST dl/attention.py
  That kernel streams and rescales as it goes, so a bug in the rescale shows up
  mixed with the loop. This one does exactly one merge, of two halves that are
  each a plain softmax, so a wrong answer is the merge and nothing else.

THE TWO HALVES ARE DELIBERATELY UNBALANCED in scale, so m1 and m2 differ by
enough that dropping the e^(m-m) factor is visible rather than a rounding.

NO DRAM ROUND TRIPS, AND IT ONCE CARRIED 8 WITH 2 POINTLESS (rule 8b). The two
had `torchsim.relayout = array<i64: a, a>` -- both ends the same lane axis, so a
tile went out to DRAM and came back banked exactly as it left, buying nothing.
It also used to die in stage `lower` on a `math.exp` left on a scalar, which is
why nothing annotated the count. Measured today: it passes, 0 trips here and 0
pointless anywhere in the suite; tests/baselines/tpuv6e.txt is the tripwire.

Compared against a softmax over the CONCATENATION, so the merge is checked against the thing it claims to equal rather than against itself. Six scalar reductions feed one rescale.
"""

#: WHY THIS REFUSES, TRACED 2026-09-23, AND IT USED TO PASS. A full fold of a
#: rank-1 tile is constrained to one bank (`AddReduceConstraints`), and that
#: constraint used to stop at `IsAtMostRank1` and only ever reach a COPY made for
#: the reduce -- buf1 stayed spread and the generic that writes it was happy.
#: With the premise corrected the one-bank answer reaches buf1 itself, and the
#: generic then reads a 32-lane tile into a one-bank destination.
#:
#: Making the full-fold arm non-mandatory does NOT move it (measured), so the
#: copy is not being declined for want of standing. The disagreement is between
#: the graph's answer for the operand's version and what `StampLayouts` wrote on
#: the global -- that seam is where the next round starts.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, D = 32, 16                          # keys per half, head dim


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def flash_decode_splitk_kernel(q_ptr, k1_ptr, v1_ptr, k2_ptr, v2_ptr,
                                   o_ptr, S: tl.constexpr, D: tl.constexpr):
        rs = tl.arange(0, S)
        rd = tl.arange(0, D)
        q = tl.load(q_ptr + rd)
        off = rs[:, None] * D + rd[None, :]
        scale = 1.0 / (D ** 0.5)

        # Half 1, normalised by its OWN max.
        s1 = tl.sum(tl.load(k1_ptr + off) * q[None, :], axis=1) * scale
        m1 = tl.max(s1, axis=0)
        e1 = tl.exp(s1 - m1)
        l1 = tl.sum(e1, axis=0)
        o1 = tl.sum(e1[:, None] * tl.load(v1_ptr + off), axis=0) / l1

        # Half 2, likewise -- nothing shared with half 1 yet.
        s2 = tl.sum(tl.load(k2_ptr + off) * q[None, :], axis=1) * scale
        m2 = tl.max(s2, axis=0)
        e2 = tl.exp(s2 - m2)
        l2 = tl.sum(e2, axis=0)
        o2 = tl.sum(e2[:, None] * tl.load(v2_ptr + off), axis=0) / l2

        # The merge: rescale both to the joint max before they may be added.
        m = tl.maximum(m1, m2)
        w1 = l1 * tl.exp(m1 - m)
        w2 = l2 * tl.exp(m2 - m)
        tl.store(o_ptr + rd, (o1 * w1 + o2 * w2) / (w1 + w2))

    return flash_decode_splitk_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda s=1.0: torch.randn(S, D, generator=g, dtype=torch.float32) * s
    # Unbalanced on purpose: the two maxima must not coincide.
    return {"q": torch.randn(D, generator=g, dtype=torch.float32),
            "k1": r(1.0), "v1": r(), "k2": r(3.0), "v2": r()}


def reference(inputs):
    import torch
    k = torch.cat([inputs["k1"], inputs["k2"]], dim=0)
    v = torch.cat([inputs["v1"], inputs["v2"]], dim=0)
    s = (k @ inputs["q"]) / (D ** 0.5)
    return {"o": s.softmax(dim=0) @ v}


SPEC = KernelSpec(
    name="flash_decode_splitk_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "k1_ptr": "*fp32", "v1_ptr": "*fp32",
               "k2_ptr": "*fp32", "v2_ptr": "*fp32", "o_ptr": "*fp32",
               "S": "constexpr", "D": "constexpr"},
    constexprs={"S": S, "D": D},
    args=[Arg("q", IN, "float32", (D,)),
          Arg("k1", IN, "float32", (S, D)), Arg("v1", IN, "float32", (S, D)),
          Arg("k2", IN, "float32", (S, D)), Arg("v2", IN, "float32", (S, D)),
          Arg("o", OUT, "float32", (D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
