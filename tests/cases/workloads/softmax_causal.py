"""Causal masked softmax over a ragged tile -- the softmax chain with every
boundary condition an attention block actually has.

torch equivalent, for a score tile s of shape (BM, BN):

    s = s.masked_fill(~(causal & valid), -inf)
    o = softmax(s, dim=1)                     # rows with no valid column -> 0

WHAT THIS ADDS OVER softmax.py.  softmax.py is the clean chain. This one carries
the three things that make a real attention epilogue harder, and carries them
TOGETHER, because separately each is already covered elsewhere and their
interaction is not:

  a 2-D mask built from two aranges     `offs_m[:, None] >= offs_n[None, :]`
                                        is a cmp between two tiles that were
                                        broadcast along OPPOSITE axes -- one
                                        along the lanes, one inside a lane
  an `&` of that with a 1-D bound       `n < N_VALID`, broadcast the other way,
                                        so MaskAnalysis sees an `andi` whose two
                                        arms disagree about which axis they pin
  a masked load with `other`            the tile is ragged in BOTH axes, so the
                                        load takes a mask and the padding value
                                        has to be the reduction's identity or
                                        the row max is wrong

The reduction is then over a tile that is PARTLY DEFINED, which is where the
padding value stops being cosmetic: `tl.max` over a lane whose tail holds zeros
instead of -inf returns 0 for a row whose real values are all negative. The
`other=-inf` on the load and the `tl.where` after it are two different fixes for
two different holes -- the load's is for columns past N_VALID, the where's is for
columns above the diagonal -- and both are needed.

DEGENERATE ROWS ARE PART OF THE TEST.  With N_VALID = 100 and a causal mask,
row 0 keeps exactly one column and rows >= 100 keep 100. No row is fully masked,
which would make the reference NaN; the test is the ragged edge, not the 0/0.

Complex-op tier: a 2-D mask from two opposite-axis broadcasts, anded with a 1-D bound, feeding a masked load and two reductions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM = 128         # rows == PYTORCHSIM_COMPILE_VECTORLANE_SIZE
BN = 128         # columns, in-lane
N_VALID = 100    # ...of which only this many are real. NOT a power of two.
NEG = -1.0e30    # a finite stand-in for -inf: exp() of the difference is 0,
                 # and it survives the arithmetic that a true -inf turns to NaN.


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def softmax_causal_kernel(s_ptr, o_ptr, BM: tl.constexpr, BN: tl.constexpr,
                              N_VALID: tl.constexpr, NEG: tl.constexpr):
        rows = tl.arange(0, BM)
        cols = tl.arange(0, BN)
        idx = rows[:, None] * BN + cols[None, :]

        # Two masks pinned to opposite axes, then combined.
        causal = rows[:, None] >= cols[None, :]
        valid = cols[None, :] < N_VALID
        keep = causal & valid

        s = tl.load(s_ptr + idx, mask=keep, other=NEG)
        s = tl.where(keep, s, NEG)

        m = tl.max(s, axis=1)
        e = tl.exp(s - m[:, None])
        e = tl.where(keep, e, 0.0)
        o = e / tl.sum(e, axis=1)[:, None]
        tl.store(o_ptr + idx, o)

    return softmax_causal_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"s": torch.randn(BM * BN, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    s = inputs["s"].reshape(BM, BN)
    rows = torch.arange(BM)[:, None]
    cols = torch.arange(BN)[None, :]
    keep = (rows >= cols) & (cols < N_VALID)
    s = torch.where(keep, s, torch.full_like(s, NEG))
    e = torch.exp(s - s.max(dim=1, keepdim=True).values)
    e = torch.where(keep, e, torch.zeros_like(e))
    return {"o": (e / e.sum(dim=1, keepdim=True)).reshape(-1)}


SPEC = KernelSpec(
    name="softmax_causal_kernel",
    kernel=kernel,
    signature={"s_ptr": "*fp32", "o_ptr": "*fp32", "BM": "constexpr",
               "BN": "constexpr", "N_VALID": "constexpr", "NEG": "constexpr"},
    constexprs={"BM": BM, "BN": BN, "N_VALID": N_VALID, "NEG": NEG},
    args=[
        Arg("s", IN, "float32", (BM * BN,)),
        Arg("o", OUT, "float32", (BM * BN,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-5, atol=1e-6,
)
