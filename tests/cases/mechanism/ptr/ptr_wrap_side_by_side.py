"""A wrap on the COLUMN axis, and a clamp on the axis that does NOT carry it.

Split out of the old tests/cases/ptr_structured.py section E. The mirror image
of ptr_wrap_stacked.py: identical source shape, the `%` moved from the row term
to the column term. Reading the two side by side is the fastest way to see that
the wrap AXIS is chosen purely by which `shape:` slot PtrAnalysis filled --
`isSplitDimension(parentShape[0])` is tested first, so dim 0 wins if both were
non-zero, and only a zero in slot 0 gets you here.

WHAT THIS ONE PINS THAT THE STACKED FILE CANNOT
_cut_facts refuses when `_narrowed_axes(...) & wdims` is non-empty --
the SAME axis clamped and wrapped. Load 2 is clamped on the rows and wrapped on
the columns, so the intersection is empty and that guard must stay quiet. It
does: the transfer carries masked_axes = [0] and is refused for the other
reason instead (the wrap start `cols = pid * BN + rn` is runtime, so
_wrap_cut cannot count the laps). Without this file a guard written as "any
clamp anywhere" would pass every test.

COVERS (2)   convert_tts_to_pytorchsim.py

  wrap-offset-runtime-col   load a. Same refusal as ptr_wrap_stacked.py's load
      a, on the other axis: dram_stride comes out [24, 0] rather than [0, 1],
      because the axis inside the index gives up its stride and here that is
      axis 1.
  clamp-on-a-different-axis load b. masked_axes = [0], wrap on axis 1, and the
      same-axis guard does not fire.

WN is 24 and the block is 32 columns, so the seam is inside the tile for
program 0. ptr_wrap_side_by_side_run.py is the static-start sibling and does
get its runs.

The masked load has its OWN base argument (wcm_ptr) for a reason that no longer
exists -- see the note at the end of ptr_wrap_stacked.py.

Column-modulo wrap. Mirror image of ptr_wrap_stacked.py, and the one that pins the same-axis half of the clamp guard. Numerically runnable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

WN = 24          # wrap extent in COLUMNS -- smaller than BN, seam inside
BM = 16
BN = 32
NW = BM * WN
NO = BM * BN
ROWKEEP = 8      # the masked load's row bound


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_side_by_side_kernel(wc_ptr, wcm_ptr, o_ptr,
                                     WN: tl.constexpr, BM: tl.constexpr,
                                     BN: tl.constexpr):
        pid = tl.program_id(0)
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        cols = pid * BN + rn
        wc = cols % WN

        # 1 unmasked
        a = tl.load(wc_ptr + rm[:, None] * WN + wc[None, :])
        # 2 masked -- its OWN base; the meaningful bound is on the ROWS
        b = tl.load(wcm_ptr + rm[:, None] * WN + wc[None, :],
                    mask=(rm[:, None] < 8) & (rn[None, :] < BN), other=-2.0)

        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], a + b)

    return ptr_wrap_side_by_side_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"wc": torch.randn(NW, generator=g, dtype=torch.float32),
            "wcm": torch.randn(NW, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    wc = inputs["wc"].reshape(BM, WN)
    wcm = inputs["wcm"].reshape(BM, WN)
    cols = torch.arange(BN) % WN            # pid == 0
    a = wc[:, cols]
    b = torch.where(torch.arange(BM).reshape(BM, 1) < ROWKEEP,
                    wcm[:, cols], torch.tensor(-2.0))
    return {"o": (a + b).reshape(NO)}


SPEC = KernelSpec(
    name="ptr_wrap_side_by_side_kernel", kernel=kernel,
    signature={"wc_ptr": "*fp32", "wcm_ptr": "*fp32", "o_ptr": "*fp32",
               "WN": "constexpr", "BM": "constexpr", "BN": "constexpr"},
    constexprs={"WN": WN, "BM": BM, "BN": BN},
    args=[Arg("wc", IN, "float32", (NW,)), Arg("wcm", IN, "float32", (NW,)),
          Arg("o", OUT, "float32", (NO,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
