"""A bitonic network written out, so the compare-exchange is visible in the IR.

torch equivalent: x.sort(dim=1).values, same as sort_row_tl.py

WHY FOUR LAYOUT OPS STAND AFTER absorb_layout_ops, measured 2026-09-02. They are the
`tensor.reshape`s of the exchange above, and neither direction takes them.

  FORWARDS IS PROVED IMPOSSIBLE, not merely refused. The reshape 128x8x2 ->
  128x2x2x4 preserves the linear index, and the `extract_slice` under it takes
  [0,0,0,0] sizes [128,2,1,4]. Written in the OPERAND's 128x8x2 axes those
  elements are axis-1 positions {0, 1, 4, 5}, which is not an arithmetic
  progression, and a box is `offset + k*stride` per axis by definition. The
  slice cannot state it, so `_restated_slice` is right to say no.

  BACKWARDS THE FIRST STEP WORKS AND THE GROUP DOES NOT. Restating the
  `insert_slice` in 128x2x2x4 gives offsets [0,0,0,1] sizes [128,2,2,2] strides
  [1,1,1,2] -- a box. But its `%src` is typed by those non-unit sizes, so the
  select chain above it joins the group, and the group then reaches the other
  end's `extract_slice`, which is the forward case again. Five attempts, all
  refused further up:

      %src walked into the group           fires 27 times, refused 27 times
      the rank-reducing lift on the slice   condition matches, refusal moves
                                            from the length check to _restate_box
      try linalg.generic [8] -> [2, 2, 2]   a rank-1 operand splitting into 3

  So this is not "nobody has tried"; it is a group with at least three walls
  that have to fall together. AND IT WOULD NOT OPEN THIS KERNEL ANYWAY: the
  `expect` below is about a unit stride at `vector.load`, not about layout.


WHY SPELL IT OUT WHEN tl.sort EXISTS
  `tl.sort` is one op to this backend and expands inside triton, so a failure
  there says "sort did not work" and stops. Here every stage is separate IR, so
  a failure names the stride and the direction it happened at. When the two
  disagree, the difference is the network triton chose versus this one.

HOW THE EXCHANGE IS EXPRESSED
  A register tile cannot be indexed dynamically, so `x[i ^ stride]` has nowhere
  to come from. The pairs are made ADJACENT instead: reshape the row so the two
  partners land in the last axis, `tl.split` them apart, take min and max, and
  `tl.join` them back. That is the same data movement, said in ops that exist.

      [M, N]  ->  [M, N/2s, 2, s]  ->  permute  ->  [M, N/2s, s, 2]  ->  split

  THE 4D TILE IS THE POINT OF THE PROBE. Every other kernel here works in one or
  two dimensions, because the DMA descriptor and the lane banking are built for
  that. A bitonic network cannot be written without a third and fourth axis
  unless the data goes back to memory between stages, so this asks whether the
  backend can hold one -- and if it refuses, that refusal is the finding, and
  the alternative is a kernel launch per stage.

DIRECTION. A bitonic merge alternates ascending and descending by block, which
is `(i & size) == 0` on the block index. Descending blocks swap the min/max
assignment rather than reversing the data.

GREEN SINCE 2026-09-04, max_abs_err 0, and it took two fixes stacked.

FIRST WALL: `tl.split` leaves a rank-reducing subview whose minor axis steps two,
and vectorize asked its BANK strides rather than the type before emitting a
wide load -- see hadamard_rotate, which is the same butterfly and closed on that
alone. This one got past it and answered a PERMUTATION of the input with every
run of four sorted and the two widest merges not done.

SECOND WALL, found by truncating the network to its first k stages: 1..8 err 0,
1..9 err 5.03383. Stage 9 is right on its own, so is the pair (8, 9), and so is
every suffix from 6..9 -- only a long prefix followed by stage 9 was wrong.
Writing the stage-8 intermediate to a second output made the same nine stages
correct, which is what a buffer reused while still live looks like.

It was `select_lane_axis._merge_relayouts`. Two relayouts of one buffer to one
layout are one copy, and the cache is dropped when an intervening op touches the
source -- but the walk that answers "which buffer is this" stopped at a
`memref.subview`, so the two `linalg.generic`s that WRITE the source between the
stage 4->5 boundary and the stage 8->9 one were invisible. The copy made for the
first was handed to the second across forty-two ops and two writes.

Rows sorted in place: the unconstrained tile takes upstream's default layout, whose most major axis -- the rows -- rides the lanes, so the network's reshapes never move an element between lanes.

LAYOUT PIN. `expect_ir` holds `torchsim.copy` in 08-passes/04-torchsim-materialize-layouts.mlir at 0; what that pins is the two-round anchoring (reshape seeds settle before the reduces are pinned) and the first-non-unit-axis default; with the reduces pinned first 9.
"""

#: WHY THIS REFUSES, TRACED 2026-09-23. An `expand_shape` splits the lane axis
#: of `graphcopy13_spad_8lane` into several axes, so no axis of the view is the
#: lane axis and lower_to_dma has nothing to read. The written-reshape bridge is what
#: answers that -- one leg in, one leg out -- and it does not fit here: the
#: view is live over [72, 217] and four `linalg.generic` and a `linalg.transpose`
#: reach the same storage inside that range, so the return leg cannot stand in
#: one place. Two legs need the storage reached through this view alone while
#: it is live; here it is not, and this says so rather than guessing.
#:
#: THE PER-USE TRICK DOES NOT CARRY OVER FROM SLICES, measured the same day.
#: `LowerCrossingSlices` needs no live range because it copies in front of each
#: USE, and a subview has one use each. A reshape does not: `collapse -> expand`
#: chains are common, and repointing one use leaves the chain inconsistent
#: (ops_shape: pass -> lower refusal), while repointing the view users too gives
#: WRONG NUMBERS (ops_shape 14.09, this kernel 4.10). A view of a materialised
#: buffer is not the view it replaced.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 16


def kernel():
    import triton
    import triton.language as tl

    # ONE STAGE, as its own jit function taking the geometry as constexpr
    # PARAMETERS. Everything else was tried and does not work: a `while` is a
    # runtime loop in triton, so the stride it carries becomes a tensor and
    # tl.reshape rejects it as a shape element; a jit body cannot read a module
    # global; and iterating a list literal is not something the AST walker
    # handles. A constexpr parameter IS accepted as a shape element, so the ten
    # stages of a 16-wide network are ten calls with literal geometry.
    @triton.jit
    def _stage(x, M: tl.constexpr, g: tl.constexpr, stride: tl.constexpr,
               size: tl.constexpr):
        # Bring the two partners into the last axis: [M,g,2,s] -> [M,g,s,2].
        y = tl.reshape(x, [M, g, 2, stride])
        y = tl.permute(y, (0, 1, 3, 2))
        lo, hi = tl.split(y)
        small = tl.minimum(lo, hi)
        large = tl.maximum(lo, hi)
        # Ascending where this block's `size` bit is clear. Every element of a
        # block shares that bit because size >= 2*stride always holds here.
        blk = tl.arange(0, g)[None, :, None]
        up = ((blk * (2 * stride)) % (2 * size)) < size
        y = tl.join(tl.where(up, small, large), tl.where(up, large, small))
        y = tl.permute(y, (0, 1, 3, 2))
        return tl.reshape(y, [M, g * 2 * stride])

    @triton.jit
    def sort_bitonic_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
        x = _stage(x, M, 8, 1, 2)
        x = _stage(x, M, 4, 2, 4)
        x = _stage(x, M, 8, 1, 4)
        x = _stage(x, M, 2, 4, 8)
        x = _stage(x, M, 4, 2, 8)
        x = _stage(x, M, 8, 1, 8)
        x = _stage(x, M, 1, 8, 16)
        x = _stage(x, M, 2, 4, 16)
        x = _stage(x, M, 4, 2, 16)
        x = _stage(x, M, 8, 1, 16)
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], x)

    return sort_bitonic_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].sort(dim=1).values}


SPEC = KernelSpec(
    name="sort_bitonic_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE DEFAULT IS THE CLAIM: upstream's descending layout puts the lanes on
    # the rows, so every exchange stays in its lane. On the sorted axis: 13 trips.
    expect_ir={"08-passes/04-torchsim-materialize-layouts.mlir": {"torchsim.copy": 0}},
)
