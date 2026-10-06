"""A [N, 2, 1] tile reduced against a 2-element computed row. Two defects, both fixed.

IT WAS RED FOR THE SECOND OF THEM. It was written to pin the candidate filter
that `A constant in the outs map is not a dim the loop failed to name` fixed,
and it does -- but with that fix in place the kernel compiled and then got 381
of its 2048 elements wrong. That was a SECOND defect the first had been hiding,
and it stood here undiagnosed until legalize's widening FLOOR was asked
the right question. See "WHAT WAS STILL WRONG" below.

WHAT IT PINS (the fixed half). `_map_broadcast_pins` refuses an op whose outs
does not span the iteration space, which is the right question for a reduction.
It asked it as a set equality over `_map_dims`, and `_map_dims` answers None for
a constant, so an outs read `(d0, d1) -> (d0, d1, 0)` -- a rank-3 tile with a
unit tail -- put None in the set and was dropped as if it were a reduction. The
2-element row beside it was then never offered for staging, and vectorize
refused the op five passes later:

    one operand of this elementwise op is in a single bank (ONE_LANE) while
    another is banked across the lanes

    measured   on a41b6d8~1 this kernel stops there. With a41b6d8 it clears
               adapt, lower, binary and runs.

WHAT WAS STILL WRONG: THE WIDENING FLOOR WROTE PAST THE ROW IT WAS GIVEN. Every
in-lane extent here is 2 -- the tile is [N, 2] and [N, 2, 1] -- and MIN_VEC is
8, so legalize raised each elementwise loop from `0 to 2 step 2` to
`0 to 8 step 8`:

    affine.vector_store %v, %buf[%d, %j] : memref<2048x2xi32, 1>, vector<8xi32>

Eight consecutive i32 from `[d, j]` in a buffer whose minor dim is TWO are rows
d..d+3. Each store therefore overwrote the three rows after it, and what
survived was whatever a later store happened to put back -- 381 of 2048 in the
gaps. The banking was never the problem: `buf8_spad_2048lane[16]` is 16 elements
per lane over 128 lanes, which is 2048, and all nine `torchsim.lane_axis` stamps in
04-transformed.mlir read 0.

WHY THE GUARD DID NOT FIRE. The floor exists because a short tile owns a whole
vector per lane -- spad.per_lane rounds the reservation up to MIN_VEC and the
DMA never reads the surplus back -- and it was declined only for a STRIDED VIEW,
read off `strides[-1] != 1`. This buffer is contiguous, so the stride said
nothing, and the extent is what was short. `_owns_nothing_past_its_extent` now
asks the question both shapes are: how far may a vector run from a position on
the minor dim before it reaches the next thing in the SAME buffer. A strided
view answers zero; a minor dim of 2 under an outer dim of 2048 answers two.

IT IS NOT reduce/reduce_lane_broadcast_across_loop.py (develop-e2e-bert), which
looks like the same family -- a lane-broadcast operand inside a reduction -- and
is not. That one is a lane-axis DISAGREEMENT: its generics move to dim 1 while
the linalg.fill that zeroes the accumulator stays at dim 0, so a loop carries a
buffer in banked on one axis and yields it banked on another (d30d73c names the
chain). There is no loop carry here to disagree across.

THE SHAPE, and every part of it is load-bearing:

    y = reshape(x, (N, 2, 1))     the unit tail is what the filter dropped
    m = reshape(1 - c, (1, 2, 1)) a row COMPUTED on chip, so `widen the load`
                                  (menu option 1) is unavailable and the
                                  replicating round trip is the only answer
    sum(y * m, 1)                 the row meets a lane-banked tile, which is
                                  the contradiction vectorize names

Drop the unit tail and the op is an ordinary rank-2 one that was never filtered;
load the row instead of computing it and option 1 answers it for free.

torch equivalent: (x[:, 0] - x[:, 1]) over an [N, 2] input.

rank-3 tile with a unit tail reduced against a computed 2-element row. Pins two things: the candidate filter that offers the row for staging, and legalize's widening floor, which may not raise a loop past what its buffer holds at that position. Every extent here is 2 against a MIN_VEC of 8, so a floor that ignores the buffer writes over the next three rows.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 2048
W = 2
XN = N * W


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_unit_tail_axis_row_reduce_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                              W: tl.constexpr):
        r = tl.arange(0, N)[:, None]
        c = tl.arange(0, W)[None, :]
        x = tl.load(x_ptr + (r * W + c))
        y = tl.reshape(x, (N, W, 1))                 # [N, 2, 1], unit tail
        m = tl.reshape((1 - c).to(tl.int64), (1, W, 1))   # computed 2-element row
        s1 = tl.sum(y * m, 1)
        s2 = tl.sum(y * tl.reshape(c.to(tl.int64), (1, W, 1)), 1)
        tl.store(o_ptr + tl.arange(0, N)[:, None], s1 - s2)

    return tile_unit_tail_axis_row_reduce_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(-50, 50, (XN,), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(N, W).to(torch.int64)
    y = x.reshape(N, W, 1)
    c = torch.arange(W).to(torch.int64).reshape(1, W, 1)
    s1 = (y * (1 - c)).sum(1)
    s2 = (y * c).sum(1)
    return {"o": (s1 - s2).reshape(-1).to(torch.int32)}


SPEC = KernelSpec(
    name="tile_unit_tail_axis_row_reduce_kernel", kernel=kernel,
    signature={"x_ptr": "*i32", "o_ptr": "*i32",
               "N": "constexpr", "W": "constexpr"},
    constexprs={"N": N, "W": W},
    args=[Arg("x", IN, "int32", (XN,)), Arg("o", OUT, "int32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
