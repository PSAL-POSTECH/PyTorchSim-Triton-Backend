"""`tl.sum(x)` on a 1-D tile: the smallest fold with nowhere to put the lane.

WAS A REFUSAL, AND THE REASONING WAS WRONG. This file used to live in reject/
and say that the fold is impossible because element i lives in lane i and
nothing moves a value between banks. The first half is true and the second is
true, but they do not add up to this kernel: it is not a reduction ACROSS the
banks, it is a reduction with no axis left over to put a lane on.

Those are different things and they have different answers. A tile whose values
are genuinely scattered -- one axis forced onto the lanes by a matmul while a
reduce folds that same axis -- is answered by a DRAM copy (fit_layout_to_lanes's
_relayout_conflicts), not by refusing. A rank-1 fold has no axis to move at all,
and the answer is to stop spreading the tile: lib_lane_axis.ONE_LANE puts all 128
elements in ONE bank, and the fold is then the ordinary in-lane one that every
rank-2 axis-1 reduction here already compiles to.

WHAT IT COSTS, AND WHY THAT IS THE PRICE
  One lane does the work and 127 idle. That is what a value every lane would
  need costs on a machine whose parallelism IS the lanes, and it is not an
  oversight: the alternative is not a faster lowering, it is no lowering.

  A rank-2 `tl.sum(x, axis=1)` pays none of it -- each lane owns a whole row and
  they fold at once. Which is the useful advice this file carries: reduce along
  the innermost axis of a 2-D tile, not along a 1-D one.

WHY THE RESULT NEVER BECOMES A SCALAR
  `tensor.extract` on the rank-0 result would put it in a scalar register, and
  there is ONE scalar register file for all the lanes -- `vfmv.f.s` is refused
  by name in pipeline.py. The value stays a one-element tile in lane 0 and
  leaves through a one-element MVOUT, so no scalar ever exists. That is also
  why the sibling case is still refused: ops_math_elementwise splats its
  reduction result back over a tile, so every lane needs it, and that needs a
  replicating DRAM round trip nothing builds yet.

WHAT THE expect_ir PINS, AND WHY IT IS THE `.ll`
  p11's reserve_spad_storage sizes each global as ONE lane's share and reads the
  extent out of the symbol name, because the LLVM dialect has lost
  `torchsim.dim_order` by then. `06-kernel.ll` is the last text the toolchain
  reads, so it is where the reservation is settled. The fallback -- the narrowest
  dim of the type -- is safe for every tile that rides an axis, since the lane
  axis IS one of the dims. ONE_LANE is the one case it cannot state: the extent
  is 1 and that is no dim at all. Sized as
  `narrow=128 -> 1 element per lane`, this tile got 8 bytes, the accumulator
  global landed at `buf1_spad + 8`, and `sw zero` wrote over element 2. A
  one-hot sweep of all 128 positions lost exactly index 2; all-ones summed 127.
  Every IR level was arithmetically correct, so only the storage says it.

torch equivalent: x.sum().reshape(1)

A rank-1 fold: no axis left to put a lane on, so the whole tile goes in ONE bank and the fold is an ordinary in-lane one. 127 lanes idle -- reduce the inner axis of a 2-D tile instead.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_lane_axis_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + r)
        tl.store(o_ptr + tl.arange(0, 1), tl.sum(x, axis=0))

    return reduce_lane_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BLOCK, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].sum().reshape(1)}


SPEC = KernelSpec(
    name="reduce_lane_axis_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (BLOCK,)), Arg("o", OUT, "float32", (1,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # ONE LANE HOLDS THE WHOLE TILE, so its bank reserves all 128 elements. The
    # load lands across the lanes and one XLU crossing gathers it in. The claim
    # names no buf number: allocation order moves, one lane's 128 floats do not.
    expect_ir={"06-kernel.ll": {"global [128 x float] zeroinitializer": 1}},
)
