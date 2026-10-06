"""A reduction loop whose body holds a GATHER indexed along the reduced axis.

IT FAILS, AND IT CARRIES NO `expect` MARKER, WHICH IS THE POINT. `blocked`
and `refuse` are for a program that never runs -- the runner says so itself:
"every stage produced a program and the ANSWER is what failed, which is a
wrong number, not a refusal." This one compiles clean through all eight
stages and returns garbage, so it stands in the sweep as a red kernel until
the edge below goes in. The fix moves every kernel's layout, so it is not
made here; this file is the record of what to fix and how to know it worked.

The gather's index is a rank-1 vector of length R_BLOCK -- the reduction
extent -- so it can only be banked on its own axis, and `linalg.broadcast
dimensions = [0]` carries that onto every 2-D tile in the loop body: lane
axis 1, the axis being reduced. The accumulator that crosses the loop's back
edge, and the `linalg.reduce dimensions = [1]` after it, want lane axis 0.
Both are reachable; nothing makes them agree.

    memref<128x512xf32>   axis=0    the accumulator's init
    scf.for result        axis=0
    memref<128x512xi64>   axis=1    the gather's index tile
    memref<128x512xf32>   axis=1    the loop body, INCLUDING what it yields

One storage, two layouts, no relayout between them. It compiles and every
element is wrong -- 32 of 32 here, max_abs_err 65.6.

WHY THE GRAPH LETS IT. lib_axis_graph's own docstring names the edge that
would forbid this:

    scf.for / while   position.  init i, block argument i, result i and the
                      yielded value i are the same buffer seen four times.

`scf` appears in that file ONCE, in that sentence. `_maps_of` has no arm for
`scf.for` or `scf.yield`, so both fall through to `return None` -- no edge --
and the two sides of the back edge stay in different axis classes. Measured on
this kernel's `04-passes/07-torchsim-bufferize.mlir`: 22 rank-2 buffers, split 0/1
as above,
`merges()` 0 and `contradictions()` 0, so nothing reports it either. Union the
four positions by hand and all 22 resolve to axis 0, which is the right answer
-- x on the lanes, the reduction in-lane.

reduce_general_over_loop_partials.py pins the same identity one pass over, in
p04_bufferize, which reads three of the four spellings. This reads none.

WHERE IT CAME FROM. Stable Diffusion v1.5's UNet, first executed kernel,
`triton_npu_fused_convolution_native_group_norm_3` -- GroupNorm's Welford mean
over a conv output plus its per-channel bias. The bias load is the gather
(`r0_index // 1024`, a non-affine index), the Welford accumulators are what
crosses the back edge. 32 of 32 channels wrong, all of them equal to the mean
over the WHOLE tile rather than over their own row.

WHAT IS NOT WRONG, because it was checked element by element under SPIKE_DEBUG:
the gather itself. Its index values are exact (0,0,1,1 over the four loop
trips), the channel term rides the descriptor correctly as
`p_mm_stride = (0,0,2,0)`, and the values that land in the scratchpad are the
right ones in the right banks. The DMA does its job; the layout it lands in
disagrees with the layout the reduction reads.

WHAT WOULD BREAK THIS KERNEL -- i.e. what it is guarding
  * `a`'s x term is what makes the failure visible. Drop it and every row
    holds the same value, so collapsing the x axis returns the right answer
    and the kernel goes green while pinning nothing. That was measured: the
    probe with the x dependence removed passes on the unfixed tree.
  * the x dependence must come through `a`, the DIRECT load, not through the
    gather. Then the failure cannot be read as a gather addressing bug -- the
    gather's mere PRESENCE is what moves the lane axis.
  * K < R_BLOCK is what makes `r // K` non-affine within a block, which is
    what makes it a gather rather than a strided load.
  * THE 2-D MASK. Without it `b`'s gather lowers to `memref<1x256xf32>` -- one
    row, broadcast afterwards -- and the broadcast's axis 1 never reaches the
    accumulator, so the kernel PASSES and pins nothing. That was measured: the
    first draft of this file was unmasked and went green. Inductor's kernel
    carries `r0_mask & xmask`, which is what makes the gathered tile 2-D there.
  * the accumulator must cross an scf.for back edge. Fold the reduction in one
    tile and there is no back edge for the two classes to disagree across.

A gather indexed along the reduced axis banks the loop body on axis 1 while the accumulator crossing the scf.for back edge stays on axis 0. lib_axis_graph._maps_of has no scf.for arm, so the two never meet. Reduced from Stable Diffusion v1.5 GroupNorm.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

X = 128             # the row BLOCK -- the axis that must stay on the lanes
XLIVE = 32          # < X, so the mask is real and the gather's tile is 2-D
R = 1024            # the reduced axis
R_BLOCK = 256       # so the loop really has a back edge (4 trips)
K = 128             # < R_BLOCK, so `r // K` is non-affine inside one block
NB = R // K         # the gathered table


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_loop_gather_index_lane_axis_kernel(
            a_ptr, b_ptr, o_ptr, X: tl.constexpr, XLIVE: tl.constexpr,
            R: tl.constexpr, R_BLOCK: tl.constexpr, K: tl.constexpr):
        x = tl.arange(0, X)[:, None]
        xm = x < XLIVE
        acc = tl.zeros([X, R_BLOCK], tl.float32)
        for r_off in range(0, R, R_BLOCK):
            r = r_off + tl.arange(0, R_BLOCK)[None, :]
            m = xm & (r < R)
            # DIRECT: this is where the row dependence comes from.
            a = tl.load(a_ptr + (r + R * x), m, other=0.0)
            # GATHER: a non-affine index built from the REDUCED axis alone. The
            # 2-D mask is what makes the gathered TILE 2-D (128 x R_BLOCK)
            # rather than one broadcast row -- see the note below.
            b = tl.load(b_ptr + (r // K), m, other=0.0)
            acc += a + b
        xi = tl.arange(0, X)
        tl.store(o_ptr + xi, tl.sum(acc, 1), xi < XLIVE)

    return reduce_loop_gather_index_lane_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(XLIVE, R, generator=g, dtype=torch.float32),
            "b": torch.randn(NB, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = torch.arange(R)
    return {"o": (inputs["a"] + inputs["b"][r // K]).sum(dim=1)}


SPEC = KernelSpec(
    name="reduce_loop_gather_index_lane_axis_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "X": "constexpr", "XLIVE": "constexpr", "R": "constexpr",
               "R_BLOCK": "constexpr", "K": "constexpr"},
    constexprs={"X": X, "XLIVE": XLIVE, "R": R, "R_BLOCK": R_BLOCK, "K": K},
    args=[Arg("a", IN, "float32", (XLIVE, R)),
          Arg("b", IN, "float32", (NB,)),
          Arg("o", OUT, "float32", (XLIVE,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
