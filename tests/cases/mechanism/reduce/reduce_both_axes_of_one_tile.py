"""One tile reduced along BOTH of its axes, so neither can carry the lanes.

A FOLD NOW. The axis on the lanes folds through the cross-lane unit (xlu_push_reduce and its pop): the
pin is that one fold move.

WHAT IT IS FOR. `linalg.reduce` says "this axis must not be the lane axis": folding
it would move elements between lanes and no instruction did that. Fold both axes of
one tile and the demands leave nothing, so `axis_of` answers ONE_LANE and the whole
tile sits in ONE bank of 256 -- correct, and 1/256 of the machine, with nothing above
the reference check saying a word. 121 of the suite's 304 ONE_LANE stamps are struck
by exactly this demand and no other.

THE SHAPE. `x` is loaded once and reduced twice, and the two results go to two
DIFFERENT outputs on purpose -- adding them into one makes a `linalg.generic` write
the accumulator a second time, and `cross_lane_reduce_of` declines a reduce whose
`outs` anyone but its identity fill writes:

    sum(x, axis=0) -> c       -> NOT on axis 0 of x
    sum(x, axis=1) -> w       -> NOT on axis 1 of x

The two demands do not CONTRADICT each other -- each is satisfiable alone -- and
that is why this is not a conflict `_plan_cuts` can see. It is a tile with no
answer, and the pass takes the only one left.

WHAT THE CROSS-LANE UNIT DOES WITH IT. The demand stops being a prohibition and
becomes a price: one axis carries the lanes and its fold is IN-LANE, and the other
is answered by a TRANSPOSE that moves the lane onto the first -- after which that
fold is in-lane too. The unit moves the bits; the arithmetic is the lane's.

`expect_ir` counts those instructions, as the `vcix.*` DIALECT ops stage 5 mints:
`vcix.iv` is the push and `vcix.v.i` the pop, ONE op each run twice -- the
transpose's two rows, one per iteration of the loop each is emitted in. Spelt
as an `llvm.call_intrinsic` naming the intrinsic -- which is what this backend
emitted until the dialect carried the side-effecting ops -- both counts are 0
here while the assembly is unchanged, so only the artifact tells them apart.

IT WAS FOUR WHILE THE CROSSBAR FOLDED THIS. Two of those were the lane-index read
the push masked with: a unit that reduces over EVERY lane has to be handed the
identity in the ones a 32-wide tile does not reach. A fold inside a lane walks the
depth it actually has, so there is no lane to pad and the two reads are gone --
which is the whole of what retiring the crossbar's reduce bought here.

WHAT WOULD BREAK IT. Anything that puts the reduce's NOT back on an axis
fit_layout_to_lanes can rewrite, or that mints a trip lower_to_vcix declines: both
land as a vcix count of 0 and a tile back in one bank. The VALUES do not move
either way -- a sum is a sum -- which is the whole reason the claim is on the IR
(rule 2 wants a kernel that fails without the feature, and only the artifact can
fail here).

WHY THE TILE COMES OUT OF A LOOP. Loaded and summed straight away, each sum gets its
own load in its own layout -- a DMA lays a tile out any way, and upstream's
remove-layout-conversions rematerializes the load rather than convert -- and nothing
crosses a lane. A loop result is where that rematerialization stops, so the tile
the two sums read is ONE tile and the second fold needs the transpose.

WHY 32 AND NOT 16. The lane axis has to hold more than one bank's worth to make
"in one bank" different from "banked", and 32 rows over 32 lanes is the smallest
square that says so while keeping three chained sums exact in f32.

Two reduce demands on one tile, one per axis. Neither contradicts the other, so nothing reports a conflict -- the tile simply has no axis left and lands in one bank. The transpose is what turns the second fold from a prohibition into an instruction.
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def both_axes_of_one_tile_kernel(x_ptr, c_ptr, w_ptr, S: tl.constexpr):
        r = tl.arange(0, S)
        x = tl.load(x_ptr + r[:, None] * S + r[None, :])
        acc = tl.zeros((S, S), tl.float32)
        for _ in range(2):
            acc += x
        tl.store(c_ptr + r, tl.sum(acc, axis=0))   # folds axis 0 -- NOT on axis 0
        tl.store(w_ptr + r, tl.sum(acc, axis=1))   # folds axis 1 -- NOT on axis 1

    return both_axes_of_one_tile_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(S * S, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = 2 * inputs["x"].reshape(S, S)
    return {"c": x.sum(dim=0), "w": x.sum(dim=1)}


SPEC = KernelSpec(
    name="both_axes_of_one_tile_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "c_ptr": "*fp32", "w_ptr": "*fp32",
               "S": "constexpr"},
    constexprs={"S": S},
    args=[Arg("x", IN, "float32", (S * S,)),
          Arg("c", OUT, "float32", (S,)), Arg("w", OUT, "float32", (S,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
    expect_ir={"08-transformed.mlir": {"pytorchsim.fold": 1}},
)
