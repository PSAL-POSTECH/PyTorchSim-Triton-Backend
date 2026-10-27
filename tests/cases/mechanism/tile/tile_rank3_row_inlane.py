"""A LOADED plane broadcast along a rank-3 tile's MIDDLE axis, in lane.

    x = tl.load(...)              [M, K, N]   the tile, banked on axis 0
    b = tl.load(...)              [M, N]      one plane per lane
    y = x * b[:, None, :]                     b read as (d0, 0, d2)

WHAT MAKES THIS DIFFERENT FROM EVERY OTHER BROADCAST IN THE SUITE. With the
lanes on axis 0, lane L holds a whole K x N slab of the tile and a 1 x N slice
of `b`. So `b` DOES vary across the lanes -- and it also varies INSIDE one,
down axis 2. Neither of the two answers fit_layout_to_lanes had covers that:

    "the lanes are on the row's own axis"     true, and it is not enough
    "the lanes are on the axis it is         false, so nothing has to cross
     broadcast across"                        a bank and a round trip is waste

vectorize's splat reads ONE element per lane and repeats it down that
lane's vector, so an operand that varies inside the lane is refused by name:

    operand [8, 1, 16] broadcasts along dim 0 into a tile [8, 4, 16], and it
    was staged one element per lane

WHAT FIXES IT, AND IT IS FREE. `b` lives in DRAM, so it is read a SECOND time
with a zero stride on the broadcast axis -- one more MVIN of the same 128
elements, no MVOUT, nothing crossing a bank:

    pytorchsim.transfer(... memref<8x4x16xf32> ...) {direction = #pytorchsim<direction load>,
                                                 dram_stride = [16, 0, 1]}

WHAT BREAKS THIS KERNEL, three separate things and each on its own:

  the rank        _map_broadcast_pins refused `rank != 2`, so this operand was
                  never even offered to the staging menu. Restore that line and
                  the kernel dies with the message above.
  the guard       `laxis in pin` instead of `pin == [laxis]` calls this an
                  in-lane splat and does nothing. Rank 2 cannot tell the two
                  apart -- a rank-2 row has ONE axis -- which is why the wrong
                  one held for so long.
  the port        _clone_loaded_row_wide read a `tts.load` of a
                  `tts.make_tptr`, and neither survives convert_accesses_to_transfer
                  five passes above fit_layout_to_lanes. Without its memref
                  spelling the menu falls through to a DRAM round trip, and
                  _replicate_computed_row has no form for a rank-3 row either,
                  so there is nothing left and the message comes back.

DELIBERATELY LOADED AND NOT COMPUTED. The computed form of exactly this shape
is still refused -- see tile_rank3_computed_row_inlane.py, which pins it as
`blocked` -- because nothing can build a plane in lane that a reduction wrote.
The two kernels are the two halves of one gap and only one half is closed.

M IS THE LANE AXIS AND K IS SMALL ON PURPOSE. K = 4 keeps the in-lane extent
(K * N = 64) well inside one bank, so a failure here is the staging and not the
budget; N = 16 makes the plane wider than one element, which is the whole point
-- at N = 1 the operand would be constant inside the lane and the old guard
would have been right.

A loaded plane broadcast across a rank-3 tile's middle axis: the operand varies across the lanes AND inside one, so the splat is not free and no lane has to be crossed either. Staged by re-reading DRAM with a zero stride on the broadcast axis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 8, 4, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_rank3_row_inlane_kernel(x_ptr, b_ptr, o_ptr, M: tl.constexpr,
                                     K: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None, None] * (K * N)
                    + rk[None, :, None] * N + rn[None, None, :])
        # [M, N], read into the [M, K, N] tile as (d0, 0, d2): one plane per
        # lane, spread down the lane's own vector.
        b = tl.load(b_ptr + rm[:, None] * N + rn[None, :])
        y = x * b[:, None, :]
        tl.store(o_ptr + rm[:, None, None] * (K * N)
                 + rk[None, :, None] * N + rn[None, None, :], y)

    return tile_rank3_row_inlane_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * K * N, generator=g, dtype=torch.float32),
            "b": torch.randn(M * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(M, K, N)
    b = inputs["b"].reshape(M, 1, N)
    return {"o": (x * b).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="tile_rank3_row_inlane_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "K": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "K": K, "N": N},
    args=[Arg("x", IN, "float32", (M * K * N,)),
          Arg("b", IN, "float32", (M * N,)),
          Arg("o", OUT, "float32", (M * K * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
