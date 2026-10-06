"""A runtime clamp on a lane axis LONGER THAN THE MACHINE.

    tl.load(..., mask=rn < n)     over a [16, 512] tile, 256 lanes, n = 300

WHAT IT PINS. The masked axis is the one the tile is banked on, and 512 on a
256-lane machine is TWO elements per lane -- so the normal form states that axis
as dims 0 and 1 of `memref<2x256x16xf32>`, depth and vlane. A clamp is a fact
about the whole axis and there is no half of it to put the fact on: seated on
the vlane alone, `hi = 300` is compared against an extent of 256, never fires,
and elements 256..511 arrive UNMASKED.

    lower_to_dma glues the two back           max_abs_err 0
    the seat taken literally (glue off)       max_abs_err 16.8222, 3363/8192 wrong

Measured by disabling `_Tx._glue_the_lane_axis` on this tree; before the glue
existed, fit_layout_to_lanes refused the kernel outright rather than seat the
clamp -- which is why the wrong answer is reachable only by hand.

WHY THE DESCRIPTOR IS WHERE IT IS SAID. The DMA spreads one descriptor dim over
the banks itself and derives the laps from that dim's extent
(`n_outerloop = ceil(extent / (vlane_stride * lanes))`), so depth is a number
the hardware computes and not a dim the descriptor has to hold. Glued, the dim
is 512 long and `hi = 300` means 300.

WHY THIS SHAPE AND NOT `tile_reduce_over_unit_axis`, which measured the same
defect: that one carries an extent-1 axis, so a `linalg.generic` there names a
place in an axis with a literal zero and the normal form has its own wall in
front of it. This kernel's maps are identities and nothing else stands between
it and the descriptor.

`n` is a RUNTIME scalar, and that is what makes the transfer carry the fact this
pins -- the bound arrives as an operand the descriptor is written from. The IR it
reaches, so the next reader can check it is still this shape:

    "torchsim.transfer"(...) {dma_kind = "MVIN", dram_stride = array<i64: 256, 1, 512>,
                              masked_axes = array<i64: 1>, masked_fill = 0 : i64}
                             : (..., memref<2x256x16xf32, 1>, ...) -> ()

`dram_stride`'s 256 and 1 are the lap relation the glue checks: one step of depth
is 256 steps of the vlane, which is what makes the two dims one axis.

ARITHMETIC. `x * x` under a one-axis mask, exact in f32.

a runtime clamp on the lane axis of a tile two laps deep: the descriptor is where depth and vlane are one axis again.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 16
N = 512
NBOUND = 300


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_clamp_on_a_deep_lane_axis_kernel(x_ptr, o_ptr, n,
                                              M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)[:, None]
        rn = tl.arange(0, N)[None, :]
        v = tl.load(x_ptr + rm * N + rn, mask=rn < n, other=0.0)
        tl.store(o_ptr + rm * N + rn, v * v)

    return tile_clamp_on_a_deep_lane_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(M, N)
    keep = (torch.arange(N).reshape(1, N) < NBOUND)
    v = torch.where(keep, x, torch.zeros_like(x))
    return {"o": v * v}


SPEC = KernelSpec(
    name="tile_clamp_on_a_deep_lane_axis_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M * N,)),
          Arg("o", OUT, "float32", (M, N))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    extra={"scalar_args": [("n", "int32_t")],
           "scalar_values": {"n": NBOUND}},
)
