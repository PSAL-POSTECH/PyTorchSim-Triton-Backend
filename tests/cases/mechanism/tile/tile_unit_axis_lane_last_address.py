"""A tile banked on its LAST axis with a UNIT axis in the middle, and the row
that is read alongside it.

    row = tl.load(a_ptr + rn[None, :])                     # (1, N)
    out = tl.load(x_ptr + rm[:, None] * N + rn) + row      # (M, N) + (1, N)

WHAT THIS PINS IS THE ADDRESS, not the widening. -linalg-fold-unit-extent-dims
rewrites the row to rank 1 and leaves the tile expanded, so vectorize is
handed

    outs   memref<8x1x16xf32, 1>   torchsim.lane_axis = 2
    ins    memref<1x16xf32, 1>     torchsim.lane_axis = 1   (one element per lane)

-- the [8, 16]-banked-on-axis-1 case with a name inserted in the middle. The
nest used to be chosen by RANK: `lane_nest` required `len(shape) == 2`, so rank
3 declined it, and the dim-0 fallback was taken because its guard -- does one
vector of `shape[-1]` cover what a lane holds -- is true here (16 elements per
lane, 16 wide). That nest walks the tile's LAST axis in-lane, and the last axis
is the one the DMA spread over the lanes, so the row came out as

    affine.vector_load %row[0, %i] : memref<1x16xf32, 1>    i = 0..15

on a buffer that holds ONE element per lane. Every lane read fifteen elements
past its own bank. MEASURED on the same shape at M = N = 16
(ops/ops_widen_row_narrow_reader, once lower_to_dma stopped refusing
the collapse in front of the MVOUT): `max_abs_err = 3.69704, max_rel_err = 1`
against torch, compiling and verifying clean the whole way down -- the failure
this backend cannot see in the IR (rule 12).

M != N ON PURPOSE, AND THE TWO ANSWERS ARE THE POINT. The dim-0 fallback's
guard also asks whether one vector of `shape[-1]` fits inside what spad
RESERVED per lane, and at M = 8, N = 16 it does not -- a lane holds 8 -- so the
old tree stops one branch further down instead:

    NotVectorisable: lane axis 2 of a rank-3 tile is neither dim 0 nor the last
    dim of a rank-2 one, and there is no loop nest for it

At M == N the same shape family passes that guard and comes back wrong. One
shape, two verdicts, decided by whether two extents happen to be equal: that is
what choosing a nest off the shape instead of off the layout costs, and it is
why this file is here rather than a note on the square one.

WHAT MAKES IT FAIL AGAIN: ask the nest question of the shape instead of the
layout -- any test of `len(shape)`, of `laxis == len(shape) - 1`, or of whether
one vector happens to cover `shape[-1]`. The nest is `_bank_order` over
`tile_strides(shape, laxis)` and nothing else; a unit axis is not an axis for
that purpose, which is what makes rank 3 and rank 2 one case.

torch equivalent: o = x + a[None, :].

A [M, 1, N] tile banked on axis 2 read alongside a [1, N] row banked on axis 1. The nest is chosen off the bank's own axis order (tile_strides + _bank_order); choosing it off the rank walked the lane axis in-lane and returned max_abs_err 3.69704 with no diagnostic anywhere.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 8, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def unit_axis_lane_last_kernel(x_ptr, a_ptr, o_ptr, M: tl.constexpr,
                                   N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        # A [1, N] row: one row described as repeated, which is what the unit
        # extent the fold takes off comes from.
        row = tl.load(a_ptr + rn[None, :])                       # (1, N)
        tile = tl.load(x_ptr + rm[:, None] * N + rn[None, :])     # (M, N)
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], tile + row)

    return unit_axis_lane_last_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32),
            "a": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(M, N)
    a = inputs["a"]
    return {"o": (x + a[None, :]).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="unit_axis_lane_last_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "a_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M * N,)),
          Arg("a", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
