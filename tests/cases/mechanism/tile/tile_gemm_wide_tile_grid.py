"""A gather whose gathered dim IS the lane axis, at two elements per lane.

    A   [256, 512] tile, gathered on dim 0 -- NOT the lane axis      correct
    B   [512, 256] tile, gathered on dim 1 -- the lane axis, depth 2 WRONG

Both operands are wrapped (`% M`, `% N`) and the store wraps both axes at once,
so all three go through an indirect transfer. The one whose gathered dim is the
lane axis is the one that breaks,
and the lane axis here carries two elements per lane because BN = 256 on 128
lanes.

WHAT IS MEASURED, and it was two defects stacked -- both now fixed, and both on
the same blind spot: a lane axis carrying more than one element per lane.
Counting DISTINCT values in runtime/indirect_access, which is
layout-independent:

                    before      after
    A's index        256         256    gathered on dim 0, never the lane axis
    B's index        128         256    gathered on dim 1, WHICH IS the lane axis
    the store      32768       65536    2-D, and its column row is rank-1

    answer         110.934    5.34e-05

THE GATHER'S INDEX came from `linalg.index` on the lane dim inside a vectorised
generic, where the depth is a LOOP, so the slot is that loop's counter.
tile_gather_on_lane_axis_deeper.py shows it by storing the staged tile.

THE SCATTER'S INDEX is assembled from a prepared rank-1 row, where the depth is
carried in the VECTOR instead -- `affine.for 0 to 2 step 2` with a vector<2> --
so the counter is always 0 and the slot is `vector.step`.
tile_scatter_wrapped_rows_past_lanes.py isolates it, and its M = 128 control
passes throughout because a lane owns one slot there.

WHAT IS NOT THE CAUSE, measured before this was written. Not the tile depth:
tile_gemm_lane_axis_deeper.py puts BN = 256 on 128 lanes on the axis a matmul
demanded and comes back at 9.54e-06. Not the tile width or the grid: the same
shape, same tile and same 2x2 grid with the wraps REMOVED passes at 5.34e-05.
Not the array lowering: the instructions come out as a 2 x 4 x 2 loop nest, the
right count for a 128-cubed array. Not one index buffer serving two transfers,
which was a real defect with this shape and is fixed in
ptr_two_gathers_one_index_buffer.py. And not the broadcast-pointer gather, which
was the OTHER lane-depth defect and is fixed upstream -- see
tile_bias_row_deeper_than_one_per_lane.py.

WHY IT STILL MATTERS NOW THAT NOTHING REACHES IT. PyTorchSim's Triton route no
longer makes gathers it does not need -- the wrap is removed where the block
divides and replaced with a load bound where it does not -- so no test in that
suite lands here any more. A gather with a genuinely data-dependent index still
does, and that is what a RoPE lookup is.

expect stays "pass" and the runner is right to refuse "blocked" for a wrong
number: nothing refused, every stage produced a program. So it reads as a FAIL,
which is what a reproducer for a live bug looks like. Do not change the shape to
make it pass.

Tile wider than the lane count, both operands wrapped, and a 2x2 grid over the output. PyTorchSim's tile mapping picks exactly this for 512x512x512 and the answer is 158.57 off, which is why the Triton route caps its offered tiles at the lane count.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 512, 512, 512
BM, BN, BK = 256, 256, 512       # BM/BN = 2 x the 128 lanes; K in one trip


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_wide_tile_grid_kernel(a_ptr, b_ptr, c_ptr,
                                        M: tl.constexpr, N: tl.constexpr,
                                        K: tl.constexpr, BM: tl.constexpr,
                                        BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = (pid_m * BM + tl.arange(0, BM)) % M
        offs_n = (pid_n * BN + tl.arange(0, BN)) % N
        offs_k = tl.arange(0, BK)

        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b)

        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return tile_gemm_wide_tile_grid_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g, dtype=torch.float32),
            "b": torch.randn(K, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"], inputs["b"])}


SPEC = KernelSpec(
    name="tile_gemm_wide_tile_grid_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "float32", (M, K)), Arg("b", IN, "float32", (K, N)),
          Arg("c", OUT, "float32", (M, N))],
    grid=(M // BM, N // BN),
    reference=reference, make_inputs=make_inputs,
    # Same reason as tests/cases/smoke/gemm.py: the accumulation order is not torch's.
    rtol=1e-3, atol=1e-3,
)
