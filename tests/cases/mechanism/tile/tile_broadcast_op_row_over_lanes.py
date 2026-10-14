"""A row broadcast written as an OP, over the axis the lanes ended up on.

    row   memref<32xi32>        one value per K, banked one per lane
    tile  memref<32x16xi32>     torchsim.lane_axis = 1 -- the matmul holds B, a
                                column per lane, so N is on the lanes
    op    linalg.broadcast ins(row) outs(tile) dimensions = [1]

WHAT IS BEING PINNED. fit_layout_to_lanes has a ranked menu for a stranded row --
widen the load with a zero DRAM stride, else rebuild it, else a replicating
relayout, else a round trip -- and `_stage_one_map_row` offers it to a
`linalg.generic`'s operand and to NOTHING ELSE. A `linalg.broadcast`'s rank-1
SOURCE is the same row saying the same thing in the other spelling, and the two
paths that saw the op both declined at this pass's position:
`_widen_broadcast_chain` on `if not len(list(bcast.results))` (a bufferized
broadcast has no result) and `_replicate_pinned_broadcasts` on a matmul MUST
that is by then resolved. So the op form got no staging at all and
vectorize refused it five passes later.

`_fold_broadcast_into_readers` restates the op as the map -- rule 8's
"broadcast를 소비자의 map으로" -- which mints no value, deletes one op and one
allocation, and hands the row to the menu that was already written for it.

WHY THERE IS A MATMUL IN A KERNEL ABOUT A BROADCAST, and it is not decoration.
Without one the tile banks on the axis the row RUNS ALONG and the splat is an
ordinary in-lane broadcast -- guard 0 of the menu, free, nothing to stage. It is
the matmul's "holds B, a column per lane" that puts the lanes on the axis the
broadcast ADDS, which is the only arrangement where a lane holds one element of
a row it needs all of. Drop the `tl.dot` and this kernel passes either way and
pins nothing (rule 2).

MEASURED, run.py on this file, before at 5a86d18 and after:

    before   FAIL  adapt: NotVectorisable: broadcast [32] -> [32, 16] over dims
                   [1] runs its source along axis 0 while the tile is banked on
                   axis 1: every lane would need the whole source, which lives
                   one element per lane in other lanes' banks
    after    PASS  max_abs_err 3.8147e-06, 0 DRAM round trips

AND ZERO IS THE POINT OF THE SECOND NUMBER. Once the row is in front of the menu
the menu picks item 2, `_rebuild_computed_row`: the row's producer reads nothing
-- it is `linalg.index` arithmetic -- so every lane builds its own copy and no
element crosses a bank and no tile crosses DRAM. A repair that merely made this
compile would show a round trip here; this one costs a loop over data that never
existed. dl/quant_int4_matmul, dl/mxfp4_shift_scale_matmul and
dl/gptq_actorder_gather take the same fold and DO pay two trips each, because
their rows are computed from a LOADED value rather than from nothing and the
memref spelling of `_widen_chain` -- widen the transfer that staged the row and
retype the chain above it -- does not exist.

WHY i32 AND NOT i1. The replicating round trip goes through the scratch
`_add_dram_arg` appends, which is `memref<*xf32>`: 32 bits, whatever the tile is
made of (see SCRATCH_BITS in p08_fit_layout_to_lanes). A row of another width
crosses at the wrong addresses, so an i1 form of this shape is blocked on that
instead, and mask/mask_col_flag_broadcast_over_rows is where it is recorded with
its measured wrong answer.

WHAT WOULD BREAK THIS. Restoring `_stage_one_map_row`'s deference to
`_widen_broadcast_chain` for a row a `linalg.broadcast` reads; making the fold
unconditional so it also fires where the splat was already free (it would spend
a round trip on a row that needed none); or dropping `_live_uses_of_write`, at
which point a reused allocation hands one broadcast's reader another broadcast's
row -- dl/gptq_actorder_gather writes one `memref<32x16xi32>` from two
broadcasts and is the kernel that shows it.

torch equivalent: x @ w[arange(K) // 2].

The OP spelling of a stranded row, staged by folding it into its readers' maps so _stage_one_map_row's ranked menu can see it. Red at 5a86d18 with vectorize's `broadcast [32] -> [32, 16] over dims [1]` refusal; green after at 3.8147e-06 and 0 round trips. The tl.dot is load-bearing -- it is what puts the lanes on the axis the broadcast adds -- and so is the `// 2`, which is what makes the index a rank-1 row convert_accesses_to_transfer materialises as an op.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 16, 32, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def broadcast_op_row_over_lanes_kernel(x_ptr, w_ptr, o_ptr,
                                           M: tl.constexpr, K: tl.constexpr,
                                           N: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :])       # [M, K]
        # THE ROW IS AN ADDRESS, AND THAT IS WHAT MAKES IT AN OP. `rk // 2`
        # varies along K and not along N, so it reaches the load as a rank-1
        # index that convert_accesses_to_transfer materialises with `linalg.broadcast`
        # -- the same value written as plain data is folded into a consumer's
        # indexing map instead, and the map spelling was already staged. Two
        # rows of w share one row of the source, which is why the index is not
        # simply `rk`.
        w = tl.load(w_ptr + (rk[:, None] // 2) * N + rn[None, :])  # [K, N]
        tl.store(o_ptr + rm[:, None] * N + rn[None, :],
                 tl.dot(x, w.to(tl.float32)))

    return broadcast_op_row_over_lanes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, K, generator=g, dtype=torch.float32),
            "w": torch.randint(-4, 4, (K, N), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    w = inputs["w"]
    rows = torch.arange(K) // 2
    return {"o": (x @ w[rows].to(torch.float32)).reshape(-1)}


SPEC = KernelSpec(
    name="broadcast_op_row_over_lanes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*i32", "o_ptr": "*fp32",
               "M": "constexpr", "K": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "K": K, "N": N},
    args=[Arg("x", IN, "float32", (M, K)),
          Arg("w", IN, "int32", (K, N)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect="blocked",
    expect_why="matmul K=32 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
