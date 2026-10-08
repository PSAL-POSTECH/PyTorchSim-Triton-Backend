"""ONE row every lane needs, written BOTH WAYS in one function -- and only the
map spelling is offered a repair.

    flag  memref<16xi1>          one value per COLUMN, banked one per lane
    tile  memref<8x16xi1>        torchsim.lane_axis = 0, 8 rows one per lane
    op    linalg.broadcast ins(flag) outs(tile) dimensions = [0]
    map   the same flag read `(d0,d1) -> (d1)` by the generic below it

THE 15-LINE FORM OF dl-side mask/mask_2d_and_arms, which carries the identical
pair (its `%17`/`buf7_spad_16lane`) inside 130 lines of MaskAnalysis coverage.
Reduced here because the shape is not about masks: it is about a rank-1 value
whose axis is not the lane axis, and the mask is only the cheapest way to make
the linalg lowering MATERIALISE the broadcast as an op instead of folding it into a
consumer's map. The load needs the flag narrow (`(d0,d1)->(d1)`); the masked
store needs it as a TILE for the read-modify-write `arith.select`, and that
tile is the `linalg.broadcast`.

WHAT IS REFUSED, and it is vectorize:

    NotVectorisable: broadcast [16] -> [8, 16] over dims [0] runs its source
    along axis 1 while the tile is banked on axis 0: every lane would need the
    whole source, which lives one element per lane in other lanes' banks

THE REFUSAL IS CORRECT AND THE FIX IS NOT THERE. Read off the banking rather
than off the message: `buf7 : memref<16xi1, 1>` is stamped `torchsim.lane_axis = 0`,
so its 16 elements are in 16 different banks, one each; the destination is
stamped 0 as well, so lane r holds row r -- all 16 columns of it. Lane r
therefore needs all 16 source elements and 15 of them are in banks that are not
its own. This machine has no instruction that crosses banks (rule 8), so no
address vectorize could emit is right, and the repair the message names --
put the whole row in EVERY bank -- mints a value and belongs to fit_layout_to_lanes
(rule 17 step 3/4, rule 7).

THE SPELLING PROBLEM IS FIXED AND IT WAS NOT THE ONLY ONE. This file was written
against `_stage_one_map_row`'s ranked menu being offered to a `linalg.generic`'s
stranded operand and to NOTHING ELSE, so the broadcast OP's rank-1 source never
reached it -- `_widen_broadcast_chain` declines on `if not
len(list(bcast.results))` (a bufferized broadcast has no result) and
`_replicate_pinned_broadcasts` asks a matmul MUST for an axis that is by then
resolved. `_fold_broadcast_into_readers` closes that: the op is restated as the
map spelling, which mints nothing, and BOTH of this file's readings of the row
then reach the menu. tile/tile_broadcast_op_row_over_lanes.py is the green pin
for it.

WHAT STOPS IT NOW IS THE WIDTH OF THE ROW, and it is a fact about the round trip
rather than about the broadcast. The row is `memref<16xi1>` and it is neither
loadable wide (its buffer's writer is a compare, not a transfer) nor rebuildable
(that compare reads something), so the menu's only remaining answer is the
replicating relayout -- and the scratch `_add_dram_arg` appends is
`memref<*xf32>`, so the descriptor's element_size is 32 bits whatever the tile
is made of (p08_fit_layout_to_lanes.SCRATCH_BITS). Measured over tests/cases/
at 5a86d18: every one of the 436 emitted relayout transfers is f32 or i32 and
NOT ONE is any other width, so nothing had ever asked.

    with the trip allowed   compiles clean, verify 2.56723 -- of the eight
                            columns the flag selects ONE arrives. The ROW
                            structure is right and the COLUMN structure is gone,
                            which is what reading 16 one-bit elements back at
                            f32 addresses looks like in memory.
                            mask/mask_2d_and_arms, the same row inside 130
                            lines, answers 9.21007.
    with it declined        back to a refusal, and a refusal is the right
                            answer while the trip cannot be built (rule 12).

BOTH OF THE TWO BELOW HAVE SINCE HAPPENED, and this kernel is GREEN. The section
is kept because it is the clearest statement of the two answers and of why they
are different, but read it as history rather than as a plan:

    the second one landed first -- `_widen_chain` is the memref spelling now,
    and `_clone_row_chain` hands it a copy when a reader still wants the row
    narrow. That is what takes this kernel and mask_2d_and_arms, both for free.

    the first one landed differently. The scratch is STILL `memref<*xf32>` and
    the signature is unchanged; what changed is that a sub-word row is widened
    to i32 on the way out and truncated back on the way in, so the trip itself
    is 32 bits and the descriptor goes on telling the truth
    (p08_fit_layout_to_lanes._travels_as_scratch). The pin for that is
    mask/mask_conv_bounds_row_over_lanes, whose row has none of the free answers
    this one has.

WHAT WOULD MAKE IT GREEN, and it is two separate things:

    a DRAM scratch that carries the tile's own element width -- `_add_dram_arg`
    hardcodes `memref<*xf32>` and stage 5 allocates that many f32s, so this is a
    signature-and-wrapper change, not a fit_layout_to_lanes-local one; or

    the memref spelling of `_widen_chain`: widen the `torchsim.transfer` that
    staged `cf` with a zero `dram_stride` on the added axis and retype the
    compare above it. That is FREE -- no round trip at all, and it is menu item
    1 for a row whose leaf is a load -- and it is what `_widen_chain`'s own
    docstring says is missing ("what is missing is its memref spelling"). It
    would take this kernel, mask_2d_and_arms, and the six round trips
    dl/quant_int4_matmul, dl/mxfp4_shift_scale_matmul and
    dl/gptq_actorder_gather now pay.

WHAT WOULD BREAK IT ONCE GREEN. Dropping vectorize's `saxis != laxis`
refusal in `_bank_broadcast` to make this file green without the staging -- that
refusal is the only thing between this shape and a silent wrong answer, because
every lane would splat its own single flag across all 16 of its columns and a
per-column mask would become a per-row one. Measured on
mask_gs_dim_mismatch when the two took the same path: all 64 positions that
should have been written came back 0, with no diagnostic anywhere.

MEASURED. At 083800f and at 5a86d18 the refusal named the broadcast OP:

    FAIL  adapt: NotVectorisable, `broadcast [16] -> [8, 16] over dims [0]`
          on `linalg.broadcast ins(%8 : memref<16xi1, 1>)
          outs(%7 : memref<8x16xi1, 1>) dimensions = [0]`

With the fold in, the op is gone and the refusal names the row itself, which is
the more accurate sentence and the same kernel:

    FAIL  adapt: NotVectorisable: operands of one elementwise op are banked on
          different iteration dims [0, 1] ... ins(%7, %9, %7, ... :
          memref<16xi1, 1>, memref<8xi32, 1>, memref<16xi1, 1>, ...)

This file is the reproducer, not the fix. It stays red on purpose.

torch equivalent: a[i, j] where both flags are set and 0 elsewhere, stored only
into the columns whose flag is set.

The OP spelling of tile_one_bank_row_times_banked_tile's row, in the same function as the MAP spelling of it. The spelling half is fixed -- _fold_broadcast_into_readers restates the op as the map and tile/tile_broadcast_op_row_over_lanes pins that -- and this stays RED on the half that is not fit_layout_to_lanes's: the row is i1 and the DRAM scratch a replicating relayout goes through is memref<*xf32>, so the trip addresses a 1-bit tile at 32-bit strides. Allowed through, it compiles and answers 2.56723 (mask_2d_and_arms 9.21007); declined, it refuses. The 15-line form of mask/mask_2d_and_arms.
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
    def col_flag_broadcast_over_rows(a_ptr, rf_ptr, cf_ptr, o_ptr,
                                     M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        # TWO i1 MASKS ON DIFFERENT AXES. MaskAnalysis has one gather dim and
        # cannot fold them into it, so both survive as tensors and their `&` is
        # a 2-D `arith.andi` -- which is what makes the per-column one arrive as
        # a `linalg.broadcast` op rather than as a constant in a consumer's map.
        rowf = tl.load(rf_ptr + rm) != 0        # one flag per ROW
        colf = tl.load(cf_ptr + rn) != 0        # one flag per COLUMN
        v = tl.load(a_ptr + rm[:, None] * N + rn[None, :],
                    mask=rowf[:, None] & colf[None, :], other=0.0)
        # AND THE SAME FLAG ON THE STORE. A masked store becomes a
        # read-modify-write `arith.select`, which needs the flag as a TILE --
        # and that is the `linalg.broadcast` op. The load above needs it as a
        # rank-1 read through `(d0,d1) -> (d1)`. One row, both spellings, one
        # function: this is where the two answers can be compared directly.
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], v,
                 mask=colf[None, :])

    return col_flag_broadcast_over_rows


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M * N, generator=g, dtype=torch.float32),
            "rf": (torch.arange(M, dtype=torch.int32) % 3).contiguous(),
            "cf": (torch.arange(N, dtype=torch.int32) % 2).contiguous()}


def reference(inputs):
    import torch
    a = inputs["a"].reshape(M, N)
    rf = (inputs["rf"] != 0).reshape(M, 1)
    cf = (inputs["cf"] != 0).reshape(1, N)
    return {"o": torch.where(rf & cf, a, torch.zeros_like(a)).reshape(-1)
            .contiguous()}


SPEC = KernelSpec(
    name="col_flag_broadcast_over_rows", kernel=kernel,
    signature={"a_ptr": "*fp32", "rf_ptr": "*i32", "cf_ptr": "*i32",
               "o_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("a", IN, "float32", (M * N,)),
          Arg("rf", IN, "int32", (M,)),
          Arg("cf", IN, "int32", (N,)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
