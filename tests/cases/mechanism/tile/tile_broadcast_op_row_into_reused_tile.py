"""Inductor's conv template, depthwise, where the epilogue's row broadcast lands in a REUSED tile.

    op     linalg.broadcast ins(memref<32xi32>) outs(memref<32x16xi32>)
           dimensions = [1]         -- BLOCK_M over BLOCK_N
    tile   torchsim.lane_axis = 1       -- the tl.dot holds B, a column per lane
    AND    the loop body stages an index tile of the SAME memref type, so
           bufferize hands both of them ONE allocation

WHAT IS BEING PINNED, AND HOW IT DIFFERS FROM ITS SIBLING.
tile_broadcast_op_row_over_lanes pins the same refusal --

    broadcast [32] -> [32, 16] over dims [1] runs its source along axis 0 while
    the tile is banked on axis 1: every lane would need the whole source, which
    lives one element per lane in other lanes' banks

-- and it is answered there for FREE by `_fold_broadcast_into_readers`: the op is
restated as its readers' indexing maps, no value minted, 0 round trips. That fold
has a correctness guard, `_live_uses_of_write`, and THIS is the shape that trips
it. Measured with a counter on the guard, on this template at these constexprs:
`_live_uses_of_write` returns 0 uses and the fold returns False. The loop's index
tile and the epilogue's are the same memref type, they share one allocation, and
so a reader of the broadcast's tile may be reading what the loop left there
rather than the row -- which is exactly what the guard refuses to assume away.

So rule 17's stage 3 is genuinely empty here, and that is the point of the file.
The candidates at this position, enumerated (rule 6):

    _widen_broadcast_chain          declines -- a bufferized broadcast has no
                                    result, and _widen_chain follows tts ops that
                                    died five passes above this one
    _fold_broadcast_into_readers    declines -- the write is untieable, above
    _rebuild_computed_row           never asked: it is offered to a
                                    linalg.generic's OPERAND, and a broadcast op
                                    is not an operand of anything
    _replicate_broadcast_on_lane_axis   answers, with one replicating relayout

That last one is stage 4, a contradiction in rule 17's sense -- what the lanes
need is data that is not in their bank, so no rearrangement of what exists can
answer and exactly one value must be minted. MVOUT + MVIN. This kernel therefore
reports one round trip where its sibling reports none, and that is the correct
answer for it rather than a regression against it (rule 8b: name the trip).

WHY THE TEMPLATE AND NOT A HAND-WRITTEN ROW. Two hand-written attempts are
recorded here because both FAILED to pin anything, and the reason is worth more
than the attempts:

    same row in both places      cse merges the two broadcasts into one value.
                                 The loop and the epilogue then share a VALUE,
                                 not an allocation, `_live_uses_of_write` ties
                                 its readers, and the fold answers for free.
    different rows, `// 2` and   the epilogue's `tl.broadcast_to(row[:, None],
    `// 4`                       [BM, BN])` never becomes an op at all --
                                 triton-shared folds that spelling into the
                                 load's map. Only one broadcast op survives to
                                 bufferize, inside the loop, and it is answered.

Both passed with the repair removed, which is rule 2's disqualification. What
actually produces the pair is the template's own arithmetic: the epilogue's row
is `idx_y_h` recombined with the output-channel stride, the loop's is
`idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H`, and they are DIFFERENT rows of
the SAME type because BLOCK_K and BLOCK_N are both 16. Trimmed, not rewritten:
the dims are shrunk (IN_C 2560 -> 16, IN_H 32 -> 8) and everything structural is
the template's.

WHERE IT CAME FROM. RecurrentGemma's temporal block -- a depthwise conv1d,
GROUPS == IN_C, KERNEL_H=4, PADDING_H=3. It compiled until 6a4d285 stopped
widening every indirect transfer's index tile to i64: that widening had been
minting a fresh single-reader buffer in front of the broadcast, which the
map-spelling ladder could answer, and removing it left the kernel's own row --
read six times over -- standing where nothing else has an answer.

MEASURED, run.py on this file:

    without _replicate_broadcast_on_lane_axis   FAIL adapt, the refusal quoted
                                                at the top, out of _bank_broadcast
    with it                                     PASS

WHAT WOULD BREAK THIS. Removing `_replicate_broadcast_on_lane_axis` again on a
count of firings -- a616ea5 did exactly that, measured 423 entries and 0 True,
and read it as "the suite has no such shape" when the suite simply had no such
kernel; this file is that kernel. Also: making `_fold_broadcast_into_readers`
ignore `_live_uses_of_write`, which would answer this for free and silently hand
the loop's reader the row instead of what the loop wrote.

torch equivalent: F.conv1d(x, w, groups=IN_C, padding=3) + bias, over one
BLOCK_M of output positions.

Inductor's depthwise conv template, where the epilogue's row broadcast writes a tile allocation the loop body also uses -- so _fold_broadcast_into_readers declines on _live_uses_of_write (0 uses, measured) and only the replicating round trip answers. Red without _replicate_broadcast_on_lane_axis with vectorize's `broadcast [32] -> [32, 16] over dims [1]` refusal. RecurrentGemma's temporal block is where this arrives.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# The template's own names, trimmed. GROUPS == IN_C is what makes it depthwise,
# and it is what drives GROUP_IN_C to 1 -- so BLOCK_K covers the whole reduction
# in one step and the loop is over KERNEL_H alone.
BATCH, IN_C, IN_H, KERNEL_H, PADDING_H = 1, 1, 8, 4, 3
OUT_C = IN_C
OUT_H = IN_H + 2 * PADDING_H - KERNEL_H + 1          # 11
BLOCK_M, BLOCK_N, BLOCK_K = 32, 16, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def broadcast_op_row_into_reused_tile_kernel(
            X, W, in_ptr2, out_ptr1,
            BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr,
            OUT_C: tl.constexpr, OUT_H: tl.constexpr,
            KERNEL_H: tl.constexpr, PADDING_H: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr):
        GROUPS: tl.constexpr = IN_C
        STRIDE_H: tl.constexpr = 1
        stride_xn = IN_C * IN_H
        stride_xc = IN_H
        stride_xh = 1
        stride_wc_out = KERNEL_H
        stride_wc_in = KERNEL_H
        stride_wh = 1

        # ONE PROGRAM, AND THE SHAPE DOES NOT NEED MORE. The template drives
        # (nhw, out-channel, group) off three program ids; at IN_C = 1 the
        # group axis has one value and BLOCK_M = 32 covers every output
        # position (OUT_H = 11), so the whole convolution is one program and
        # the structure this kernel is about -- two same-typed index tiles,
        # one in the loop and one in the epilogue -- is untouched.
        #
        # MEASURED, and the reason it is not a multi-program grid: at IN_C=16
        # only the first program ran, leaving 165 of 176 outputs as whatever
        # the buffer held. A kernel whose coverage depends on programs that do
        # not run pins nothing (rule 2), and chasing the grid is a different
        # question from the one this file is for.
        group = tl.program_id(0)
        nhw = tl.arange(0, BLOCK_M)
        idx_y_h = nhw % OUT_H
        idx_n = nhw // OUT_H
        idx_y_c = tl.arange(0, BLOCK_N)
        GROUP_IN_C: tl.constexpr = IN_C // GROUPS
        GROUP_OUT_C: tl.constexpr = OUT_C // GROUPS

        x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
        w_base = W + (group * stride_wc_out * GROUP_OUT_C
                      + idx_y_c * stride_wc_out)[None, :]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for i in range(KERNEL_H):
            # THE LOOP'S ROW: idx_x_h, one value per BLOCK_M, staged over
            # BLOCK_K. Same memref type as the epilogue's row below, and a
            # DIFFERENT value -- that pair is what bufferize collapses onto one
            # allocation and what `_live_uses_of_write` then cannot untangle.
            idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
            idx_x_c = tl.arange(0, BLOCK_K)
            x_ptrs = x_base + ((idx_x_h * stride_xh)[:, None]
                               + (idx_x_c * stride_xc)[None, :])
            mask_x = ((idx_n < BATCH)[:, None]
                      & (idx_x_h >= 0)[:, None]
                      & (idx_x_h < IN_H)[:, None]
                      & (idx_x_c < GROUP_IN_C)[None, :])
            matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

            w_ptrs = w_base + ((idx_x_c * stride_wc_in)[:, None] + i * stride_wh)
            mask_w = ((idx_x_c[:, None] < GROUP_IN_C)
                      & (idx_y_c[None, :] < GROUP_OUT_C))
            matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
            acc += tl.dot(matrix_x, matrix_w, allow_tf32=True)

        mask = ((idx_n < BATCH)[:, None]
                & (idx_y_h < OUT_H)[:, None]
                & (idx_y_c[None, :] < GROUP_OUT_C))
        idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
        idx_h = idx_y_h[:, None]
        # THE EPILOGUE'S ROW, and the reason it is an OP rather than a map: the
        # index is recombined arithmetic over BLOCK_M that the store and the bias
        # load both read, so convert_accesses_to_transfer materialises it once as a tile
        # instead of folding it into either one's map.
        tmp0 = tl.load(in_ptr2 + tl.broadcast_to(idx_c, [BLOCK_M, BLOCK_N]),
                       mask, eviction_policy='evict_last')
        tl.store(out_ptr1 + tl.broadcast_to(idx_h + OUT_H * idx_c,
                                            [BLOCK_M, BLOCK_N]),
                 acc + tmp0, mask)

    return broadcast_op_row_into_reused_tile_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"X": torch.randn(BATCH, IN_C, IN_H, generator=g, dtype=torch.float32),
            "W": torch.randn(OUT_C, 1, KERNEL_H, generator=g, dtype=torch.float32),
            "bias": torch.randn(OUT_C, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    import torch.nn.functional as F
    y = F.conv1d(inputs["X"], inputs["W"], padding=PADDING_H, groups=IN_C)
    y = y + inputs["bias"][None, :, None]
    return {"out": y.reshape(-1)}


SPEC = KernelSpec(
    name="broadcast_op_row_into_reused_tile_kernel", kernel=kernel,
    signature={"X": "*fp32", "W": "*fp32", "in_ptr2": "*fp32",
               "out_ptr1": "*fp32",
               "BATCH": "constexpr", "IN_C": "constexpr", "IN_H": "constexpr",
               "OUT_C": "constexpr", "OUT_H": "constexpr",
               "KERNEL_H": "constexpr", "PADDING_H": "constexpr",
               "BLOCK_M": "constexpr", "BLOCK_N": "constexpr",
               "BLOCK_K": "constexpr"},
    constexprs={"BATCH": BATCH, "IN_C": IN_C, "IN_H": IN_H, "OUT_C": OUT_C,
                "OUT_H": OUT_H, "KERNEL_H": KERNEL_H, "PADDING_H": PADDING_H,
                "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
    args=[Arg("X", IN, "float32", (BATCH, IN_C, IN_H)),
          Arg("W", IN, "float32", (OUT_C, 1, KERNEL_H)),
          Arg("bias", IN, "float32", (OUT_C,)),
          Arg("out", OUT, "float32", (BATCH * OUT_C * OUT_H,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
