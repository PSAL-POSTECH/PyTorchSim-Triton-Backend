"""A clamped softmax row: every element of the answer is 1/N, or the count is wrong.

Every input is 0.0, so exp(x - max) is 1.0 everywhere and each row of N = 128
must sum to 128 -- softmax then puts 1/128 = 0.0078125 in every element. The
reciprocal of any output element is the number of terms that were actually
summed, and any element that was never written is 0.0, so `rtol = atol = 0`
turns both kinds of loss into an exact count with no tolerance to argue about.

WHY THIS KERNEL. It is what Inductor emits for softmax(x, dim=1), reduced to
what pytorchsim-triton-compiler can load. A plain tl.sum over the same axis does not get this far --
it stops in vectorize, the way tests/cases/smoke/reduce.py describes. This shape gets
through and can therefore be wrong instead of refused, which is the point of it:
XBLOCK(128) != xnumel(64), so the store is CLAMPED, and the two axes of the
clamp box carry different extents.

TWO THINGS IT HAS CAUGHT, both silent, both only visible at stage 8 (rule 12):

  1/64 WHERE 1/128 WAS CORRECT. The reduction added exactly half its elements:
  in 04-passes/09-torchsim-legalize.mlir the tree ran over `vector<64xf32>`
  and halved from there, `vector<128xf32>` appearing only as constants and
  broadcasts. 64 is the
  ROW count, not the column count, so the widening had followed the lane axis
  rather than the axis inside a lane. Also measured on PyTorchSim's Inductor
  route as layer_norm over 128 columns coming out scaled by sqrt(2) (var =
  64/128); with +1 on one half of a row and -1 on the other the computed mean
  stayed 0, so the missing half was not a contiguous run.

  HALF OF EVERY ROW LEFT AT 0.0, which is this file's live measurement --
  `max_abs_err = 0.0078125`, 4096 of 8192 elements, first bad at flat index 64.
  absorb_layout_ops turned the tile under the MVOUT and permuted the descriptor's
  `dram_stride` [128, 1] -> [1, 128] without renaming `masked_axes`, so the
  clamp that meant "64 real ROWS" was read as "64 columns" of a tile whose axis
  0 had become the 128-wide reduced axis. Pinned in the small at
  tests/cases/mechanism/mask/mask_clamped_store_turned_tile.py; this file is the one
  that found it, and it found it because a softmax's two axes are different
  lengths and the clamp is on the shorter one.

IT SPENT A WHILE NOT PINNING ANY OF THAT. Between the axis-graph restructure and
the scratchpad fix it stopped at stage `binary` instead -- "kernel claims 66752
bytes/lane but Spike maps only 65536" -- because its [128, 1, 128] temporary was
reserved whole in every lane (see pytorchsim_triton_compiler/lowering/passes/lib_bank_layout.py reserve_per_lane, and
tests/cases/mechanism/tile/tile_spad_unit_axis.py, which pins that separately). Red
for a reason that was not this one, which is the same as holding nothing down.

A clamped softmax row. rtol=atol=0, so 1/count falls straight out of any element: half a sum reads 1/64, an unwritten column reads 0.0, and both used to be reported as success.
"""

#: WRONG NUMBERS FOR A WHILE ON 2026-09-23 (`inf`), AND WHAT FLIPPED IT BACK.
#: The commit that made a layout answer reach an op through the slot it WRITES
#: -- destination passing, which upstream has no arm for -- took this kernel
#: from a pass to `inf` while taking two others from wrong numbers back to
#: refusals (3 -> 1 across the suite). Ruled out then: the masked MVIN carries
#: `masked_fill = 0`, the select writes every element, the in-bank turn after it
#: is a real bitcast, and the round trip after THAT rebanks a written tile.
#: Making the new arm non-mandatory does not fix it and costs a second wrong
#: answer elsewhere (440/2 against 440/1). Giving the cross-lane unit the copies
#: cleared it: the suite went to 442/0 and this kernel refuses rather than lying.
#:
#: FOUND IT LATER THE SAME DAY, AND IT WAS THE SPLIT. `SplitDisagreeingBuffers`
#: moved the version a transpose defines onto a new global and repointed "every
#: user of it" -- including the SECOND transpose, which takes that buffer at the
#: slot it WRITES. Moving that slot relocated the version that transpose defines
#: while its reader, the second reduce, kept reading buf1, which then nobody
#: wrote. The sum was 0 and 1/0 is the `inf`.
#:
#: A slot the user writes is not a reader of this version: it carries the NEXT
#: one. Where the op overwrites, the slot stays put; where it starts from what
#: was there, the next version lives in the new buffer too and its readers move
#: with it. Refusing instead of carrying that chain costs
#: `kv_cache_quant_mixed_axis` (measured).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 64           # rows
N = 128          # columns -- the reduced axis
BM = 128         # rows per program == TORCHSIM_COMPILE_VECTORLANE_SIZE


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_half_sum(in_ptr0, out_ptr2, xnumel: tl.constexpr,
                        r0_numel: tl.constexpr, XBLOCK: tl.constexpr):
        R0_BLOCK: tl.constexpr = 128
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_1 = tl.arange(0, R0_BLOCK)[None, :]
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (r0_1 + 128 * x0), xmask, other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = tl.where(xmask, tmp1, float("-inf"))
        # tl.max stands in for triton_helpers.max2, which lives in torch.
        tmp4 = tl.max(tmp3, 1)[:, None].to(tl.float32)
        tmp5 = tmp0 - tmp4
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = tl.where(xmask, tmp7, 0)
        tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)   # <- returns N/2
        tmp11 = tmp6 / tmp10
        tl.store(out_ptr2 + (r0_1 + 128 * x0), tmp11, xmask)

    return reduce_half_sum


def make_inputs(torch, seed=0):
    # All zeros: exp(x - max) == 1, so every element of the answer is 1/N and
    # the reciprocal of any output element is the number of terms summed.
    return {"x": torch.zeros(M * N, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.full((M * N,), 1.0 / N, dtype=torch.float32)}


SPEC = KernelSpec(
    name="reduce_half_sum",
    kernel=kernel,
    signature={"in_ptr0": "*fp32", "out_ptr2": "*fp32", "xnumel": "constexpr",
               "r0_numel": "constexpr", "XBLOCK": "constexpr"},
    constexprs={"xnumel": M, "r0_numel": N, "XBLOCK": BM},
    args=[
        Arg("x", IN, "float32", (M * N,)),
        Arg("o", OUT, "float32", (M * N,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=0, atol=0,
)
