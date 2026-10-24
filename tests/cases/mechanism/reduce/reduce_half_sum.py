"""A clamped softmax row: every element of the answer is 1/N, or the count is wrong."""

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
