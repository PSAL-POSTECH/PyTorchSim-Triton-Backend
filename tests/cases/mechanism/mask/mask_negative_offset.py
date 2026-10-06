"""A masked load whose masked-off lanes address out of range, and segfault.

    tl.load(x_ptr + (idx - 128), idx >= 128, other=0.0)

For idx < 128 the offset is negative and the mask is false, so nothing should be
read. Spike traps instead:

    User load segfault @ 0x00000004d0800fe0

The address is just past the end of the banked window (0xd0000000 + 8 MiB), with
a spurious bit at 34 -- a negative element offset added to the scratchpad base
and then read as unsigned. The mask selected the value but did not gate the
access.

WHERE IT COMES FROM. This is the shape Inductor emits whenever an output draws
from more than one source region -- cat and constant_pad_nd both do it, and both
fail here:

    # torch.cat, second operand
    tmp9  = tl.load(in_ptr1 + (x0 + 16*((-8) + x1)), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)

    # constant_pad_nd
    tmp12 = tl.load(in_ptr0 + ((-12) + x0 + 10*x1), tmp11 & xmask, other=0.0)

Both branches are emitted unconditionally and the mask picks between them, so
the out-of-range branch is always addressed. On a GPU the masked lanes simply do
not issue; here they do.

Measured on PyTorchSim's Inductor route: tests/ops/view/test_cat.py faults at
0x4d0801220 and tests/ops/misc/test_masked_nondividing.py at 0x4d0800fe0.

THE ARITHMETIC, NOW THAT THE HARNESS PRINTS IT. The sweep reported this as
`spike: command failed with exit code 255` for one release -- spike writes its
whole trap report to STDOUT and pipeline._run only ever carried stderr. It now
says `User load segfault @ 0x00000004d0800e10`, and the register dump beside it
closes the case:

    a7 (the DMA's DRAM base)              0xd0801010
    the offset the kernel computed        idx - 128 = -128, as i32
    the DMA's byte scale                  offset_stride 1 x 4 bytes
    0xd0801010 + zext_i64(-128) * 4    =  0x4d0800e10     the faulting address
    0xd0801010 -           128   * 4   =  0x0d0800e10     what it meant

So the spurious bit at 34 is not a lost width in OUR address arithmetic: it is
32 + log2(4), the signature of a 32-bit offset field read UNSIGNED by the DMA.
ms_deform_attn faults at bit 37 for the same reason with offset_stride 8.

WHAT FIXED IT, AND IT WAS NOT A CLAMP. The DMA is told the index element width
in the descriptor (`indirect_esize`, byte +130) and reads that many bytes
UNSIGNED -- load_uint8 / _uint16 / _uint32 / _uint64. At eight bytes the zero
extension is the identity, so the multiply and the add wrap in two's complement
and a negative index addresses BELOW the base, which is what the program said.
convert_accesses_to_transfer therefore widens every indirect index tile to i64
unconditionally (_Ctx.index_buffer), and the descriptor's esize follows the
buffer, as it always did.

Clamping each offset into [0, numel) would also stop the trap, and was the
earlier plan recorded here. It is a worse answer: it changes which element an
out-of-range gather reads, so it hides the bug in the cases where the mask does
NOT discard the value, while i64 leaves the program's meaning alone. Nothing has
to decide which kernels might go negative either -- at 64 bits there is no width
left to get wrong.

WHAT WOULD BREAK IT AGAIN: narrowing that tile. rwkv_token_shift moved with this
one; causal_conv1d, conformer_conv and ms_deform_attn stopped trapping and now
fail on a second, unrelated bug -- see tile/tile_lane_axis_row_overrun.py.

Masked-off lanes are still addressed, so a negative offset faults. cat and constant_pad_nd both emit this shape.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256          # elements out
HALF = 128       # the first half reads nothing; the second reads x[0:128]
BN = 128         # == TORCHSIM_COMPILE_VECTORLANE_SIZE


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_negative_offset(x_ptr, o_ptr, HALF: tl.constexpr,
                               BN: tl.constexpr):
        idx = tl.program_id(0) * BN + tl.arange(0, BN)
        take = idx >= HALF
        # For idx < HALF this offset is negative. take is false there, so the
        # load must not happen.
        v = tl.load(x_ptr + (idx - HALF), take, other=0.0)
        tl.store(o_ptr + idx, v)

    return mask_negative_offset


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(HALF, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.cat([torch.zeros(HALF), inputs["x"]])}


SPEC = KernelSpec(
    name="mask_negative_offset",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "HALF": "constexpr",
               "BN": "constexpr"},
    constexprs={"HALF": HALF, "BN": BN},
    args=[
        Arg("x", IN, "float32", (HALF,)),
        Arg("o", OUT, "float32", (N,)),
    ],
    grid=(N // BN,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=0, atol=0,
)
