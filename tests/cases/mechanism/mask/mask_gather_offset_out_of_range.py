"""A per-element mask guarding offsets that are NOT near the buffer's end.

_empty_read_mask_carriers moves a gather's mask into a select AFTER the read: everything
is gathered, then the masked-off elements are replaced. That leaves every
masked-off OFFSET live in the descriptor, and the DMA dereferences each one.

The pass used to say that was safe because emit_wrapper_c callocs every buffer
rounded up and DOUBLED, so a boundary mask (`offs < n` over an arange) puts the
masked-off offsets inside the padding. TWO THINGS ARE WRONG WITH THAT. The
doubling is only on `out`/`inout` -- the factor exists for _park_masked_offsets,
and Llama 4 Scout's 1.25 GiB expert stack is what stopped it being applied to an
`in` -- so a READ, which is what this is, has 512-byte alignment and nothing
more. And the mask here does not guard a boundary at all: it guards a SEPARATE
expression, and the offsets it turns off are nowhere near the end.

WHERE IT CAME FROM. DeepSeek V3's RoPE kernel, which Inductor emits as

    tmp21 = tl.load(in_ptr0 + (512 + 2*(((-32) + x0) % 32) + 576*x1 + ...),
                    tmp18 & xmask, other=0.0)          # tmp18 is x0 >= 32

`%` there is arith.remsi, which TRUNCATES, so the lanes the mask turns off
(x0 < 32) get a large negative remainder rather than a small positive one: at
x0=1 the offset came out 11970 where the live lanes were asking for 12035.
Measured on that kernel before the fix, the dumped index buffer held 11970 at
lane 1 and 12607 at lane 64; after it, both are 0 and the live lanes keep their
real offsets.

This kernel keeps that SHAPE -- a mask on one expression, an offset built from
another -- and makes the consequence observable by putting the masked-off
offsets far past the allocation rather than 65 elements away from the live
ones. What it pins is the rule, not DeepSeek's arithmetic: an offset under a
mask is not read, so it must not be dereferenced either.

WHAT BREAKS IT: computing the gather's offsets before the mask reaches them --
that is, applying the mask to the loaded VALUES only. Revert the offset select
in _rewrite_gather and this reads OFF elements past the end
of a 64-element buffer.

Zero is safe to clamp to because this is a gather: the value comes back and is
thrown away by the select the pass already builds. The argument against
clamping in that file is about SCATTER, where several masked-off elements
colliding on element 0's slot would each write it.

A gather whose masked-off offsets are far out of range rather than just past the end. Pins that the mask reaches the offsets, not only the loaded values.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 64
BLOCK = 64
# Far past x's 512-byte round-up, so a masked-off read is not quietly absorbed
# the way a boundary mask's is.
OFF = 1 << 18


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_gather_offset_out_of_range_kernel(x_ptr, o_ptr, OFF: tl.constexpr,
                                               BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        keep = r >= 32
        # The offset is its own expression, not the mask's: `keep` says nothing
        # about how big this gets, which is the whole point.
        idx = tl.where(keep, r - 32, OFF)
        v = tl.load(x_ptr + idx, mask=keep, other=0.0)
        tl.store(o_ptr + r, v)

    return mask_gather_offset_out_of_range_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    r = torch.arange(BLOCK)
    v = torch.where(r >= 32, x[torch.clamp(r - 32, min=0)],
                    torch.zeros(BLOCK, dtype=torch.float32))
    return {"o": v}


SPEC = KernelSpec(
    name="mask_gather_offset_out_of_range_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "OFF": "constexpr",
               "BLOCK": "constexpr"},
    constexprs={"OFF": OFF, "BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
