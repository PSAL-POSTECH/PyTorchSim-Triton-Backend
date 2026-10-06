"""The edge of the assumption a dense read plus a select rests on.

    User load segfault @ 0x00000000d8025010

STRUCTURED + ELEMENTWISE MASK IS LOWERED THE CHEAP WAY, and that is the right
default: the tile comes in as one strided burst, the mask tensor is already
there, and the masked-off lanes are chosen away afterwards. No per-element
address, no indirect DMA, no index buffer.

    tts.make_tptr(...)                     sizes = [64, 16]
    tts.load(%p, %fill)                    static_mask_dims = [64, 16]
    linalg.generic { arith.select }        the mask, on values

IT IS SOUND WHILE EVERY MASKED-OFF ADDRESS IS INSIDE THE BUFFER, which is what
this backend assumes of a structured tile. A mask that bounds a tail folds into
the BOX -- static_mask_dims, a prefix per axis -- and shrinks the read; what is
left as an `elem_mask` is a mask that picks WITHIN a tile the buffer already
holds. `offs < n` never reaches here.

THIS KERNEL IS BUILT TO BREAK THAT. It declares `x` with room for 8 rows and
asks for 64 at a 4096-element stride, so nothing but `flag` -- loaded from
memory, so MaskAnalysis cannot fold it -- says the other 56 must not be read.
Row 63 is 258048 elements past the base. The DMA walks all 64.

GREEN SINCE 2026-09-05, err 0, AND IT COST NOTHING. The 2026-09-01 attempt kept
the mask ON the load, which materialises every address and turns six kernels
from a strided burst into an indirect transfer; it was reverted, rightly. The
question it asked was "does the tile fit the buffer?", which needs the base and
so cannot be answered while the base is program-id dependent.

The question that CAN be answered is the other one: the ARGUMENT'S OWN EXTENT.
An address past the end of an argument is not one the program meant to touch,
whatever the mask says -- so it folds into the box exactly as a comparison mask
does, and what is left of the mask is still the select. Here that turns

    static_mask_dims = [64, 16]   ->   [8, 16]

because row 8 starts at 32768 and the buffer holds 32768. Rows 8..63 are not
read and take `other`; the select still applies `flag`. No index tile, no
indirect transfer, no extra round trip -- one number in the descriptor.

It is folded where the mask is EMPTIED, which is where the sentence above
already said it belonged, and the question is asked OF THE AXIS: a wrapped axis
comes back round, so it reaches `(min(n, W) - 1) * stride` and has no tail to
cut, but the affine axis beside it still folds --
mask_read_leaves_the_buffer_beside_a_wrap is that shape. Measured: asking it of
the ACCESS instead leaves that one open, and asking it in the shared transfer
tail breaks 18 kernels, 11 `axis is both wrapped and clamped` and 7 wrong.

A GATHERED ADDRESS CANNOT MAKE THE ASSUMPTION AT ALL and does not: its index
comes out of memory, so _rewrite_gather and _rewrite_gs_mask clamp masked-off
indices to 0 before the read. That is not this, and the sibling
mask_gs_row_mask_offset_out_of_range is where it is pinned.

torch equivalent: rows where flag is false read nothing and come back 0.

An unstructured mask on a STRUCTURED pointer. Pins that the select flag is sound only where the masked-off address is in bounds.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

LIVE = 8          # rows x actually holds
ROWS = 64         # rows the tile asks for
COLS = 16
# Row stride, so the overrun clears the arena rather than landing in the next
# buffer -- at COLS the read stays mapped, returns garbage, and the select
# throws it away, which is silent rather than a signal.
STRIDE = 4096


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_unstructured_read_leaves_the_buffer_kernel(
            x_ptr, f_ptr, o_ptr, ROWS: tl.constexpr, COLS: tl.constexpr,
            STRIDE: tl.constexpr):
        r = tl.arange(0, ROWS)
        c = tl.arange(0, COLS)
        # Loaded, then broadcast with [:, None]: MaskAnalysis's unstructured
        # pattern, and it cannot become an extent.
        flag = tl.load(f_ptr + r) != 0
        # AFFINE, so the pointer stays structured -- no gather anywhere.
        v = tl.load(x_ptr + r[:, None] * STRIDE + c[None, :],
                    mask=flag[:, None], other=0.0)
        tl.store(o_ptr + r[:, None] * COLS + c[None, :], v)

    return mask_unstructured_read_leaves_the_buffer_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.zeros(ROWS, dtype=torch.int32)
    f[:LIVE] = 1
    return {"x": torch.randn(LIVE * STRIDE, generator=g, dtype=torch.float32),
            "f": f}


def reference(inputs):
    import torch
    out = torch.zeros(ROWS, COLS, dtype=torch.float32)
    out[:LIVE] = inputs["x"].reshape(LIVE, STRIDE)[:, :COLS]
    return {"o": out}


SPEC = KernelSpec(
    name="mask_unstructured_read_leaves_the_buffer_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "f_ptr": "*i32", "o_ptr": "*fp32",
               "ROWS": "constexpr", "COLS": "constexpr",
               "STRIDE": "constexpr"},
    constexprs={"ROWS": ROWS, "COLS": COLS, "STRIDE": STRIDE},
    args=[Arg("x", IN, "float32", (LIVE * STRIDE,)),
          Arg("f", IN, "int32", (ROWS,)),
          Arg("o", OUT, "float32", (ROWS, COLS))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
