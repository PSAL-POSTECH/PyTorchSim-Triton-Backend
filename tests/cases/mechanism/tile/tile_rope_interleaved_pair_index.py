"""Interleaved RoPE: a pair read with `2*(i//2)` and `2*(i//2)+1`.

    tile   32 rows x 64 columns, the rotary half of a 512-wide head
    index  even element at 2*(x//2), odd at 2*(x//2)+1

WHAT IT PINS. The rotary halves are INTERLEAVED, so the kernel reads the pair
(2k, 2k+1) for column x = 2k and x = 2k+1 alike -- the same two loads feed both
columns, and which of the two is negated is decided by x's parity. The index
that says so is not affine: `2*(x//2)` is a floor-divide inside an address, and
the axis has to be split for it to be one.

    tmp32 = tl.load(in_ptr + (448 + 2*(x1 // 2) + 512*x2 + 2048*y0))
    tmp18 = tl.load(in_ptr + (449 + 2*(x1 // 2) + 512*x2 + 2048*y0))

WHERE IT CAME FROM, and it PASSES -- which is the point. DeepSeek-V4-Flash-Base
diverged in exactly this kernel shape, so this was written to reproduce it and
did not. That cleared the pairing arithmetic and sent the search to the address,
where the cause turned out to be a constant base dropped off a split axis
(the linalg lowering's rebuildAsGatherScatter; see ../ptr/ptr_const_offset_with_
split_axis.py). It stays because the interleaved convention is real and nothing
else in the suite reads a pair off `2*(x//2)`.

WHAT BREAKS IT AGAIN: anything that treats `2*(x//2)` as affine in x, or that
splits the axis without carrying the +1 of the odd load with it. A rotary that
uses the HALF-SPLIT convention (first 32 columns against last 32) reads
`x + 32` instead and never exercises this.

Interleaved rotary: both halves of the pair addressed off 2*(x//2), parity deciding the sign. DeepSeek-V4's first divergent kernel is this shape.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS, COLS = 32, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_rope_interleaved_pair_index_kernel(
            x_ptr, cos_ptr, sin_ptr, out_ptr,
            ROWS: tl.constexpr, COLS: tl.constexpr):
        y = tl.arange(0, ROWS)[:, None]
        x = tl.arange(0, COLS)[None, :]

        # The pair, both halves addressed off the SAME floor-divided column.
        even = tl.load(x_ptr + (2 * (x // 2) + COLS * y))
        odd = tl.load(x_ptr + (1 + 2 * (x // 2) + COLS * y))

        c = tl.load(cos_ptr + (x // 2 + (COLS // 2) * y))
        s = tl.load(sin_ptr + (x // 2 + (COLS // 2) * y))

        # Parity picks which half of the pair this column is, and which one is
        # negated -- the whole reason both loads are needed for every column.
        is_even = (x % 2) == 0
        rotated = tl.where(is_even, even * c - odd * s, odd * c + even * s)
        tl.store(out_ptr + (x + COLS * y), rotated)

    return tile_rope_interleaved_pair_index_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(ROWS, COLS, generator=g, dtype=torch.float32),
            "cos": torch.randn(ROWS, COLS // 2, generator=g, dtype=torch.float32),
            "sin": torch.randn(ROWS, COLS // 2, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x, c, s = inputs["x"], inputs["cos"], inputs["sin"]
    even = x[:, 0::2]
    odd = x[:, 1::2]
    out = torch.empty_like(x)
    out[:, 0::2] = even * c - odd * s
    out[:, 1::2] = odd * c + even * s
    return {"out": out}


SPEC = KernelSpec(
    name="tile_rope_interleaved_pair_index_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "cos_ptr": "*fp32", "sin_ptr": "*fp32",
               "out_ptr": "*fp32", "ROWS": "constexpr", "COLS": "constexpr"},
    constexprs={"ROWS": ROWS, "COLS": COLS},
    args=[Arg("x", IN, "float32", (ROWS, COLS)),
          Arg("cos", IN, "float32", (ROWS, COLS // 2)),
          Arg("sin", IN, "float32", (ROWS, COLS // 2)),
          Arg("out", OUT, "float32", (ROWS, COLS))],
    grid=(1,),
    reference=reference, make_inputs=make_inputs,
)
