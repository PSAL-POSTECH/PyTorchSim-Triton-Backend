"""The same overrun as its sibling, with a WRAPPED axis beside the affine one.

    mask_unstructured_read_leaves_the_buffer   rows affine, cols affine
    this file                                  rows affine, cols WRAPPED

WHAT IT IS FOR. The argument's own extent folds into the box on the axis that
overruns -- and that axis is affine here, while the axis next to it is not. A
rule that asks "does anything in this access wrap?" declines and the read leaves
the buffer; a rule that asks "does THIS axis walk with a stride?" folds, because
the wrap is on the other one. The two are the same everywhere else in the suite,
which is why the distinction needs a file of its own (rule 2).

    x holds LIVE * STRIDE elements
    the tile asks for ROWS rows at STRIDE, so row LIVE starts exactly at the end
    the column index is `c % WRAP`, so the column axis comes back round

`flag` is loaded, so MaskAnalysis cannot fold it into an extent; it is the only
thing that says rows past LIVE must not be read, and it is emptied into a select
after the box is folded. The wrap keeps every column inside WRAP of the row's
start, which is why the row axis is the one that has to be narrowed.

torch equivalent: rows where flag is false come back 0; the rest read the row's
first WRAP elements, cycled to COLS.

The buffer folds into the box on the AFFINE axis while the axis beside it wraps. Asking the question of the access refuses; asking it of the axis folds.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

LIVE = 8          # rows x actually holds
ROWS = 64         # rows the tile asks for
COLS = 16
STRIDE = 4096
WRAP = 5          # the column index cycles through this many


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_read_leaves_the_buffer_beside_a_wrap_kernel(
            x_ptr, f_ptr, o_ptr, ROWS: tl.constexpr, COLS: tl.constexpr,
            STRIDE: tl.constexpr, WRAP: tl.constexpr):
        r = tl.arange(0, ROWS)
        c = tl.arange(0, COLS)
        flag = tl.load(f_ptr + r) != 0
        v = tl.load(x_ptr + r[:, None] * STRIDE + (c % WRAP)[None, :],
                    mask=flag[:, None], other=0.0)
        tl.store(o_ptr + r[:, None] * COLS + c[None, :], v)

    return mask_read_leaves_the_buffer_beside_a_wrap_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.zeros(ROWS, dtype=torch.int32)
    f[:LIVE] = 1
    return {"x": torch.randn(LIVE * STRIDE, generator=g, dtype=torch.float32),
            "f": f}


def reference(inputs):
    import torch
    out = torch.zeros(ROWS, COLS, dtype=torch.float32)
    rows = inputs["x"].reshape(LIVE, STRIDE)
    cols = torch.arange(COLS) % WRAP
    out[:LIVE] = rows[:, cols]
    return {"o": out}


SPEC = KernelSpec(
    name="mask_read_leaves_the_buffer_beside_a_wrap_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "f_ptr": "*i32", "o_ptr": "*fp32",
               "ROWS": "constexpr", "COLS": "constexpr",
               "STRIDE": "constexpr", "WRAP": "constexpr"},
    constexprs={"ROWS": ROWS, "COLS": COLS, "STRIDE": STRIDE, "WRAP": WRAP},
    args=[Arg("x", IN, "float32", (LIVE * STRIDE,)),
          Arg("f", IN, "int32", (ROWS,)),
          Arg("o", OUT, "float32", (ROWS, COLS))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
