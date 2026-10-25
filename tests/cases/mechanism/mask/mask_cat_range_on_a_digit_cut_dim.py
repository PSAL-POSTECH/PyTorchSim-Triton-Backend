"""A torch.cat range masked on a dim wider than the lanes, which a03 cuts into digits.

Inductor's rotate_half (Gemma's RoPE) and channel cats (Yolo) read each part of a row of 256 with a
mask on the column, x0 < 128 and x0 >= 128. The row is laid as 2 laps of 128 lanes, so a03 cuts the
column into digits (2, 128) and the DMA's mask must move onto them. [0, 128) and [128, 256) cut whole
runs of 128: they bound the lap digit alone. [192, 256) lies inside the second run: it bounds the
lap digit to [1, 2) and the lane digit to [64, 128). a03 refused every bound but [0, k * 128).

torch equivalent: o = cat(-x[:, 128:], x[:, :128], dim=1); w = cat(FILL x 192, x[:, :64], dim=1).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS = 8
COLS = 256
HALF = 128
TAIL = 192
FILL = -3.0


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_cat_range_on_a_digit_cut_dim(x_ptr, o_ptr, w_ptr, XBLOCK: tl.constexpr,
                                          COLS: tl.constexpr, HALF: tl.constexpr,
                                          TAIL: tl.constexpr,
                                          FILL: tl.constexpr):
        xindex = tl.arange(0, XBLOCK)
        x0 = xindex % COLS
        x1 = xindex // COLS
        first = x0 < HALF
        a = tl.load(x_ptr + (HALF + COLS * x1 + x0), first, other=0.0)
        b = tl.load(x_ptr + (COLS * x1 + ((-HALF) + x0)), x0 >= HALF, other=0.0)
        tl.store(o_ptr + xindex, tl.where(first, -a, b))
        w = tl.load(x_ptr + (COLS * x1 + ((-TAIL) + x0)), x0 >= TAIL, other=FILL)
        tl.store(w_ptr + xindex, w)

    return mask_cat_range_on_a_digit_cut_dim


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(ROWS * COLS, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].view(ROWS, COLS)
    o = torch.cat([-x[:, HALF:], x[:, :HALF]], dim=1)
    w = torch.full_like(x, FILL)
    w[:, TAIL:] = x[:, :COLS - TAIL]
    return {"o": o.reshape(-1), "w": w.reshape(-1)}


SPEC = KernelSpec(
    name="mask_cat_range_on_a_digit_cut_dim", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "w_ptr": "*fp32", "XBLOCK": "constexpr",
               "COLS": "constexpr", "HALF": "constexpr", "TAIL": "constexpr", "FILL": "constexpr"},
    constexprs={"XBLOCK": ROWS * COLS, "COLS": COLS, "HALF": HALF, "TAIL": TAIL, "FILL": FILL},
    args=[Arg("x", IN, "float32", (ROWS * COLS,)), Arg("o", OUT, "float32", (ROWS * COLS,)),
          Arg("w", OUT, "float32", (ROWS * COLS,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
