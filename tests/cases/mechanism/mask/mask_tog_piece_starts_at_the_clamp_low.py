"""A wrapped row walk, rows `(i * 8) % 24`, cut into three pieces, each masked from below on its columns
(`x_ptr + row * 40 + ((-24) + j)` with `j >= 24`): each piece's TOG DMA reads its rows' 40 live columns
from its row start. Breaks if a piece keeps its unmasked size and start (tog.cpp names -24)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

WR, N, ST, C, CL = 24, 8, 8, 64, 24
W = C - CL


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_tog_piece_starts_at_the_clamp_low(x_ptr, y_ptr, WR: tl.constexpr, N: tl.constexpr,
                                               ST: tl.constexpr, C: tl.constexpr, CL: tl.constexpr,
                                               W: tl.constexpr):
        i = tl.arange(0, N)[:, None]
        j = tl.arange(0, C)[None, :]
        row = (i * ST) % WR
        v = tl.load(x_ptr + row * W + ((-CL) + j), mask=j >= CL, other=0.0)
        tl.store(y_ptr + i * C + j, v)

    return mask_tog_piece_starts_at_the_clamp_low


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(WR, W, generator=g)}


def reference(inputs):
    import torch
    y = torch.zeros(N, C)
    y[:, CL:] = inputs["x"][(torch.arange(N) * ST) % WR]
    return {"y": y}


SPEC = KernelSpec(
    name="mask_tog_piece_starts_at_the_clamp_low", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "WR": "constexpr", "N": "constexpr", "ST": "constexpr",
               "C": "constexpr", "CL": "constexpr", "W": "constexpr"},
    constexprs={"WR": WR, "N": N, "ST": ST, "C": C, "CL": CL, "W": W},
    args=[Arg("x", IN, "float32", (WR, W)), Arg("y", OUT, "float32", (N, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"09-custom.mlir": {"piece_box": 3}, "tog.cpp": {"-24": 0}}, expect_stage="togcheck",
)
