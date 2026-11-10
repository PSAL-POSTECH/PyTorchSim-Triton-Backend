"""Rows masked `i >= lo`, lo a scalar argument: the TOG builds the start (dram_idx + max(lo, 0) * 32) and
the extent at run time; togcheck judges the footprint. expect_ir counts tog.cpp's `+` (the grid loop's and
the start's): breaks if the start is the unmasked dram_idx again, as before the start shift."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, LO = 16, 32, 5


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_tog_runtime_low_from_a_scalar(x_ptr, y_ptr, lo, R: tl.constexpr, C: tl.constexpr):
        i = tl.arange(0, R)[:, None]
        j = tl.arange(0, C)[None, :]
        v = tl.load(x_ptr + (i - lo) * C + j, mask=i >= lo, other=0.0)
        tl.store(y_ptr + i * C + j, v)

    return mask_tog_runtime_low_from_a_scalar


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R - LO, C, generator=g)}


def reference(inputs):
    import torch
    y = torch.zeros(R, C)
    y[LO:] = inputs["x"]
    return {"y": y}


SPEC = KernelSpec(
    name="mask_tog_runtime_low_from_a_scalar", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "lo": "i32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (R - LO, C)), Arg("y", OUT, "float32", (R, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_values": {"lo": LO}},
    expect_ir={"tog.cpp": {"+": 2}}, expect_stage="togcheck",
)
