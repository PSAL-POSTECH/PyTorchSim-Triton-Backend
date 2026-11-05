"""A [16, 64] load masked from below on both axes, rows `i >= 4` and columns `j >= 8` of an x of 12 rows
of 56: the TOG DMA reads x[0..11, 0..55] from the box's corner, 4 * 56 + 8 = 232 past the unmasked start.
Breaks if either axis's low is left out of the start (tog.cpp names -232, togcheck finds x read before 0)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, RL, CL = 16, 64, 4, 8
W = C - CL


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_tog_dma_starts_at_the_corner_of_two_clamps(x_ptr, y_ptr, R: tl.constexpr, C: tl.constexpr,
                                                        RL: tl.constexpr, CL: tl.constexpr, W: tl.constexpr):
        i = tl.arange(0, R)[:, None]
        j = tl.arange(0, C)[None, :]
        v = tl.load(x_ptr + ((-RL) + i) * W + ((-CL) + j), mask=(i >= RL) & (j >= CL), other=0.0)
        tl.store(y_ptr + i * C + j, v)

    return mask_tog_dma_starts_at_the_corner_of_two_clamps


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R - RL, W, generator=g)}


def reference(inputs):
    import torch
    y = torch.zeros(R, C)
    y[RL:, CL:] = inputs["x"]
    return {"y": y}


SPEC = KernelSpec(
    name="mask_tog_dma_starts_at_the_corner_of_two_clamps", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "R": "constexpr", "C": "constexpr",
               "RL": "constexpr", "CL": "constexpr", "W": "constexpr"},
    constexprs={"R": R, "C": C, "RL": RL, "CL": CL, "W": W},
    args=[Arg("x", IN, "float32", (R - RL, W)), Arg("y", OUT, "float32", (R, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"tog.cpp": {"-232": 0}}, expect_stage="togcheck",
)
