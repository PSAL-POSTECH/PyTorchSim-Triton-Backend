"""GRID: a wrap on ONE axis and a step on ANOTHER.

Axes are independent, so this cell must be ALLOWED. The pair that must not is a
wrap and a step on the SAME axis -- ptr_wrap_then_step_one_axis.py is the half
of that which composes (K divides W, and the wrap rides the outer half at W/K).

    r % 3   on axis 0        c // 5   on axis 1, and 5 does not divide 32

A wrap and a step on DIFFERENT axes -- independent, so allowed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, W, K, ROW = 8, 32, 3, 5, 16
NX = W * ROW + (N + K - 1) // K + 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def step_wrap_on_another_axis_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                         N: tl.constexpr, W: tl.constexpr,
                                         K: tl.constexpr, ROW: tl.constexpr):
        r = tl.arange(0, M)[:, None]
        c = tl.arange(0, N)[None, :]
        v = tl.load(x_ptr + (r % W) * ROW + c // K)
        tl.store(o_ptr + r * N + c, v)

    return step_wrap_on_another_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = torch.arange(M)[:, None]
    c = torch.arange(N)[None, :]
    return {"o": inputs["x"][(r % W) * ROW + c // K].reshape(-1)}


SPEC = KernelSpec(
    name="step_wrap_on_another_axis_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr",
               "N": "constexpr", "W": "constexpr", "K": "constexpr",
               "ROW": "constexpr"},
    constexprs={"M": M, "N": N, "W": W, "K": K, "ROW": ROW},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
