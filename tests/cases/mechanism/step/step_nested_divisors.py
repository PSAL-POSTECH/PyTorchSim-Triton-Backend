"""GRID: `(p // K1) // K2`, which is `p // (K1*K2)`.

Two divisions on one axis COMPOSE rather than refuse -- 3 then 2 is a step of
6, and the runs are the same six a single divisor of 6 gives. The alternative,
refusing the second because the axis already steps, would leave a shape that is
not even hard.

Two divisors on one axis compose into one period of 6.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K1, K2 = 32, 3, 2
NX = (N + K1 * K2 - 1) // (K1 * K2) + 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def step_nested_divisors_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                    K1: tl.constexpr, K2: tl.constexpr):
        p = tl.arange(0, N)
        v = tl.load(x_ptr + (p // K1) // K2)
        tl.store(o_ptr + p, v)

    return step_nested_divisors_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    p = torch.arange(N)
    return {"o": inputs["x"][(p // K1) // K2]}


SPEC = KernelSpec(
    name="step_nested_divisors_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "K1": "constexpr", "K2": "constexpr"},
    constexprs={"N": N, "K1": K1, "K2": K2},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
