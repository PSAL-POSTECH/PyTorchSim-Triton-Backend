"""GRID: a stepped axis meeting a PLAIN one. `(p // K)*A + p*B` is not `(p//K)*(A+B)`.

Two index functions on one axis, so no single stride says it and the access has
to demote. This is the minimal form of the wrong answer that held the step slot
back: addState took whichever side had a step and SUMMED the strides.

One axis, two index functions: the address demotes to per-element rather than folding both terms into one stride.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# K DOES NOT DIVIDE N, which is what sends the division to the step slot rather
# than to the axis split -- the split is sound and answers this shape for free.
N, K, A = 32, 11, 8
NX = ((N - 1) // K) * A + (N - 1) + 1        # 48


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def step_and_a_plain_run_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                    K: tl.constexpr, A: tl.constexpr):
        p = tl.arange(0, N)
        v = tl.load(x_ptr + (p // K) * A + p)
        tl.store(o_ptr + p, v)

    return step_and_a_plain_run_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    p = torch.arange(N)
    return {"o": inputs["x"][(p // K) * A + p]}


SPEC = KernelSpec(
    name="step_and_a_plain_run_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "K": "constexpr", "A": "constexpr"},
    constexprs={"N": N, "K": K, "A": A},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
