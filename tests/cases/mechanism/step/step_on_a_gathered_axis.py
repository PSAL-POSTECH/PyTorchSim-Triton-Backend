"""GRID: a stepped axis that also GATHERS. The index is data, so the step has no run to stand on.

`(p // K)*A` added to an index the kernel LOADS puts two index functions on one
axis, and the data-dependent one wins: the whole address goes per-element.

A step and a gather on one axis: the address is per-element.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K, A = 32, 11, 4          # K does not divide N -- the step slot's shape
NX = ((N - 1) // K) * A + N + 1


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def step_on_a_gathered_axis_kernel(x_ptr, i_ptr, o_ptr, N: tl.constexpr,
                                       K: tl.constexpr, A: tl.constexpr):
        p = tl.arange(0, N)
        idx = tl.load(i_ptr + p)
        v = tl.load(x_ptr + idx + (p // K) * A)
        tl.store(o_ptr + p, v)

    return step_on_a_gathered_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "i": torch.randperm(N, generator=g).to(torch.int32)}


def reference(inputs):
    import torch
    p = torch.arange(N)
    return {"o": inputs["x"][inputs["i"].long() + (p // K) * A]}


SPEC = KernelSpec(
    name="step_on_a_gathered_axis_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "N": "constexpr", "K": "constexpr", "A": "constexpr"},
    constexprs={"N": N, "K": K, "A": A},
    args=[Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
