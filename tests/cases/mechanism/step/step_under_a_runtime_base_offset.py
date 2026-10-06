"""GRID: `(off + p) // K` where `off` is a RUNTIME scalar. Nobody here knows where the runs fall.

An offset the divisor does not divide shifts the first run, and a runtime one
means its length is unknown at compile time -- so the division cannot be turned
into a step and the access goes per-element.

A runtime base offset under the division: the runs are not known here, so the address is per-element.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K, OFF = 32, 11, 3        # K does not divide N -- the step slot's shape
NX = (OFF + N - 1) // K + 1


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def step_under_a_runtime_base_offset_kernel(x_ptr, o_ptr, off,
                                                N: tl.constexpr,
                                                K: tl.constexpr):
        p = tl.arange(0, N)
        v = tl.load(x_ptr + (off + p) // K)
        tl.store(o_ptr + p, v)

    return step_under_a_runtime_base_offset_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    p = torch.arange(N)
    return {"o": inputs["x"][(OFF + p) // K]}


SPEC = KernelSpec(
    name="step_under_a_runtime_base_offset_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "off": "i32",
               "N": "constexpr", "K": "constexpr"},
    constexprs={"N": N, "K": K},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("off", "int32_t")], "scalar_values": {"off": OFF}},
)
