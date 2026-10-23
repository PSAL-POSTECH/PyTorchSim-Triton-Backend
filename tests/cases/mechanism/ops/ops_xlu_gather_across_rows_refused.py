"""A gather along a 1-D axis of 128 * 128 elements, longer than the lanes: an index may name an element
of another register, which the cross-lane unit's pattern (a lane number) cannot say. Refused by name
rather than read as a lane number (rows past the first were left unwritten).

torch equivalent: x.exp()[torch.arange(N) ^ 1] * 2 + x.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Stop

N = 128 * 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_gather_across_rows_refused_kernel(x_ptr, y_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        x = tl.load(x_ptr + r)
        tl.store(y_ptr + r, tl.gather(tl.exp(x), r ^ 1, 0) * 2 + x)

    return ops_xlu_gather_across_rows_refused_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    return {"y": x.exp()[torch.arange(N) ^ 1] * 2 + x}


SPEC = KernelSpec(
    name="ops_xlu_gather_across_rows_refused_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="refuse", expect_stop=Stop("lower", "builtins.NotImplementedError"),
    expect_why="a lane gather whose axis is longer than the lanes: the pattern names a lane, not a register",
)
