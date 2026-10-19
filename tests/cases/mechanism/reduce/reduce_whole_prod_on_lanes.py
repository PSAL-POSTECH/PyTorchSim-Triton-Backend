"""A product of all 4096 elements of one row, which the lanes do not fold: the axis goes on the lanes,
each lane multiplies its 32 laps, and the 128 partials leave the lanes. Breaks if a whole-axis reduce the
lanes do not fold is walked in one lane again.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C = 1, 4096


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _mul(a, b):
        return a * b

    @triton.jit
    def reduce_whole_prod_on_lanes_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :])
        tl.store(o_ptr + tl.arange(0, R), tl.reduce(x, 1, _mul))

    return reduce_whole_prod_on_lanes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": 1.0 + 0.001 * torch.randn(R, C, generator=g)}


def reference(inputs):
    import torch
    x = inputs["x"]
    return {"o": x.prod(1)}


SPEC = KernelSpec(
    name="reduce_whole_prod_on_lanes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (R, C)), Arg("o", OUT, "float32", (R,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
)
