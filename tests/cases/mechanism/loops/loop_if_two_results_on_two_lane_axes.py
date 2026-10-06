"""An scf.if with two tile results whose sums want different lane axes, its arms of different
sizes: each result is its own carry, and the larger arm's layout is the one the result takes.
Breaks if a condition's results are tied to one layout, or its smaller arm decides.

Two scf.if results whose sums want different lane axes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C = 16, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_if_two_results_on_two_lane_axes_kernel(
            x_ptr, f_ptr, a_ptr, b_ptr, R: tl.constexpr, C: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :])
        if tl.load(f_ptr) > 0:
            a = x * 2.0 + 1.0
            b = tl.exp(x) * 3.0 - x
        else:
            a = x
            b = x + 1.0
        tl.store(a_ptr + tl.arange(0, R), tl.sum(a, axis=1))
        tl.store(b_ptr + tl.arange(0, C), tl.sum(b, axis=0))

    return loop_if_two_results_on_two_lane_axes_kernel


def make_inputs(torch, seed=0):
    return {"x": torch.randn(R, C, generator=torch.Generator().manual_seed(seed)),
            "f": torch.ones(1)}


def reference(inputs):
    x = inputs["x"]
    return {"a": (x * 2.0 + 1.0).sum(1), "b": (torch_exp(x) * 3.0 - x).sum(0)}


def torch_exp(x):
    import torch
    return torch.exp(x)


SPEC = KernelSpec(
    name="loop_if_two_results_on_two_lane_axes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "f_ptr": "*fp32", "a_ptr": "*fp32", "b_ptr": "*fp32",
               "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (R, C)), Arg("f", IN, "float32", (1,)),
          Arg("a", OUT, "float32", (R,)), Arg("b", OUT, "float32", (C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-3,
)
