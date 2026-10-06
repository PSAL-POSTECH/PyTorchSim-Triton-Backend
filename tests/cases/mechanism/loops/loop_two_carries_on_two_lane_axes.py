"""Two loop carries of one rank that want different lane axes: one is summed over axis 1, the
other over axis 0, so each rides the axis its sum keeps. Breaks if every same-rank carry of a
loop is tied to one layout (a loop's operands are carries, not siblings).

Two same-rank loop carries whose sums want different lane axes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, STEPS = 16, 32, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_two_carries_on_two_lane_axes_kernel(
            x_ptr, a_ptr, b_ptr, R: tl.constexpr, C: tl.constexpr, STEPS: tl.constexpr):
        r = tl.arange(0, R)[:, None]
        c = tl.arange(0, C)[None, :]
        acc_a = tl.zeros((R, C), dtype=tl.float32)
        acc_b = tl.zeros((R, C), dtype=tl.float32)
        for k in range(STEPS):
            x = tl.load(x_ptr + k * R * C + r * C + c)
            acc_a += x
            acc_b += x * x
        tl.store(a_ptr + tl.arange(0, R), tl.sum(acc_a, axis=1))
        tl.store(b_ptr + tl.arange(0, C), tl.sum(acc_b, axis=0))

    return loop_two_carries_on_two_lane_axes_kernel


def make_inputs(torch, seed=0):
    return {"x": torch.randn(STEPS, R, C, generator=torch.Generator().manual_seed(seed))}


def reference(inputs):
    x = inputs["x"]
    return {"a": x.sum(0).sum(1), "b": (x * x).sum(0).sum(0)}


SPEC = KernelSpec(
    name="loop_two_carries_on_two_lane_axes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "a_ptr": "*fp32", "b_ptr": "*fp32",
               "R": "constexpr", "C": "constexpr", "STEPS": "constexpr"},
    constexprs={"R": R, "C": C, "STEPS": STEPS},
    args=[Arg("x", IN, "float32", (STEPS, R, C)), Arg("a", OUT, "float32", (R,)),
          Arg("b", OUT, "float32", (C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
)
