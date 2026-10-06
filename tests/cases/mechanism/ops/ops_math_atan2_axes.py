"""atan2 on the axes: the quadrant is signbit(x), not `x < 0`, and x = 0 is
selected, not inherited from atan(+-inf) -- before, 16/128 lanes wrong.
Breaks it: going back to `x < 0.0`, or dropping the explicit x = 0 select.

convert_unsupported_ops, POSITIVE. The 16 (y, x) pairs are every combination of {+0.0, -0.0, +1.5, -1.5} on both arguments, repeated to fill the block: the whole axis behaviour plus the four open quadrants. MEASURED BEFORE: 16/128 elements outside tolerance, worst at index 1 got NaN want 3.1415927. Two bugs at once -- atan2(1.0, -0.0) came back -pi/2 instead of +pi/2 because -0.0 < 0.0 is FALSE so no quadrant shift ran, and atan2(0.0, -0.0) came back NaN instead of +pi because 0/-0 is NaN and the origin test is bitwise. AFTER: PASS, max_abs_err 5.96e-08. _derive_pow_from_exp_log already tested the sign with signbit for exactly this reason and said so; atan2 denominator is the same question about the same value, so the file now holds one policy instead of two. The axis value is selected rather than inherited from the machine atan(+-inf) -- the dependency _derive_asin_from_atan had already refused. No correct answer here is NaN, which matters: tests/runtime/verify.py has no equal_nan.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
BLOCK = 128

_PAIRS = [(y, x) for y in (0.0, -0.0, 1.5, -1.5)
          for x in (0.0, -0.0, 1.5, -1.5)]


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    globals()["libdevice"] = libdevice

    @triton.jit
    def ops_math_atan2_axes_kernel(y_ptr, x_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        y = tl.load(y_ptr + r)
        x = tl.load(x_ptr + r)
        tl.store(o_ptr + r, libdevice.atan2(y, x))

    return ops_math_atan2_axes_kernel


def make_inputs(torch, seed=0):
    reps = N // len(_PAIRS)
    y = torch.tensor([p[0] for p in _PAIRS] * reps, dtype=torch.float32)
    x = torch.tensor([p[1] for p in _PAIRS] * reps, dtype=torch.float32)
    return {"y": y.contiguous(), "x": x.contiguous()}


def reference(inputs):
    import torch
    return {"o": torch.atan2(inputs["y"], inputs["x"]).contiguous()}


SPEC = KernelSpec(
    name="ops_math_atan2_axes_kernel",
    kernel=kernel,
    signature={"y_ptr": "*fp32", "x_ptr": "*fp32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("y", IN, "float32", (N,)), Arg("x", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-6, atol=1e-6,
)
