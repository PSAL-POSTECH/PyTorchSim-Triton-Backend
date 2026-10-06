"""math.asinh far below zero, where log(x + sqrt(x*x + 1)) cancels: at x = -100 the sum is
0.005 out of two terms near 100. Breaks if asinh goes to upstream's math-expand-ops, which
writes it that way; this pass takes it on |x| and puts the sign back."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    globals()["libdevice"] = libdevice

    @triton.jit
    def ops_math_asinh_large_negative_kernel(x_ptr, o_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        tl.store(o_ptr + r, libdevice.asinh(tl.load(x_ptr + r)))

    return ops_math_asinh_large_negative_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    mag = torch.rand(N, generator=g, dtype=torch.float32) * 290.0 + 10.0
    sign = torch.where(torch.rand(N, generator=g) < 0.5, -1.0, 1.0)
    return {"x": (mag * sign).contiguous()}


def reference(inputs):
    import torch
    return {"o": torch.asinh(inputs["x"].double()).float()}


SPEC = KernelSpec(
    name="ops_math_asinh_large_negative_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
