"""math.powf on a NEGATIVE base with integer-valued exponents: real, and signed by the
exponent's parity. Breaks if powf goes to upstream's math-expand-ops, which is exp(y*log x)
alone and answers NaN for every negative base."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    globals()["libdevice"] = libdevice

    @triton.jit
    def ops_math_powf_negative_base_kernel(x_ptr, y_ptr, o_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        tl.store(o_ptr + r, libdevice.pow(tl.load(x_ptr + r), tl.load(y_ptr + r)))

    return ops_math_powf_negative_base_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    mag = torch.rand(N, generator=g, dtype=torch.float32) * 2.0 + 0.5
    sign = torch.where(torch.rand(N, generator=g) < 0.5, -1.0, 1.0)
    y = torch.randint(-3, 4, (N,), generator=g).to(torch.float32)
    return {"x": (mag * sign).contiguous(), "y": y.contiguous()}


def reference(inputs):
    import torch
    return {"o": torch.pow(inputs["x"].double(), inputs["y"].double()).float()}


SPEC = KernelSpec(
    name="ops_math_powf_negative_base_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
