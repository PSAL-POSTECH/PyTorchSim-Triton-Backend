"""bf16 in, f32 arithmetic, bf16 out: vfwcvtbf16.f.f.v and vfncvtbf16.f.f.w (Zvfbfmin) on a vector,
compared bit-exact with torch. Breaks on a Spike without Zvfbfmin or without zvfbfmin in its --isa
(illegal instruction, measured on 670ae21), and a wrong rounding or NaN shows as a mismatch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_bf16_convert_elementwise_kernel(x_ptr, o_ptr, N: tl.constexpr):
        i = tl.arange(0, N)
        v = tl.load(x_ptr + i).to(tl.float32)
        tl.store(o_ptr + i, (v * 1.7 + 0.5).to(tl.bfloat16))

    return ops_bf16_convert_elementwise_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, generator=g, dtype=torch.float32)
    x[0], x[1], x[2], x[3] = float("nan"), float("inf"), -float("inf"), 3.0e38
    return {"x": x.to(torch.bfloat16)}


def reference(inputs):
    import torch
    return {"o": (inputs["x"].to(torch.float32) * 1.7 + 0.5).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="ops_bf16_convert_elementwise_kernel", kernel=kernel,
    signature={"x_ptr": "*bf16", "o_ptr": "*bf16", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "bfloat16", (N,)), Arg("o", OUT, "bfloat16", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=0, atol=0,
)
