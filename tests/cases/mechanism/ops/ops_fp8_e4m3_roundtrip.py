"""E4M3 round trip, f32 -> f8E4M3FN -> f32: the fp8 vtype does NOT mark (e8, where E5M2 is e8alt).
Breaks if fp8_to_zvfp8's _SEW swaps or drops f8E4M3FN (E5M2 rounding is off by up to 0.25 here),
if transform's FP8_DTYPES drops it, or if Spike reads altfmt the other way round."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_fp8_e4m3_roundtrip_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + r)
        small = x.to(tl.float8e4nv)     # tt.fp_to_fp f32 -> f8E4M3FN (truncate)
        back = small.to(tl.float32)     # tt.fp_to_fp f8E4M3FN -> f32 (extend)
        tl.store(o_ptr + r, back)

    return ops_fp8_e4m3_roundtrip_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": inputs["x"].to(torch.float8_e4m3fn).to(torch.float32)}


SPEC = KernelSpec(
    name="ops_fp8_e4m3_roundtrip_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    expect="pass",
)
