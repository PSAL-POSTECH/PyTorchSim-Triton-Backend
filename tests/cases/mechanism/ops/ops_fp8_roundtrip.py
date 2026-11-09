"""E5M2 round trip, f32 -> f8E5M2 -> f32: tt.fp_to_fp both ways, as vfncvt.f.f.q (one rounding) and
vfwcvt.f.f.v at e8alt. Breaks if the narrowing goes through f16 (double rounding), if _SEW loses
f8E5M2, or if a machine without zvfp8 in its isa stops refusing fp8 at transform."""
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
    def ops_fp8_roundtrip_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + r)
        small = x.to(tl.float8e5)       # tt.fp_to_fp f32 -> f8E5M2  (truncate)
        back = small.to(tl.float32)     # tt.fp_to_fp f8E5M2 -> f32  (extend)
        tl.store(o_ptr + r, back)

    return ops_fp8_roundtrip_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": inputs["x"].to(torch.float8_e5m2).to(torch.float32)}


SPEC = KernelSpec(
    name="ops_fp8_roundtrip_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    expect="pass",
)
