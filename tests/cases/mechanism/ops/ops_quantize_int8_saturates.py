"""torch's int8 quantize, x * inv_scale rounded to nearest even, clamped to [-128, 127] and cast: the clamp is the
int8's own range, so lower_round_convert converts first and clamps the int32 (RVV's vnclip narrows it saturating)
instead of a Misc minimum and maximum against two f32 constants each trip. Breaks if the f32 clamp comes back:
the vcix minimum / maximum reappear in 09-custom."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 1024


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def ops_quantize_int8_saturates_kernel(x_ptr, s_ptr, q_ptr, M: tl.constexpr, N: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, N)[None, :]
        x = tl.load(x_ptr + rows * N + cols)
        q = libdevice.nearbyint(x * tl.load(s_ptr + rows))
        q = tl.minimum(tl.maximum(q, -128.0, tl.PropagateNan.ALL), 127.0, tl.PropagateNan.ALL)
        tl.store(q_ptr + rows * N + cols, q.to(tl.int8))

    return ops_quantize_int8_saturates_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32) * 3,
            "s": torch.rand(M, generator=g, dtype=torch.float32) * 80 + 1}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(M, N) * inputs["s"][:, None]
    return {"q": x.round().clamp(-128, 127).to(torch.int8).reshape(-1)}


SPEC = KernelSpec(
    name="ops_quantize_int8_saturates_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "s_ptr": "*fp32", "q_ptr": "*i8", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M * N,)), Arg("s", IN, "float32", (M,)), Arg("q", OUT, "int8", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=0, atol=0,
    expect_ir={"09-custom.mlir": {"vcix.v.sv": 0, "arith.maxsi": 1, "arith.minsi": 1}},
)
