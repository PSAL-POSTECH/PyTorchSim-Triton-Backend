"""atan2 keeps ~12 vectors live; at 128 f32 a lane (LMUL 8) llc spills them. A narrower LMUL in
the loop relieves the spills, so stage 6 retries narrower; the spill slots live in the function's
.spad area, so the c.addi frame the kernel opens with plays no part."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 32768


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    globals()["libdevice"] = libdevice

    @triton.jit
    def ops_atan2_spills_at_lmul8_kernel(y_ptr, x_ptr, o_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        tl.store(o_ptr + r, libdevice.atan2(tl.load(y_ptr + r), tl.load(x_ptr + r)))

    return ops_atan2_spills_at_lmul8_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"y": torch.randn(N, generator=g, dtype=torch.float32),
            "x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.atan2(inputs["y"], inputs["x"])}


SPEC = KernelSpec(
    name="ops_atan2_spills_at_lmul8_kernel", kernel=kernel,
    signature={"y_ptr": "*fp32", "x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("y", IN, "float32", (N,)), Arg("x", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
