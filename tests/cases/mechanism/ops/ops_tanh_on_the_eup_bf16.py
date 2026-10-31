"""gemma's softcap 30 * tanh(x / 30), bf16 in and out, its tanh pushed to the EUP two groups ahead of its pops where
its sf.vc.v.iv was (the unpacks stay sf.vc.v.iv). Breaks if tanh goes back to its sf.vc.v.iv (the counts), or the
push's field is lost (the model then pushes 2^x: the values)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 32768
EUP_COUNTS = {'"llvm.riscv.sf.vc.vv.se"': 3, '"llvm.riscv.sf.vc.v.x.se"': 3, "pytorchsim.eup_pipeline": 0}


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def ops_tanh_on_the_eup_bf16_kernel(x_ptr, y_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        x = tl.load(x_ptr + r).to(tl.float32)
        tl.store(y_ptr + r, (30.0 * libdevice.tanh(x / 30.0)).to(tl.bfloat16))

    return ops_tanh_on_the_eup_bf16_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": (torch.randn(N, generator=g, dtype=torch.float32) * 2).to(torch.bfloat16)}


def reference(inputs):
    import torch
    x = inputs["x"].float()
    return {"y": (30.0 * torch.tanh(x / 30.0)).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="ops_tanh_on_the_eup_bf16_kernel", kernel=kernel,
    signature={"x_ptr": "*bf16", "y_ptr": "*bf16", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "bfloat16", (N,)), Arg("y", OUT, "bfloat16", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
    expect_ir={"09-custom.mlir": EUP_COUNTS},
)
