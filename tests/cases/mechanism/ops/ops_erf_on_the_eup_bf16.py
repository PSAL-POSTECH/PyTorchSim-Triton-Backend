"""The exact GELU x / 2 * (1 + erf(x / sqrt 2)), bf16 in and out, its erf pushed to the EUP and popped, unpipelined:
the x after the pop is the bf16 unpack's, which schedule_vpu_stream cannot make again (misc.unpack_* state no effects).
Breaks if erf goes back to its sf.vc.v.iv, or the push's field is lost; pipelining it moves the counts to 3."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 32768
EUP_COUNTS = {'"llvm.riscv.sf.vc.vv.se"': 1, '"llvm.riscv.sf.vc.v.x.se"': 1, "pytorchsim.eup_pipeline": 1}


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_erf_on_the_eup_bf16_kernel(x_ptr, y_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        x = tl.load(x_ptr + r).to(tl.float32)
        tl.store(y_ptr + r, (0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))).to(tl.bfloat16))

    return ops_erf_on_the_eup_bf16_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": (torch.randn(N, generator=g, dtype=torch.float32) * 2).to(torch.bfloat16)}


def reference(inputs):
    import torch
    x = inputs["x"].float()
    return {"y": (torch.nn.functional.gelu(x)).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="ops_erf_on_the_eup_bf16_kernel", kernel=kernel,
    signature={"x_ptr": "*bf16", "y_ptr": "*bf16", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "bfloat16", (N,)), Arg("y", OUT, "bfloat16", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
    expect_ir={"09-custom.mlir": EUP_COUNTS},
)
