"""exp(x) written as bf16: y has a buffer of its own, so the EUP loop's pushes run ahead of its pops,
two groups deep (schedule_vpu_stream): a prologue's push, the loop's, and the epilogue's pops."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 32768
# Three pushes and three pops (prologue, loop, epilogue), no loop that keeps its order.
EUP_COUNTS = {"\"llvm.riscv.sf.vc.vv.se\"": 3, "\"llvm.riscv.sf.vc.v.x.se\"": 3, "torchsim.eup_pipeline": 0}


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_exp_on_the_eup_pipelined_kernel(x_ptr, y_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        tl.store(y_ptr + r, tl.exp(tl.load(x_ptr + r)).to(tl.bfloat16))

    return ops_exp_on_the_eup_pipelined_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"y": inputs["x"].exp().to(torch.bfloat16)}


SPEC = KernelSpec(
    name="ops_exp_on_the_eup_pipelined_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*bf16", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", OUT, "bfloat16", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
    expect_ir={"09-custom.mlir": EUP_COUNTS},
)
