"""An f32 log on the EUP as exp is: a push naming log (sf.vc.vv 3, field 5 at 11:7), two groups ahead of its pops,
and no sf.vc.v.iv left. Breaks if log goes back to its sf.vc.v.iv (the counts), or the push's field is lost or the
model predates feat/eup-push-fn (it then pushes 2^x: the values)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 32768
EUP_COUNTS = {'"llvm.riscv.sf.vc.vv.se"': 3, '"llvm.riscv.sf.vc.v.x.se"': 3, "vcix.v.iv": 0, "math.log": 0,
              "pytorchsim.eup_pipeline": 0}


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_log_on_the_eup_kernel(x_ptr, y_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        tl.store(y_ptr + r, tl.math.log(tl.load(x_ptr + r)))

    return ops_log_on_the_eup_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32).abs() + 0.05}


def reference(inputs):
    import torch
    x = inputs["x"]
    return {"y": x.log()}


SPEC = KernelSpec(
    name="ops_log_on_the_eup_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-transformed.mlir": {"math.log": 1}, "09-custom.mlir": EUP_COUNTS},
)
