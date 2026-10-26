"""An f32 exp on the EUP: a push of its argument (sf.vc.vv opcode 3) and a pop of its answer (sf.vc.v.x
opcode 3), the pushes two groups ahead of the pops. tpuv6e's EUP holds 26 registers (tpu_sfu_queue_registers
16 + tpu_sfu_latency_cycles 10), three groups of LMUL 8, so the loop keeps the kernel's LMUL. Without the
SFU's exp (math_late_cases pops it) the exp stays math.exp after transform and is refused by name. Here y
is written over x's buffer: every trip reads and writes only its own elements, so the loop is still
pipelined (eup_pipeline_cases pins the loops that are not)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 32768
# No math.exp left, three pushes and three pops (prologue, loop, epilogue), and no loop that keeps its order.
EUP_COUNTS = {'"llvm.riscv.sf.vc.vv.se"': 3, '"llvm.riscv.sf.vc.v.x.se"': 3, "math.exp": 0,
              "pytorchsim.eup_pipeline": 0}


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_exp_on_the_eup_kernel(x_ptr, y_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        tl.store(y_ptr + r, tl.exp(tl.load(x_ptr + r)))

    return ops_exp_on_the_eup_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"y": inputs["x"].exp()}


SPEC = KernelSpec(
    name="ops_exp_on_the_eup_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-transformed.mlir": {"math.exp": 1}, "09-custom.mlir": EUP_COUNTS},
)
