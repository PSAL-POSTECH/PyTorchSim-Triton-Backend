"""One element read and one written, x[0] + 1 as f32 (validation/vector's slice): each is a one-element
transfer through the scratchpad, as every DRAM access is -- not a core load or store of the DRAM, which the
TOG would not see, so the kernel would time as a few cycles of compute."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_scalar_access_is_a_transfer_kernel(x_ptr, o_ptr):
        tl.store(o_ptr, tl.load(x_ptr).to(tl.float32) + 1.0)

    return ops_scalar_access_is_a_transfer_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32).to(torch.bfloat16)}


def reference(inputs):
    return {"o": (inputs["x"][:1].float() + 1).reshape(1)}


SPEC = KernelSpec(
    name="ops_scalar_access_is_a_transfer_kernel", kernel=kernel,
    signature={"x_ptr": "*bf16", "o_ptr": "*fp32"},
    constexprs={},
    args=[Arg("x", IN, "bfloat16", (N,)), Arg("o", OUT, "float32", (1,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
    expect_ir={"09-custom.mlir": {"torchsim.transfer": 2}},
)
