"""A scalar write under a mask, x[pid] + 1 written by program 0 alone: one one-element write bounded by the
mask, [0, mask ? 1 : 0), as a masked tile write is bounded -- no scf.if, which a TOG cannot state, and no read
of the element -- so program 1 writes nothing and its element stays as it was."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, INOUT

N = 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_scalar_masked_store_is_a_transfer_kernel(x_ptr, o_ptr):
        pid = tl.program_id(0)
        tl.store(o_ptr + pid, tl.load(x_ptr + pid).to(tl.float32) + 1.0, mask=pid == 0)

    return ops_scalar_masked_store_is_a_transfer_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32).to(torch.bfloat16),
            "o": torch.full((2,), -7.0, dtype=torch.float32)}


def reference(inputs):
    o = inputs["o"].clone()
    o[0] = inputs["x"][0].float() + 1
    return {"o": o}


SPEC = KernelSpec(
    name="ops_scalar_masked_store_is_a_transfer_kernel", kernel=kernel,
    signature={"x_ptr": "*bf16", "o_ptr": "*fp32"},
    constexprs={},
    args=[Arg("x", IN, "bfloat16", (N,)), Arg("o", INOUT, "float32", (2,))],
    grid=(2,), reference=reference, make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
    expect_ir={"09-custom.mlir": {"pytorchsim.transfer": 2, "scf.if": 0}},
)
