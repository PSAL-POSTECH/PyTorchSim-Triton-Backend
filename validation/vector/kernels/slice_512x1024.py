"""x[:1, :1] + 1 of x 512x1024 bf16 as f32: one element. Blocked as XLA blocks it (validation/vector/emit_kernels.py)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, BM, BC = 512, 1024, 1, 1


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def slice_kernel(x_ptr, o_ptr):
        tl.store(o_ptr, tl.load(x_ptr).to(tl.float32) + 1.0)

    return slice_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R * C, generator=g, dtype=torch.float32).to(torch.bfloat16)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(R, C)
    return {"o": (x[:1, :1].float() + 1).reshape(-1)}


SPEC = KernelSpec(
    name="slice_kernel",
    kernel=kernel,
    signature={"x_ptr": "*bf16", "o_ptr": "*fp32"},
    constexprs={},
    args=[Arg("x", IN, "bfloat16", (R * C,)), Arg("o", OUT, "float32", (1,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
)
