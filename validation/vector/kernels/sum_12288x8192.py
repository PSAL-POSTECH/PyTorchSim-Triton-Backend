"""sum of all of x 12288x8192 bf16 as f32: one program walking 128-row blocks. Blocked as XLA blocks it (validation/vector/emit_kernels.py)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, BM, BC = 12288, 8192, 128, 8192


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def sum_kernel(x_ptr, o_ptr):
        rows = tl.arange(0, 128)
        cols = tl.arange(0, 8192)
        acc = tl.full([128, 8192], 0.0, tl.float32)
        for r in range(0, 12288, 128):
            x = tl.load(x_ptr + (r + rows)[:, None] * 8192 + cols[None, :]).to(tl.float32)
            acc = acc + x
        tl.store(o_ptr, tl.sum(tl.sum(acc, axis=1), axis=0))

    return sum_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R * C, generator=g, dtype=torch.float32).to(torch.bfloat16)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(R, C)
    return {"o": x.float().sum().reshape(1)}


SPEC = KernelSpec(
    name="sum_kernel",
    kernel=kernel,
    signature={"x_ptr": "*bf16", "o_ptr": "*fp32"},
    constexprs={},
    args=[Arg("x", IN, "bfloat16", (R * C,)), Arg("o", OUT, "float32", (1,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
)
