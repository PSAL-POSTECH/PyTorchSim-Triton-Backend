"""max of all of x 3072x2048 bf16 as f32: one program walking 512-row blocks. Blocked as XLA blocks it (validation/vector/emit_kernels.py)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, BM, BC = 3072, 2048, 512, 2048


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def max_kernel(x_ptr, o_ptr):
        rows = tl.arange(0, 512)
        cols = tl.arange(0, 2048)
        acc = tl.full([512, 2048], -1.0e30, tl.float32)
        for r in range(0, 3072, 512):
            x = tl.load(x_ptr + (r + rows)[:, None] * 2048 + cols[None, :]).to(tl.float32)
            acc = tl.maximum(acc, x)
        tl.store(o_ptr, tl.max(tl.max(acc, axis=1), axis=0))

    return max_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R * C, generator=g, dtype=torch.float32).to(torch.bfloat16)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(R, C)
    return {"o": torch.max(x).reshape(1).float()}


SPEC = KernelSpec(
    name="max_kernel",
    kernel=kernel,
    signature={"x_ptr": "*bf16", "o_ptr": "*fp32"},
    constexprs={},
    args=[Arg("x", IN, "bfloat16", (R * C,)), Arg("o", OUT, "float32", (1,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
)
