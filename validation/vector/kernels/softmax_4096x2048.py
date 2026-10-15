"""softmax 4096x2048: 512 rows per program, C padded to 2048. Blocked as XLA blocks it (validation/vector/emit_kernels.py)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, BM, BC = 4096, 2048, 512, 2048


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def softmax_kernel(x_ptr, o_ptr):
        rows = tl.program_id(0) * 512 + tl.arange(0, 512)
        cols = tl.arange(0, 2048)
        idx = rows[:, None] * 2048 + cols[None, :]
        x = tl.load(x_ptr + idx).to(tl.float32)
        mx = tl.max(x, axis=1)
        e = tl.exp(x - mx[:, None])
        s = tl.sum(e, axis=1)
        tl.store(o_ptr + idx, (e / s[:, None]).to(tl.bfloat16))

    return softmax_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R * C, generator=g, dtype=torch.float32).to(torch.bfloat16)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(R, C)
    return {"o": torch.softmax(x, dim=-1, dtype=torch.float32).to(torch.bfloat16).reshape(-1)}


SPEC = KernelSpec(
    name="softmax_kernel",
    kernel=kernel,
    signature={"x_ptr": "*bf16", "o_ptr": "*bf16"},
    constexprs={},
    args=[Arg("x", IN, "bfloat16", (R * C,)), Arg("o", OUT, "bfloat16", (R * C,))],
    grid=(8,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
)
