"""softmax 512x1536: 512 rows per program, C = 1024 + 512. Blocked as XLA blocks it (validation/vector/emit_kernels.py)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, BM, BC = 512, 1536, 512, 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def softmax_kernel(x_ptr, o_ptr):
        rows = tl.program_id(0) * 512 + tl.arange(0, 512)
        ia = rows[:, None] * 1536 + tl.arange(0, 1024)[None, :]
        ib = rows[:, None] * 1536 + 1024 + tl.arange(0, 512)[None, :]
        xa = tl.load(x_ptr + ia).to(tl.float32)
        xb = tl.load(x_ptr + ib).to(tl.float32)
        mx = tl.maximum(tl.max(xa, axis=1), tl.max(xb, axis=1))
        ea = tl.exp(xa - mx[:, None])
        eb = tl.exp(xb - mx[:, None])
        s = tl.sum(ea, axis=1) + tl.sum(eb, axis=1)
        tl.store(o_ptr + ia, (ea / s[:, None]).to(tl.bfloat16))
        tl.store(o_ptr + ib, (eb / s[:, None]).to(tl.bfloat16))

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
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
)
