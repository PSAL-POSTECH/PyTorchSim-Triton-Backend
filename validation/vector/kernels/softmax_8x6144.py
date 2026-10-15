"""softmax 8x6144: one program, C as 128 chunks of 64 on the lanes, 8 rows. Blocked as XLA blocks it (validation/vector/emit_kernels.py)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, BM, BC = 8, 6144, 8, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def softmax_kernel(x_ptr, o_ptr):
        q = tl.arange(0, 128)
        r = tl.arange(0, 8)
        k = tl.arange(0, 64)
        idx = r[None, :, None] * 6144 + q[:, None, None] * 64 + k[None, None, :]
        x = tl.load(x_ptr + idx, mask=q[:, None, None] < 96, other=-1.0e30).to(tl.float32)
        mx = tl.max(tl.max(x, axis=2), axis=0)
        e = tl.exp(x - mx[None, :, None])
        s = tl.sum(tl.sum(e, axis=2), axis=0)
        tl.store(o_ptr + idx, (e / s[None, :, None]).to(tl.bfloat16), mask=q[:, None, None] < 96)

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
