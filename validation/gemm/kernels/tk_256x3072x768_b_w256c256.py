"""gemm 256x3072x768 (MxKxN) bf16, mode=b Kwhole Npieces=256+256+256 window=256."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 256, 768, 3072


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm_kernel(a_ptr, b_ptr, c_ptr):
        rk = tl.arange(0, 4096)
        r256 = tl.arange(0, 256)
        b_0 = tl.load(b_ptr + rk[:, None] * 768 + (0 + r256)[None, :], mask=rk[:, None] < 3072, other=0.0)
        b_1 = tl.load(b_ptr + rk[:, None] * 768 + (256 + r256)[None, :], mask=rk[:, None] < 3072, other=0.0)
        b_2 = tl.load(b_ptr + rk[:, None] * 768 + (512 + r256)[None, :], mask=rk[:, None] < 3072, other=0.0)
        for m in range(0, 256, 256):
            om = m + r256
            a = tl.load(a_ptr + om[:, None] * 3072 + rk[None, :], mask=rk[None, :] < 3072, other=0.0)
            acc = tl.dot(a, b_0, out_dtype=tl.float32)
            tl.store(c_ptr + om[:, None] * 768 + (0 + r256)[None, :], acc.to(tl.bfloat16))
            acc = tl.dot(a, b_1, out_dtype=tl.float32)
            tl.store(c_ptr + om[:, None] * 768 + (256 + r256)[None, :], acc.to(tl.bfloat16))
            acc = tl.dot(a, b_2, out_dtype=tl.float32)
            tl.store(c_ptr + om[:, None] * 768 + (512 + r256)[None, :], acc.to(tl.bfloat16))

    return gemm_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32).to(torch.bfloat16),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32).to(torch.bfloat16),
    }


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"].to(torch.float32), inputs["b"].to(torch.float32)).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="gemm_kernel",
    kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16"},
    constexprs={},
    args=[
        Arg("a", IN, "bfloat16", (M, K)),
        Arg("b", IN, "bfloat16", (K, N)),
        Arg("c", OUT, "bfloat16", (M, N)),
    ],
    grid=(1, 1),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
)
