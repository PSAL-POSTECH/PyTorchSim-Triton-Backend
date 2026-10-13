"""gemm 1x2048x2048 (MxKxN) bf16, mode=xla window=512."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 1, 2048, 2048


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm_kernel(a_ptr, b_ptr, c_ptr):
        r512 = tl.arange(0, 512)
        r2048 = tl.arange(0, 2048)
        acc_0 = tl.zeros((2048,), dtype=tl.float32)
        for k in range(0, 2048, 512):
            x = tl.load(a_ptr + k + r512).to(tl.float32)
            b = tl.load(b_ptr + (k + r512)[:, None] * 2048 + (0 + r2048)[None, :]).to(tl.float32)
            acc_0 += tl.sum(x[:, None] * b, axis=0)
        tl.store(c_ptr + 0 + r2048, acc_0.to(tl.bfloat16))

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
