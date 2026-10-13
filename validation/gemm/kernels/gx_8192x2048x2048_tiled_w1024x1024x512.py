"""gemm 8192x2048x2048 (MxKxN) bf16, mode=tiled window=1024."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 8192, 2048, 2048


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm_kernel(a_ptr, b_ptr, c_ptr):
        om = tl.arange(0, 1024)
        on = tl.arange(0, 1024)
        ok = tl.arange(0, 512)
        for m in range(0, 8192, 1024):
            for n in range(0, 2048, 1024):
                acc = tl.zeros((1024, 1024), dtype=tl.float32)
                for k in range(0, 2048, 512):
                    a = tl.load(a_ptr + (m + om)[:, None] * 2048 + (k + ok)[None, :])
                    b = tl.load(b_ptr + (k + ok)[:, None] * 2048 + (n + on)[None, :])
                    acc += tl.dot(a, b, out_dtype=tl.float32)
                tl.store(c_ptr + (m + om)[:, None] * 2048 + (n + on)[None, :], acc.to(tl.bfloat16))

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
