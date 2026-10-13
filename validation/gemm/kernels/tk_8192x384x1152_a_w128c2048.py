"""gemm 8192x384x1152 (MxKxN) bf16, mode=a Kwhole Npieces=Wn128 window=128."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 8192, 1152, 384


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm_kernel(a_ptr, b_ptr, c_ptr):
        ok = tl.arange(0, 512)
        om = tl.arange(0, 2048)
        a_0 = tl.load(a_ptr + (0 + om)[:, None] * 384 + ok[None, :], mask=ok[None, :] < 384, other=0.0)
        a_1 = tl.load(a_ptr + (2048 + om)[:, None] * 384 + ok[None, :], mask=ok[None, :] < 384, other=0.0)
        a_2 = tl.load(a_ptr + (4096 + om)[:, None] * 384 + ok[None, :], mask=ok[None, :] < 384, other=0.0)
        a_3 = tl.load(a_ptr + (6144 + om)[:, None] * 384 + ok[None, :], mask=ok[None, :] < 384, other=0.0)
        for n in range(0, 1152, 128):
            on = n + tl.arange(0, 128)
            b = tl.load(b_ptr + ok[:, None] * 1152 + on[None, :], mask=ok[:, None] < 384, other=0.0)
            acc = tl.dot(a_0, b, out_dtype=tl.float32)
            tl.store(c_ptr + (0 + om)[:, None] * 1152 + on[None, :], acc.to(tl.bfloat16))
            acc = tl.dot(a_1, b, out_dtype=tl.float32)
            tl.store(c_ptr + (2048 + om)[:, None] * 1152 + on[None, :], acc.to(tl.bfloat16))
            acc = tl.dot(a_2, b, out_dtype=tl.float32)
            tl.store(c_ptr + (4096 + om)[:, None] * 1152 + on[None, :], acc.to(tl.bfloat16))
            acc = tl.dot(a_3, b, out_dtype=tl.float32)
            tl.store(c_ptr + (6144 + om)[:, None] * 1152 + on[None, :], acc.to(tl.bfloat16))

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
