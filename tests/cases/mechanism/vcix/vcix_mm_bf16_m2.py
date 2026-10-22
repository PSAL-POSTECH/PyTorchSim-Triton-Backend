"""bf16 M=2 N=64 K=64 in BK=32 dots: an input of one word, the two rows packed in it. Breaks if the
MXU lowering refuses a last register of one word (it did, as narrower than a MIN_VEC of 2 words a
bank moves) or pops past M into C."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 2, 64, 64
BK = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def vcix_mm_bf16_m2_kernel(a_ptr, b_ptr, c_ptr,
                      M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
        offs_m = tl.arange(0, M)
        offs_n = tl.arange(0, N)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((M, N), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return vcix_mm_bf16_m2_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32).to(torch.bfloat16),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32).to(torch.bfloat16),
    }


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"].to(torch.float32), inputs["b"].to(torch.float32))}


SPEC = KernelSpec(
    name="vcix_mm_bf16_m2_kernel",
    kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BK": BK},
    args=[
        Arg("a", IN, "bfloat16", (M, K)),
        Arg("b", IN, "bfloat16", (K, N)),
        Arg("c", OUT, "float32", (M, N)),
    ],
    grid=(1, 1),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-2, atol=1e-2,
)
