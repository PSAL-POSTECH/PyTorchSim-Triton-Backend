"""f32 M=1 N=32 K=32 in BK=16 dots: one row of A times B is a vector op, which the matrix unit
refuses by name. Breaks if lower_matmul takes M=1 onto the array; on the tree before the M=1 rule
it was refused too, for a one-word register tail, so this pins the refusal, not its reason."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 1, 32, 32
BK = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def vcix_mm_f32_m1_kernel(a_ptr, b_ptr, c_ptr,
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

    return vcix_mm_f32_m1_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32).to(torch.float32),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32).to(torch.float32),
    }


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"].to(torch.float32), inputs["b"].to(torch.float32))}


SPEC = KernelSpec(
    name="vcix_mm_f32_m1_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BK": BK},
    args=[
        Arg("a", IN, "float32", (M, K)),
        Arg("b", IN, "float32", (K, N)),
        Arg("c", OUT, "float32", (M, N)),
    ],
    grid=(1, 1),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-2, atol=1e-2,
    expect="refuse",
    expect_why="tritonpytorchsim-lower-matmul-tpuv6e: M=1 is one row of A times B, a vector op and not "
               "one for the matrix unit; the harness does not match the message, read it in stage.log",
)
