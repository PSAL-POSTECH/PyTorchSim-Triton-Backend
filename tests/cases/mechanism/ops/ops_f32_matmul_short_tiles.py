"""f32 M=4 N=32 K=32 in BK=16 dots: an input of four rows of the eight a register holds, so each of
its pops is half a register of C. Breaks if the MXU lowering refuses the short K or M, or writes a
whole register of pops over C's four rows."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 4, 32, 32
BK = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_f32_matmul_short_tiles_kernel(a_ptr, b_ptr, c_ptr,
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

    return ops_f32_matmul_short_tiles_kernel


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
    name="ops_f32_matmul_short_tiles_kernel",
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
)
