"""bf16 M=8 N=64 K=64 in BK=32 dots: on a matrix unit target every dimension is short of its tile --
half a register of rows, half the lanes of columns, a quarter of the depth. Breaks if the MXU
lowering refuses a tile that is not whole (it did: 'matmul K=32 is not whole lane tiles of 128')
or pops the eight rows past M into C."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 8, 64, 64
BK = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_bf16_matmul_short_tiles_kernel(a_ptr, b_ptr, c_ptr,
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

    return ops_bf16_matmul_short_tiles_kernel


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
    name="ops_bf16_matmul_short_tiles_kernel",
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
