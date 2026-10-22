"""A dot's C stored at columns (n + 40) % 256: the store's tile is [n, m], so its cut is on axis 0,
the one lower-matmul splits a store along. That store stays whole (2 pieces after the dot): a split
needs the phase rebased by each subtile's start and p11 to box a subtile. Breaks if the guard goes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K, O = 128, 256, 256, 40
BM, BN, BK = 128, 256, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_cols_dot_store_kernel(a_ptr, b_ptr, c_ptr,
                               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                               O: tl.constexpr):
        offs_m = tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)

        tl.store(c_ptr + offs_m[:, None] * N + ((offs_n + O) % N)[None, :], acc.to(tl.bfloat16))

    return ptr_wrap_cols_dot_store_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32).to(torch.bfloat16),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32).to(torch.bfloat16),
    }


def reference(inputs):
    import torch
    c = torch.matmul(inputs["a"].to(torch.float32), inputs["b"].to(torch.float32))
    out = torch.empty_like(c)
    out[:, (torch.arange(N) + O) % N] = c
    return {"c": out.to(torch.bfloat16)}


SPEC = KernelSpec(
    name="ptr_wrap_cols_dot_store_kernel",
    kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr", "O": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK, "O": O},
    args=[
        Arg("a", IN, "bfloat16", (M, K)),
        Arg("b", IN, "bfloat16", (K, N)),
        Arg("c", OUT, "bfloat16", (M, N)),
    ],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-2, atol=1e-2,
    expect_ir={"09-custom.mlir": {"torchsim.transfer": 6, "subtile_size": 0}},
)
