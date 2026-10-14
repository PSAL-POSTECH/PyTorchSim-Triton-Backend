"""Batched matmul -- GEMM plus a batch dimension carried on the grid's z axis.

torch equivalent: torch.bmm(a, b)

What this adds over gemm.py: the batch index comes from `tl.program_id(2)` and
is folded into the base pointer, so the pointer arithmetic is 3-D. That is the
case where a pointer-structure analysis has to recover a strided access across a
dimension the loop nest never iterates -- the thing §6.3 of plan-triton.md warns
must fail loudly rather than silently degrade to a gather.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# See gemm.py: tiles must divide by the lane count, and K must span more than
# one tile so the reduction loop survives for the matrix unit.
B, M, N, K = 2, 256, 256, 256
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def bmm_kernel(a_ptr, b_ptr, c_ptr,
                   B: tl.constexpr, M: tl.constexpr,
                   N: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)

        a_base = a_ptr + pid_b * M * K
        b_base = b_ptr + pid_b * K * N
        c_base = c_ptr + pid_b * M * N

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_base + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_base + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)

        tl.store(c_base + offs_m[:, None] * N + offs_n[None, :], acc)

    return bmm_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(B, M, K, generator=g, dtype=torch.float32),
        "b": torch.randn(B, K, N, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    import torch
    return {"c": torch.bmm(inputs["a"], inputs["b"])}


SPEC = KernelSpec(
    name="bmm_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "B": "constexpr", "M": "constexpr",
               "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"B": B, "M": M, "N": N, "K": K,
                "BM": BM, "BN": BN, "BK": BK},
    args=[
        Arg("a", IN, "float32", (B, M, K)),
        Arg("b", IN, "float32", (B, K, N)),
        Arg("c", OUT, "float32", (B, M, N)),
    ],
    grid=(M // BM, N // BN, B),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
