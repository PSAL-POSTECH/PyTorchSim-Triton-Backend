"""mixed_gemm_partial_n's fault on the OTHER operand: a short K.

N is the array's width; K is the DEPTH of each column's weight queue. Both are
sa_dim = 128 in hardware, and both can be under-filled by a tile. This kernel
under-fills K (32 of 128) where its sibling under-fills N (64 of 128), so the
short axis lands on the A tile instead of on B and C.

WHY BOTH ARE NEEDED
The bug they share was in the SCRATCHPAD RESERVATION, which sized a buffer by
its dim 0 while a matmul banks on its LAST axis -- so it under-reserved exactly
when dim 0 is the WIDER dim. The two kernels put that condition on different
buffers:

    partial_n   C tile [128,  64]   reserved 128 lanes x  64, needs  64 x 128
    partial_k   A tile [128,  32]   reserved 128 lanes x  32, needs  32 x 128

A fix that repaired one and not the other would still pass a one-kernel
regression, which is the same trap mixed_gemm_nonsquare was built to close --
every matmul kernel here used to be 128 cubed, where M, K and N cannot be told
apart.

    refused by name  ->  err 43.34  ->  9.54e-06

WHY A SHORT K IS SAFE IN THE ARRAY
compute() walks whatever is in each column's weight deque and pairs entry k with
input k. The push loop already fills the K shortfall with an explicit zero
vector (`if i < K: ... else: zero_vector` in lower_to_vcix), so the deque is a
full 128 deep with zeros past K and the inputs past K multiply nothing. That
zero-fill was there all along; it was never what was broken.

K below the array depth, so the short axis is on the A tile rather than on B and C. The reservation bug its sibling found needs both to be pinned down.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# BK = 32 is the point: a QUARTER of the array depth. K = 2 * BK so the
# reduction loop survives (see the sibling).
M, N, K = 128, 128, 64
BM, BN, BK = 128, 128, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mixed_gemm_partial_k_kernel(a_ptr, b_ptr, c_ptr,
                    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)

        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return mixed_gemm_partial_k_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"], inputs["b"])}


SPEC = KernelSpec(
    name="mixed_gemm_partial_k_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[
        Arg("a", IN, "float32", (M, K)),
        Arg("b", IN, "float32", (K, N)),
        Arg("c", OUT, "float32", (M, N)),
    ],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    # Measured max_abs_err 9.54e-06.
    rtol=1e-3, atol=1e-3,
)
