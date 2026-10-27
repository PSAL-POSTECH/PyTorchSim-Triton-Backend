"""GEMM, a @ b: a K loop the linalg lowering emits as an scf.for carrying the accumulator around a
linalg.matmul, which the target's pytorchsim-lower-matmul (targets/tpu/backend/lower_matmul) lowers to
the matrix unit. K = 2 * BK so triton cannot unroll that loop away; tiles are 128, the lanes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 512, 512, 256
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm_kernel(a_ptr, b_ptr, c_ptr,
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

    return gemm_kernel


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
    name="gemm_kernel",
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
    # Accumulation order differs from torch's, so the tolerance is looser than
    # the elementwise kernels'.
    rtol=1e-3, atol=1e-3,
)
