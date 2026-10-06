"""GEMM with an elementwise EPILOGUE: C = relu(A@B).

The matmul result reaches its MVOUT through a linalg.generic rather than
straight from the matmul, so the walk that decides the store tile's lane axis
has to step through the epilogue to find the matmul behind it. Without that the
C tile is banked dim-0 while the systolic array wrote it last-dim: err 56.07.

This is the smaller half of what mixed_gemm_bias covers -- one loaded family,
demand travelling backward through a generic -- kept separate because it fails
for a reason of its own: here the matmul is BEHIND the generic on the store
path, there it is BESIDE it on a sibling operand.

It is also the first kernel that is neither matmul nor elementwise but both,
which is the condition the old per-function `any(linalg.matmul in func)` guess
was silently assuming away.

WHAT PROVES IT FIRED
  The C transfer carries `vlane_split_axis = 1` in 04-transformed.mlir even though
  its immediate producer is a linalg.generic.

matmul feeding an elementwise epilogue before the store. Demand must travel through the generic to reach the MVOUT.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# Tile dims must divide by TORCHSIM_COMPILE_VECTORLANE_SIZE (128): the DMA splits dim 0
# across lanes and pytorchsim_triton_compiler/elf/passes/p05_vectorize.py divides the iteration space by it.
# K = 2 * BK on purpose -- one K tile would let triton unroll the reduction
# loop away, and lower_to_vcix needs that loop to mark as accumulation_loop.
M, N, K = 128, 128, 256
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mixed_gemm_epilogue_kernel(a_ptr, b_ptr, c_ptr,
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

        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], tl.maximum(acc, 0.0))

    return mixed_gemm_epilogue_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    import torch
    return {"c": torch.relu(torch.matmul(inputs["a"], inputs["b"]))}


SPEC = KernelSpec(
    name="mixed_gemm_epilogue_kernel",
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
