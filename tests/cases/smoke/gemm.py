"""GEMM -- the kernel that actually exercises the F6 gap.

torch equivalent: a @ b

Unlike the elementwise kernels, this one has a reduction loop over K, which
the linalg lowering emits as `scf.for` carrying a tensor iter_arg, plus a
`linalg.matmul` inside it. Both are things no vendored pass consumes today:

  * lower_to_vcix reads the reduction trip count out of an affine `upperBoundMap`
    (lower_to_vcix.py:160) -- an scf.for is invisible to it
  * lower_to_vcix._lower_matmul expects the matmul in the form the PyTorchSim
    templates produce, not linalg.matmul on tensors

So this is the kernel to develop the stage-3 backend (pytorchsim_triton_compiler/lowering/passes/) against.
Run it with
`--to transform -v` to see exactly what the gap looks like.

Primary development target for the stage-3 backend in pytorchsim_triton_compiler/lowering/passes/.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# Tile dims must divide by TORCHSIM_COMPILE_VECTORLANE_SIZE (128): the DMA splits dim 0
# across lanes and pytorchsim_triton_compiler/lowering/passes/p05_vectorize.py divides the iteration space by it.
# K = 2 * BK on purpose -- one K tile would let triton unroll the reduction
# loop away, and lower_to_vcix needs that loop to mark as accumulation_loop.
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
