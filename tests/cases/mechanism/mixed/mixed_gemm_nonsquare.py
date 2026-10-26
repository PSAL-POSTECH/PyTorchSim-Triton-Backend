"""GEMM with every tile dimension DIFFERENT: M=64, N=128, K tiles of 32.

A verification tool, not a feature. Every other matmul kernel here is square
(128 x 128 x 128), and lower_to_vcix reads M, K and N out of the operand shapes
(mtA.shape[0], mtA.shape[1], mtB.shape[1]) -- so on a square tile it cannot tell
those three apart, and neither can a test. Anything that mixes them up, or that
claims to transpose the output and does not, produces exactly the same numbers.

That matters for the transpose-absorption work: lower_to_vcix never reads
linalg.matmul's indexing_maps (measured: zero occurrences), so rewriting the
output map to produce C transposed would change what the IR SAYS and not what
the array DOES. On square tiles the result would be identical and the bug
invisible. This kernel is what makes it visible.

WHAT IT ALSO COVERS
  BM=64 < the 128 lanes, so the C tile spreads over 64 lanes and the rest sit
  idle -- spad.per_lane divides by the lanes actually USED rather than by the
  machine's count, which is the case that used to under-reserve.
  K/BK = 2 trips, so the accumulator is loop-carried across the K loop -- the
  shape fuse_matmul_accumulate has to recover and lower_to_vcix accumulates in
  place on the array.

WHY ONLY M DIFFERS HERE
Building this kernel is what turned up err 41.54 for N = 64 and 43.34 for K = 32
against 3.43e-05 for M = 64, so it kept M as the odd one out and left the other
two at the array's dimension.

That was first read as "the array cannot take a partial tile" and refused by
name. It was not the array: the same runs under SPIKE_DEBUG show the weights,
inputs and outputs all correct. The fault was in the SCRATCHPAD RESERVATION,
which sized a buffer by dim 0 while a matmul banks on its last axis -- and this
kernel is on the safe side of it (BM=64 < BN=128 over-reserves) which is why it
passed while the two below did not. See mixed_gemm_partial_n and
mixed_gemm_partial_k, which put the short axis on the other side.

Non-square tiles, so a pass that confuses M, K and N is visible. Every other matmul kernel here is 128 cubed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# Tile dims must divide by PYTORCHSIM_COMPILE_VECTORLANE_SIZE (128): the DMA splits dim 0
# across lanes and pytorchsim-vectorize divides the iteration space by it.

# K = 2 * BK on purpose -- one K tile would let triton unroll the reduction
# loop away, and lower_to_vcix needs that loop to mark as accumulation_loop.
M, N, K = 64, 128, 256
BM, BN, BK = 64, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm_nonsquare_kernel(a_ptr, b_ptr, c_ptr,
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

    return gemm_nonsquare_kernel


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
    name="gemm_nonsquare_kernel",
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
