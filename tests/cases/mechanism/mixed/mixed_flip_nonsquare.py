"""A matmul+reduce conflict on NON-SQUARE tiles, resolved by flipping.

`tl.sum(tl.dot(a, b), axis=1)` asks for one buffer two ways. The matrix unit
MUST have C's last axis on the lanes; the reduction folds that same axis, and a
fold across lanes has no instruction behind it. fit_layout_to_lanes resolves it by
turning A@B into Bt@At, which is C transposed -- so the reduction lands on axis
0, where each lane folds its own elements and nothing crosses.

WHAT IS DIFFERENT HERE
C is [128, 256] and Ct is [256, 128], so the flip changes a TYPE. On a square
tile it changes none, and the rewrite is two attribute swaps per operand; here
the accumulator is loop-carried, so the same buffer appears under six names:

    tensor.empty -> linalg.fill -> scf.for iter_arg -> linalg.matmul outs
                 -> scf.yield  -> scf.for result    -> linalg.reduce

They transpose together or the IR does not verify, which is why the flip walks
that chain (fit_layout_to_lanes._acc_chain) instead of retyping the matmul alone.
The walk is also the safety boundary: it declines the flip if ANYTHING but those
structural uses reads the accumulator, because such a reader would see
transposed data and say nothing about it.

The tile POINTERS carry a type too. Reversing a tts.make_tptr's sizes, strides
and offsets while leaving its result type at [128,256] would have the descriptor
stage [256,128] with the type claiming otherwise -- invisible on a square tile,
which is why it was missing until this kernel existed.

WHY THE FLIP IS STILL FREE
Nothing moves. A tts.make_tptr is sizes + offsets + strides over DRAM, so the
transposed operand is those reversed and the DMA reads the same bytes in the
other order -- the descriptors already do exactly this on every gemm load. The
offsets are SUMMED rather than multiplied by their strides (PtrAnalysis has
already folded each stride into its offset), so reversing them is a no-op on the
base address and the correctness rides entirely on sizes and strides.

WHAT PROVES IT FIRED
The kernel produces the right numbers at all, and 08-transformed.mlir shows

    linalg.matmul ins(%B, %A : tensor<256x128xf32>, tensor<128x128xf32>)
    linalg.reduce ... dimensions = [0]

-- operands swapped, accumulator [256,128], reduction on dim 0. Before the flip
it refused with "axis 1 must be the lane axis (linalg.matmul) and cannot be
(linalg.reduce folds dim 1)".

WHAT WOULD STILL DECLINE IT
M not a multiple of the matrix unit's lane tile. The flip makes M the new N, and
lower_to_vcix requires N and K to be exact multiples while leaving M free, so a
flip there would trade a clear refusal for one naming a dimension the kernel does
not have. See mixed_flip_nonsquare_wide for the other aspect ratio.

matmul+reduce conflict on non-square tiles: the flip retypes the loop-carried accumulator [128,256] -> [256,128] and the tile pointers with it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# Tile dims must divide by TORCHSIM_COMPILE_VECTORLANE_SIZE (128): the DMA splits dim 0
# across lanes and pytorchsim_triton_compiler/lowering/passes/p05_vectorize.py divides the iteration space by it.
# K = 2 * BK on purpose -- one K tile would let triton unroll the reduction
# loop away, and the accumulator would stop being loop-carried, which is the
# whole thing this kernel is about.
M, N, K = 128, 256, 256
BM, BN, BK = 128, 256, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mixed_flip_nonsquare_kernel(a_ptr, b_ptr, c_ptr,
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

        tl.store(c_ptr + offs_m, tl.sum(acc, axis=1))

    return mixed_flip_nonsquare_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"], inputs["b"]).sum(dim=1)}


SPEC = KernelSpec(
    name="mixed_flip_nonsquare_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[
        Arg("a", IN, "float32", (M, K)),
        Arg("b", IN, "float32", (K, N)),
        Arg("c", OUT, "float32", (M,)),
    ],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    # The flip reassociates the sum -- it folds Ct down dim 0 instead of C along
    # dim 1 -- so the tolerance is the matmul kernels', not the elementwise ones'.
    # Measured max_abs_err 3.05e-04 on values of magnitude ~256 (rel 2.7e-05).
    rtol=1e-3, atol=1e-3,
)
