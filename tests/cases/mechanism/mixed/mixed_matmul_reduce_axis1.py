"""`tl.sum(tl.dot(a, b), axis=1)` -- the conflicting half, absorbed for free.

The array writes C a COLUMN PER LANE, so folding axis 1 sums across lanes and
there is no instruction for that. Folding axis 0 stays inside one lane
(mixed_matmul_reduce_axis0). The two demands on that one buffer are exactly
opposed:

    linalg.matmul   MUST have axis 1 on the lanes
    linalg.reduce   folds axis 1, so it must NOT be there

One buffer has one layout, so this used to be refused by name.

HOW IT IS ABSORBED
(A@B) becomes (Bt@At), which IS C transposed, so the same sum is now over axis 0
-- the half that already worked. Nothing moves: a tts.make_tptr is sizes,
offsets and strides over DRAM, and transposing the tile it stages is reversing
those three. gemm's own descriptors already read row-major DRAM into a
column-major tile (dram_stride [256,1], tile_stride [1,128]), so the hardware
does this on every load anyway.

WHY THE OPERANDS AND NOT THE MAPS
linalg.matmul carries indexing_maps, and rewriting the output map to (d1, d0)
would say "produce C transposed" in one line. But lower_to_vcix never reads
those maps -- measured, zero occurrences -- so it would change what the IR says
and not what the array does. On the square tiles every matmul kernel here uses,
the result would even be numerically identical and the bug invisible. Swapping
the operands keeps the standard maps, so the existing lowering follows without
being touched.

THIS ONE IS SQUARE, WHICH HIDES HALF THE REWRITE
On a square tile the flip changes no type at all, so it is two attribute swaps
per operand and nothing else. A non-square flip transposes the result, and the
accumulator is loop-carried, so the same buffer has to be retyped under six
names -- see mixed_flip_nonsquare, which is the kernel that made that visible.
(A tile that does not fill the array is fine either way: see
mixed_gemm_partial_n.)

    refused by name  ->  err 1.22e-04, matmul-accumulation scale

The conflicting half of matmul+reduce, resolved by flipping the matmul instead of staging through DRAM. Costs nothing: the DMA transposes on every load already.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# Tile dims must divide by TORCHSIM_COMPILE_VECTORLANE_SIZE (128): the DMA splits dim 0
# across lanes and pytorchsim_triton_compiler/lowering/passes/p05_vectorize.py divides the iteration space by it.
# K = 2 * BK on purpose -- one K tile would let triton unroll the reduction
# loop away, and lower_to_vcix needs that loop to mark as accumulation_loop.
M, N, K = 128, 128, 256
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mixed_matmul_reduce_axis1_kernel(a_ptr, b_ptr, c_ptr,
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

    return mixed_matmul_reduce_axis1_kernel


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
    name="mixed_matmul_reduce_axis1_kernel",
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
    # Accumulation order differs from torch's, so the tolerance is looser than
    # the elementwise kernels'.
    rtol=1e-3, atol=1e-3,
)
