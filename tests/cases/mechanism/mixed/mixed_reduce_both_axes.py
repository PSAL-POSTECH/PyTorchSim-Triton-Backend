"""One buffer reduced on BOTH axes -- the case that goes around through DRAM.

    c = tl.dot(a, b)
    tl.store(..., tl.sum(c, axis=1))
    tl.store(..., tl.sum(c, axis=0))

The array MUST have c's last axis on the lanes. One reduction folds that axis and
must NOT have it there; the other folds the first axis and is happy. Every axis
of that buffer is spoken for, and one buffer has one layout.

WHY FLIPPING DOES NOT HELP
(A@B) -> (Bt@At) transposes c, which swaps which reduction is the happy one --
with one reduction that resolves it (mixed_matmul_reduce_axis1), with two it
moves the conflict rather than removing it. This is the first case where
absorption is not enough, which is the whole reason the kernel exists.

WHAT RESOLVES IT
The two reductions stop sharing a buffer. fit_layout_to_lanes emits

    %c2 = torchsim.relayout(%c, %scratch)

and points the losing reduction at the copy; convert_accesses_to_transfer turns that
into one MVOUT and one MVIN over the same DRAM image with different
vlane_split_axis. Nothing transposes -- the lane assignment is a property of the
TRANSFER, not of the bytes, so the two ends disagree about it on purpose:

    MVOUT  tile_stride [1,128]  vlane 1  ->  DRAM row major
    MVIN   DRAM row major                ->  tile_stride [128,1]  vlane 0

That is why DRAM is the answer and not a shuffle: the only thing this hardware
cannot do is move a value between lanes, and DRAM is the one memory every lane
can reach.

WHERE THE DRAM COMES FROM
A function ARGUMENT that fit_layout_to_lanes appends, sized on the func as
torchsim.dram_scratch and allocated by the generated wrapper. Arguments already have
a descriptor path -- flat_view takes (memref arg, index), lower_to_dma
reads the address off the descriptor -- so the relayout carries a real memref
operand and invents no addressing. It is appended LAST so no existing index
moves.

WHAT IT COST TO GET HERE, WHICH WAS NOT THE TRANSFER
Two reductions also share the tensor.empty + linalg.fill that seeds them, so
splitting their layouts is not enough -- the shared init unions the components
straight back together. fit_layout_to_lanes clones the init, one-shot-bufferize
then gives each its own buffer and emits a memref.copy between them, and that
copy is WRONG here: both buffers are lane-banked, and memref.copy walks them
contiguously, so it reads lane 0's bank as the whole tile. bufferize._refill_copies
turns it back into a fill. Nothing lowers memref.copy, so a surviving one is now
refused by name rather than reaching LLVM as a silent loop.

THE ALTERNATIVE, WHICH IS NOT SIMPLY WORSE
Duplicating the matmul -- computing c twice, once flipped, so each reduction gets
its own layout -- needs no DRAM at all. Which is cheaper DEPENDS ON THE SHAPE.
The flipped matmul wants Bt and At, different DESCRIPTORS, so A and B are staged
again rather than reused. For A[M,K], B[K,N], C[M,N]:

    round trip   2*M*N               traffic, no array time
    recompute    2*(M*K + K*N)       traffic, plus a whole matmul

Recompute wins when K is small enough that the operands are cheaper to move than
the result -- a tall-skinny times short-fat product, where C dwarfs both inputs.
At this kernel's shape (M = N = 128, K = 256) it loses: 512 KB against 128 KB,
plus 256 array cycles. Choosing between them is a cost model; until there is one
the round trip is the fixed rule, since it never occupies the array and K >= N is
the common case.

WHAT PROVES IT FIRED
Both outputs are right, and 04-transformed.mlir holds an MVOUT and an MVIN over the
same reinterpret_cast with vlane_split_axis 1 and 0. Before this, the kernel
refused with

    Fatal: axis 1 must be the lane axis (linalg.matmul) and cannot be
           (linalg.reduce folds dim 1)

CLOSED: THE SECOND ROUND TRIP IS GONE (rule 8b)
This kernel used to carry TWO round trips. Only the first was the one described
above; the second was stamped

    torchsim.relayout = array<i64: 1, 1>     out to DRAM and back to the SAME
                                         banking. Nothing is bought.

and mixed_reduce_four_ways.py carried the identical pair, which made 2 of the
suite's 15 trips pointless. Measured today this kernel carries ONE trip, the
justified `[1, 0]` above, and the suite has 0 pointless trips anywhere. Its line
in tests/baselines/tpuv6e.txt records the count, so a second trip reappearing
here is a regression rather than a new finding.

Both axes of one buffer are claimed and the flip only moves the conflict, so the losing reduction reads a copy staged out to DRAM and back with the other lane assignment.
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
    def mixed_reduce_both_axes_kernel(a_ptr, b_ptr, c_ptr, d_ptr,
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
        tl.store(d_ptr + offs_n, tl.sum(acc, axis=0))

    return mixed_reduce_both_axes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    import torch
    p = torch.matmul(inputs["a"], inputs["b"])
    return {"c": p.sum(dim=1), "d": p.sum(dim=0)}


SPEC = KernelSpec(
    name="mixed_reduce_both_axes_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "d_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[
        Arg("a", IN, "float32", (M, K)),
        Arg("b", IN, "float32", (K, N)),
        Arg("c", OUT, "float32", (M,)),
        Arg("d", OUT, "float32", (N,)),
    ],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    # Accumulation order differs from torch's, so the tolerance is looser than
    # the elementwise kernels'.
    rtol=1e-3, atol=1e-3,
)
