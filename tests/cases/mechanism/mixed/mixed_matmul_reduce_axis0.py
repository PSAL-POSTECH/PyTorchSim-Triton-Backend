"""`tl.sum(tl.dot(a, b), axis=0)` -- a reduction on a buffer banked off dim 0.

The systolic array writes C a COLUMN PER LANE, because that is what its
serialisers produce. Folding axis 0 then stays inside a lane -- lane j holds
column j, and summing that column never crosses a lane boundary -- so this needs
no relayout at all. Folding axis 1 would cross lanes and is refused
(mixed_matmul_reduce_conflict). The conflict is not symmetric, and the axis model
has to be able to say which one it is:

    matmul MUST axis 1, reduce NOT axis 0  ->  axis 1, this kernel
    matmul MUST axis 1, reduce NOT axis 1  ->  conflict, the other one

WHY THIS ONE WAS HARD
A spad buffer is typed from its tensor, so its memref layout is row-major
whatever the DMA did with it. For a lane axis of 0 that IS the real layout, so
indexing the plain memref has always worked. Here it is not:

    lane axis 1     tile_stride [1, 128]     axis 0 packed inside the bank
    memref type                [128, 1]      axis 1 packed

and worse, the LANE axis sits in the type's unit-stride slot even though it
selects a bank rather than an offset inside one. Everything that really is an
in-bank offset gets pushed into a bigger slot than it should have.

Sorting the axes by their REAL stride puts them back, and vector.load needs that
anyway -- it requires unit stride on the most minor dimension.

THE PART THAT TOOK FOUR TRIES
The strides of the buffer AFTER the split have to be derived through the
expand_shape, not recomputed for the new shape. An expand_shape does not create a
tile, it renames the axes of one:

    split axis 0 (stride 1) into (4, 32)  ->  strides (32, 1)
    expanded [4, 32, 128]                 ->  [32, 1, 128]

Recomputing gives tile_strides([4,32,128], 2) = [1, 4, 128], which puts the CHUNK
on the unit stride and the vector 4 apart. Both split orders were measured that
way -- 465.69 with the chunk outer, 613.84 with it inner -- while the same kernel
small enough not to split was already exact at 9.16e-05. That bisect is what
showed the array's layout and the indexing were both fine and the split was not.

For a lane axis of 0 the derived and recomputed answers agree, which is why
nothing noticed until a buffer arrived banked on axis 1.

WHAT PROVES IT FIRED
  08-transformed.mlir has a memref.reinterpret_cast to strided<[128, 32, 1]> before
  the reduction's loads. A dim-0-banked kernel has none -- the real strides are
  already the type's, so the view is the buffer itself.

Reduction on a buffer the systolic array banked on its last axis. The half of matmul+reduce that needs no relayout, and the only kernel where the spad strides differ from the memref type.
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
    def mixed_matmul_reduce_axis0_kernel(a_ptr, b_ptr, c_ptr,
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

        tl.store(c_ptr + offs_n, tl.sum(acc, axis=0))

    return mixed_matmul_reduce_axis0_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"], inputs["b"]).sum(dim=0)}


SPEC = KernelSpec(
    name="mixed_matmul_reduce_axis0_kernel",
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
