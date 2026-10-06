"""A matmul whose N does not fill the systolic array.

The array is 128 wide (spike systolic_array.h: sa_dim = n_vu) and N here is 64,
so half its columns get no weights pushed into them. That is fine, and this
kernel is the one that established why.

WHY A PARTIAL FILL COSTS NOTHING
A weight push writes one value per LANE, and the lane index IS the array column
(verified under SPIKE_DEBUG: column j held exactly B[:, j]). With N = 64 only
lanes 0..63 carry weights; lanes 64..127 push whatever is in their own banks at
that address, and nothing ever writes there -- the scratchpad globals are C
arrays in `.spad`, zero at load, and the DMA for a 64-wide tile only targets the
64 lanes the tile spans. So columns 64..127 hold zero weights and produce zero
outputs. Measured, same run:

    array columns  0..63    == B[:, j] exactly
    array columns 64..127   all zero
    inputs                  == A[m, 0:128]
    outputs                 == the reference C rows

THIS USED TO BE A NEGATIVE KERNEL, FOR THE WRONG REASON
It produced err 41.54 and was read as "a partially filled array is not what the
push/compute/pop sequence builds", and refused by name. The array was never the
problem. The number came from the SCRATCHPAD RESERVATION: pytorchsim_triton_compiler/elf/passes/lib_bank_layout.py sized a
buffer by its dim 0, but linalg.matmul banks on its LAST axis, so the [128, 64]
C tile was given 128 lanes x 64 elements when it needs 64 lanes x 128. Every
lane's second half landed in the next global.

That is exactly the shape of the evidence, and it is worth keeping because the
first reading also fit the error and was still wrong:

    rows   0..63   of C correct
    rows  64..127  of C garbage        <- the half that spilled

Reserving by the narrowest dim instead took it to 3.81e-05 with the lowering
untouched. See mixed_gemm_partial_k for the same fault on the other operand.

WHAT IS STILL REFUSED
A dim LARGER than the array and not a multiple of it. That one is real: the
N//SS loop in lower_to_vcix walks it that many times and drops the tail.

N below the systolic width. The array handles it; what did not was the scratchpad reservation, which sized the C tile by dim 0 while a matmul banks on its last axis (err 41.54).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# N = 64 is the point: HALF the array width. K = 2 * BK on purpose -- one K tile
# would let triton unroll the reduction loop away, and lower_to_vcix needs that
# loop to find the accumulation it hangs the sequence off.
M, N, K = 128, 64, 256
BM, BN, BK = 128, 64, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mixed_gemm_partial_n_kernel(a_ptr, b_ptr, c_ptr,
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

    return mixed_gemm_partial_n_kernel


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
    name="mixed_gemm_partial_n_kernel",
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
    # Accumulation order differs from torch's, so the tolerance is the matmul
    # kernels', not the elementwise ones'. Measured max_abs_err 3.81e-05.
    rtol=1e-3, atol=1e-3,
)
