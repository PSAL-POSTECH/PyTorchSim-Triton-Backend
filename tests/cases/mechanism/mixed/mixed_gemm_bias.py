"""GEMM with a BIAS: C = A@B + bias. The kernel that needs the axis closure.

WHY THIS ONE AND NOT gemm.py
The lane axis is decided per tile, by asking what the tile feeds. Walking
FORWARD from the bias tile finds a linalg.generic and a store -- no matmul --
so the bias gets the default dim-0 banking while the matmul result it is added
to is banked last-dim. Element (i, j) of one then meets element (j, i) of the
other and nothing in the IR is inconsistent, so nothing complains.

    forward walk only    err 90.46
    with the closure     err 4.58e-05

Adding two tiles element by element only means anything if element (i, j) of
each is in the same lane, so an elementwise op is a CONSTRAINT that its
operands share a lane axis -- not a place the decision passes through in one
direction. _lane_demands therefore closes over the component in both
directions: forward from the bias into the generic, then BACKWARD out of the
generic into its other operand, where the matmul is.

gemm.py cannot catch this. With a single loaded-tile family the old per-function
`any(linalg.matmul in func)` guess and the per-tile decision agree, which is why
gemm and bmm report the same digits before and after. This kernel is the first
where they do not.

WHAT PROVES IT FIRED
  All four transfers (A, B, bias, C) carry `vlane_split_axis = 1` in
  08-transformed.mlir. A bias transfer with axis 0 next to a matmul is the bug.

matmul and a separately loaded tile meeting at an elementwise op. The case that needs the demand closure to walk BACKWARD out of a generic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# Tile dims must divide by TORCHSIM_COMPILE_VECTORLANE_SIZE (128): the DMA splits dim 0
# across lanes and pytorchsim_triton_compiler/lowering/passes/b09_vectorize.py divides the iteration space by it.
# K = 2 * BK on purpose -- one K tile would let triton unroll the reduction
# loop away, and lower_to_vcix needs that loop to mark as accumulation_loop.
M, N, K = 128, 128, 256
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mixed_gemm_bias_kernel(a_ptr, b_ptr, s_ptr, c_ptr,
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

        bias = tl.load(s_ptr + offs_m[:, None] * N + offs_n[None, :])
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc + bias)

    return mixed_gemm_bias_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32),
        "s": torch.randn(M, N, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"], inputs["b"]) + inputs["s"]}


SPEC = KernelSpec(
    name="mixed_gemm_bias_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "s_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[
        Arg("a", IN, "float32", (M, K)),
        Arg("b", IN, "float32", (K, N)),
        Arg("s", IN, "float32", (M, N)),
        Arg("c", OUT, "float32", (M, N)),
    ],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    # Accumulation order differs from torch's, so the tolerance is looser than
    # the elementwise kernels'.
    rtol=1e-3, atol=1e-3,
)
