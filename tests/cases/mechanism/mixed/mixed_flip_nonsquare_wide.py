"""mixed_flip_nonsquare with the aspect ratio the OTHER way round.

The flip retypes the accumulator by REVERSING its shape, and a rewrite that
reverses can be written to work in one direction and not the other -- the tile
that is tall going in is wide coming out, and every stride, size and offset
swaps with it. This kernel is [256, 128] where its sibling is [128, 256], so the
two disagree about which dimension is the long one at every step.

That is worth a second kernel because it is exactly how the last non-square bug
here was found: mixed_gemm_nonsquare turned up err 41.54 in a matmul path that
128-cubed tiles could not tell apart, because M, K and N were indistinguishable
in every kernel that existed. A transpose is the same trap one level up.

Both flips also land on the same systolic constraint from opposite sides. The
flip makes M the new N, and lower_to_vcix wants N and K to be exact multiples of
the array's dimension with M free:

    sibling   M=128 N=256 K=128   ->  N=128 M=256 K=128
    this one  M=256 N=128 K=128   ->  N=256 M=128 K=128

so both are legal, and the operand each rule applies to is not the same one.

WHAT PROVES IT FIRED
Same as the sibling: the numbers are right, and 04-transformed.mlir shows the matmul
operands swapped with the accumulator at [128, 256] and the reduction on dim 0.

the non-square flip with the aspect ratio reversed -- a rewrite that reverses shapes can be right in one direction only.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# K = 2 * BK so the accumulator stays loop-carried (see the sibling).
M, N, K = 256, 128, 256
BM, BN, BK = 256, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mixed_flip_nonsquare_wide_kernel(a_ptr, b_ptr, c_ptr,
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

    return mixed_flip_nonsquare_wide_kernel


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
    name="mixed_flip_nonsquare_wide_kernel",
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
    # Reassociated sum, same as the sibling.
    rtol=1e-3, atol=1e-3,
)
