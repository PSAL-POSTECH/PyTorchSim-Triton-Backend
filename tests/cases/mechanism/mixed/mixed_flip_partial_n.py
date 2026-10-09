"""A matmul+reduce flip that LANDS ON a tile narrower than the array.

The flip turns A[M,K] @ B[K,N] into Bt[N,K] @ At[K,M], so M becomes the new N --
the array's width. Here M is 64 and the array is 128, so the flip produces
exactly the shape mixed_gemm_partial_n covers: half the columns get no weights.

WHY THIS IS ITS OWN KERNEL
The two facts meet here and nowhere else. mixed_gemm_partial_n has a partial
tile with no flip; mixed_flip_nonsquare has a flip that lands on a full one.
Only a flip that lands on a PARTIAL one exercises the rule that connects them,
and that rule changed:

    was    a flip declines unless M is a multiple of the array dim
    now    it declines only if M is LARGER than it and not a multiple

The old rule existed because err 41.54 at N=64 was read as the array being
unable to take a partial tile, so a flip onto M=64 looked like it was
manufacturing that case. It was the scratchpad reservation instead. With that
fixed, declining here would refuse a kernel that works -- and the refusal would
name N=64, a dimension this kernel does not have, which is the confusing failure
the guard was meant to avoid in the first place.

WHAT PROVES IT FIRED
08-transformed.mlir shows the matmul operands swapped with the accumulator at
[128, 64] and the reduction on dim 0. Before the relaxation the flip bailed and
fit_layout_to_lanes refused with "axis 1 must be the lane axis (linalg.matmul) and
cannot be (linalg.reduce folds dim 1)" -- or, once the DRAM path existed, went
around through DRAM for a conflict that did not need it.

the flip lands on a tile narrower than the array -- the case the old multiple-of-SS guard on M was there to prevent, before the scratchpad reservation turned out to be the real fault.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# M = 64 is the point: after the flip it is the array WIDTH, and it is half of
# it. K = 2 * BK so the accumulator stays loop-carried.
M, N, K = 64, 128, 256
BM, BN, BK = 64, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mixed_flip_partial_n_kernel(a_ptr, b_ptr, c_ptr,
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

    return mixed_flip_partial_n_kernel


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
    name="mixed_flip_partial_n_kernel",
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
    # Reassociated sum, as for every flip kernel.
    rtol=1e-3, atol=1e-3,
)
