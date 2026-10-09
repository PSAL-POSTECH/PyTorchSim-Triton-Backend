"""GRID: two splits of ONE axis at DIFFERENT points. `64*(p//64) + 2*(p//2)`.

alignSplits compared how MANY state axes a tensor axis holds, not where the cuts
fall. Both sides hold two here -- [4, 64] and [128, 2] -- so it passed them
through, and addState took the left side's sizes without asking the right.

FIXED IN 44a30d2: the two meet at their common refinement, [4, 32, 2], and the
access is rung 1. The sibling step_three_cuts_on_one_axis.py asks whether a
THIRD meets them too.

Two cuts of one axis, meeting at their common refinement.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K1, K2 = 256, 64, 2
NX = K1 * ((N - 1) // K1) + K2 * ((N - 1) // K2) + 1        # 447


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def step_two_cuts_kernel(x_ptr, o_ptr, N: tl.constexpr,
                             K1: tl.constexpr, K2: tl.constexpr):
        p = tl.arange(0, N)
        v = tl.load(x_ptr + (K1 * (p // K1) + K2 * (p // K2)))
        tl.store(o_ptr + p, v)

    return step_two_cuts_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    p = torch.arange(N)
    return {"o": inputs["x"][K1 * (p // K1) + K2 * (p // K2)]}


SPEC = KernelSpec(
    name="step_two_cuts_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "K1": "constexpr", "K2": "constexpr"},
    constexprs={"N": N, "K1": K1, "K2": K2},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE TILE HOLDS BOTH CUTS. [4, 32, 2] is where the two factorisations meet;
    # before 44a30d2 this demoted, and the numbers were the same either way.
    expect_ir={"07-regularized.mlir": {"tts.gather": 0, "tts.make_tptr": 2}},
)
