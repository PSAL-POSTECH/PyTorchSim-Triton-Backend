"""An i16 `x & 1` beside a remainder bound: two ands, one op, two meanings.

`arith.andi` is the same op at every width -- logical at i1, bitwise at i16 --
so a pass that drops a `true &` has to ask the TYPE, and asking it as text says
yes to i16 (see the notes).

torch equivalent: (x.where(i < N - 3, 0)) & 1

WHAT BREAKS IT: asking whether an operand is i1 as `"i1" in str(type)`. `"i1"` is a substring of `"i16"`, so this kernel's `v & 1` reads as a true term and psto-fold-remainder-bound replaces the whole `and` with `v`. The IR verifies; only the numbers move. The sibling is mask_remainder_bound_is_true.py, which is this kernel without the arithmetic and.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 64
K = 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_i16_and_is_not_a_true_term_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                               K: tl.constexpr):
        i = tl.arange(0, N)
        # The left term cannot be false, which is what arms
        # psto-fold-remainder-bound; the right one is the real guard.
        keep = ((i % K) < K) & (i < N - 3)
        v = tl.load(x_ptr + i, keep, other=0)
        # AND THIS ONE IS ARITHMETIC. Same op, i16, and `& 1` is the low bit --
        # never the identity the i1 arm is entitled to fold away.
        tl.store(o_ptr + i, v & 1)

    return mask_i16_and_is_not_a_true_term_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(-3000, 3000, (N,), generator=g,
                               dtype=torch.int16)}


def reference(inputs):
    import torch
    x = inputs["x"]
    i = torch.arange(N)
    return {"o": (torch.where(i < N - 3, x, torch.zeros_like(x)) & 1)}


SPEC = KernelSpec(
    name="mask_i16_and_is_not_a_true_term_kernel",
    kernel=kernel,
    signature={"x_ptr": "*i16", "o_ptr": "*i16", "N": "constexpr",
               "K": "constexpr"},
    constexprs={"N": N, "K": K},
    args=[Arg("x", IN, "int16", (N,)), Arg("o", OUT, "int16", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    # The LEFT half of the mask, which the numbers alone do not pin: without
    # the fold the `%` stays inside it, MaskAnalysis declines, and the access
    # becomes a gather that computes the same answer more slowly.
    expect_ir={"03-linalg.mlir": {"tts.gather": 0, "tts.load": 1}},
)
