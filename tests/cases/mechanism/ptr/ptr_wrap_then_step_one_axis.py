"""`(p % W) // K` on ONE axis: the wrap survives the division at W/K.

Both decompositions land on the same axis, which is the shape RoPE writes --
`col = xh % DIM` and `head = xh // DIM` off one flat index. Until 2026-09-04
the divisor refused any state carrying a wrap at all, so the whole access fell
to a gather.

THE IDENTITY THAT MAKES IT ONE DESCRIPTOR

    (x mod W) // K  ==  (x // K) mod (W / K)        whenever K divides W

so the axis splits into (outer, inner) as any step does, and the wrap rides the
OUTER half at W/K -- the inner half is inside one lap by construction and needs
no modulus. Measured over the corpus before building it: all 16 of the
divisions refused for carrying a wrap had K | W, so the condition costs nothing.

    p in [0, 32), W = 16, K = 4
    ->  sizes [8, 4], strides [1, 0], shape [4, 0]
        outer walks 8 with a wrap of 16/4 = 4; inner stands still

WHAT IT IS NOT. A wrap on a STILL axis (stride 0) is refused, not carried: its
contribution is `o mod W`, a constant, and K need not divide that. Measured:
zero such states in the corpus, so this is a guard rather than a case.

A wrap and a step on one axis. The wrap rides the outer half of the split at W/K; without that the access is a gather.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, W, K = 32, 16, 4
NX = (W // K) * 8          # every row the quotient can name


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_then_step_one_axis_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                           W: tl.constexpr, K: tl.constexpr):
        p = tl.arange(0, N)
        v = tl.load(x_ptr + (p % W) // K)
        tl.store(o_ptr + p, v)

    return ptr_wrap_then_step_one_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    p = torch.arange(N)
    return {"o": inputs["x"][(p % W) // K]}


SPEC = KernelSpec(
    name="ptr_wrap_then_step_one_axis_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "W": "constexpr", "K": "constexpr"},
    constexprs={"N": N, "W": W, "K": K},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE CLAIM IS THE RUNG, and the answer is right either way -- a gather
    # computes the same addresses one at a time. Reads split-stepped-axes ON,
    # which is how pipeline.py ships: sizes [8, 4], strides [1, 0], shape
    # [4, 0] -- the wrap riding the OUTER half at W/K, which is the whole point
    # of the file. With the flag off it was an index table instead.
    expect_ir={"03-ttshared.mlir": {"tts.make_tptr": 2}},
)
