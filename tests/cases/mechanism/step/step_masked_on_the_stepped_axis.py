"""GRID: a mask box on the axis that steps.

THE PIECE AND THE MASK WANT THE SAME SLOT. Each piece names its own rows in the
descriptor's box, and the mask needs that box to say which rows take the fill;
one axis has one box, so the two cannot both be spelled. The stamp refuses and
the access is addressed per element instead -- which is why _address_tile had
to learn `p // K` before this cell existed at all.

Same shape as the wrap's version of this refusal (_narrowed_axes & wdims), and
for the same reason.

A clamp on the stepped axis: the piece's box and the mask's box are one slot, so the pieces are refused and the index carries it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K, B = 32, 5, 21
NX = (N + K - 1) // K + 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def step_masked_on_the_stepped_axis_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                               K: tl.constexpr,
                                               B: tl.constexpr):
        p = tl.arange(0, N)
        v = tl.load(x_ptr + p // K, mask=p < B, other=0.0)
        tl.store(o_ptr + p, v)

    return step_masked_on_the_stepped_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    p = torch.arange(N)
    return {"o": torch.where(p < B, inputs["x"][p // K], torch.zeros(N))}


SPEC = KernelSpec(
    name="step_masked_on_the_stepped_axis_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "K": "constexpr", "B": "constexpr"},
    constexprs={"N": N, "K": K, "B": B},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
