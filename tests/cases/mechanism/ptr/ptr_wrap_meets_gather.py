"""A wrap on one axis and a gather on another, on ONE pointer.

    idx = (r % M) * C + (c // 2)      wrap on axis 0, `//` on axis 1

tts.make_gather_scatter_tptr grew a `shape` slot for exactly this: a wrap on an
AFFINE axis has nowhere else to live, because that axis is walked by a stride
and only the gathered axis has an enumerated index to hide a `%` inside.

WHAT IT PINS. `shape: [24, 0]` on the gather pointer -- 24 because the modulus
is taken after the dim-0 stride, 0 on the gathered axis because a wrap there is
arithmetic in the index tensor. Without the slot the whole access fell to
tts.gather, and the two passes disagreed about the axis besides: the
unstructured one claimed the `%`'s axis and the merge refused.

R=8 and M=3 so the wrap FIRES -- rows 3..7 read rows 0..4 again.

A wrap axis and a stepped axis in one access: shape: [24, 0, 0] keeps the wrap while sizes: [8, 4, 2] splits the step. R=8 over M=3, so the wrap fires.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, M = 8, 8, 3
ASIZE = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def wrap_meets_gather(a_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr,
                          M: tl.constexpr):
        r = tl.arange(0, R)[:, None]
        c = tl.arange(0, C)[None, :]
        idx = (r % M) * C + (c // 2)
        v = tl.load(a_ptr + idx)
        tl.store(o_ptr + (r * C + c), v)

    return wrap_meets_gather


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(ASIZE, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    a = inputs["a"]
    r = torch.arange(R).reshape(R, 1)
    c = torch.arange(C).reshape(1, C)
    idx = (r % M) * C + (c // 2)
    return {"o": a[idx].reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="wrap_meets_gather", kernel=kernel,
    signature={"a_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr",
               "C": "constexpr", "M": "constexpr"},
    constexprs={"R": R, "C": C, "M": M},
    args=[Arg("a", IN, "float32", (ASIZE,)),
          Arg("o", OUT, "float32", (R * C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    # WITHOUT THE PIN THIS PASSES EITHER WAY -- the fully unstructured route
    # computes the same addresses, just one per element. The claim is the rung,
    # and the rung is now the TOP one: `c // 2` splits axis 1 instead of
    # gathering it, so the whole access is one make_tptr again.
    # READS split-stepped-axes ON, which is how pipeline.py ships: the access is
    # at the TOP rung -- sizes [8, 4, 2], shape [24, 0, 0], the wrap on axis 0
    # and `c // 2` splitting axis 1. With the flag off it sat two rungs down at
    # {"tts.make_gather_scatter_tptr": 1}, an index table per row.
    expect_ir={"03-linalg.mlir": {"tts.make_gather_scatter_tptr": 0,
                                    "tts.make_tptr": 2}},
)
