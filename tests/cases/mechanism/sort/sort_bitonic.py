"""A bitonic network written out, so the compare-exchange is visible in the IR."""

#: WHY THIS REFUSES, TRACED 2026-09-23. An `expand_shape` splits the lane axis
#: of `graphcopy13_spad_8lane` into several axes, so no axis of the view is the
#: lane axis and lower_to_dma has nothing to read. The written-reshape bridge is what
#: answers that -- one leg in, one leg out -- and it does not fit here: the
#: view is live over [72, 217] and four `linalg.generic` and a `linalg.transpose`
#: reach the same storage inside that range, so the return leg cannot stand in
#: one place. Two legs need the storage reached through this view alone while
#: it is live; here it is not, and this says so rather than guessing.
#:
#: THE PER-USE TRICK DOES NOT CARRY OVER FROM SLICES, measured the same day.
#: `LowerCrossingSlices` needs no live range because it copies in front of each
#: USE, and a subview has one use each. A reshape does not: `collapse -> expand`
#: chains are common, and repointing one use leaves the chain inconsistent
#: (ops_shape: pass -> lower refusal), while repointing the view users too gives
#: WRONG NUMBERS (ops_shape 14.09, this kernel 4.10). A view of a materialised
#: buffer is not the view it replaced.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 16


def kernel():
    import triton
    import triton.language as tl

    # ONE STAGE, as its own jit function taking the geometry as constexpr
    # PARAMETERS. Everything else was tried and does not work: a `while` is a
    # runtime loop in triton, so the stride it carries becomes a tensor and
    # tl.reshape rejects it as a shape element; a jit body cannot read a module
    # global; and iterating a list literal is not something the AST walker
    # handles. A constexpr parameter IS accepted as a shape element, so the ten
    # stages of a 16-wide network are ten calls with literal geometry.
    @triton.jit
    def _stage(x, M: tl.constexpr, g: tl.constexpr, stride: tl.constexpr,
               size: tl.constexpr):
        # Bring the two partners into the last axis: [M,g,2,s] -> [M,g,s,2].
        y = tl.reshape(x, [M, g, 2, stride])
        y = tl.permute(y, (0, 1, 3, 2))
        lo, hi = tl.split(y)
        small = tl.minimum(lo, hi)
        large = tl.maximum(lo, hi)
        # Ascending where this block's `size` bit is clear. Every element of a
        # block shares that bit because size >= 2*stride always holds here.
        blk = tl.arange(0, g)[None, :, None]
        up = ((blk * (2 * stride)) % (2 * size)) < size
        y = tl.join(tl.where(up, small, large), tl.where(up, large, small))
        y = tl.permute(y, (0, 1, 3, 2))
        return tl.reshape(y, [M, g * 2 * stride])

    @triton.jit
    def sort_bitonic_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
        x = _stage(x, M, 8, 1, 2)
        x = _stage(x, M, 4, 2, 4)
        x = _stage(x, M, 8, 1, 4)
        x = _stage(x, M, 2, 4, 8)
        x = _stage(x, M, 4, 2, 8)
        x = _stage(x, M, 8, 1, 8)
        x = _stage(x, M, 1, 8, 16)
        x = _stage(x, M, 2, 4, 16)
        x = _stage(x, M, 4, 2, 16)
        x = _stage(x, M, 8, 1, 16)
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], x)

    return sort_bitonic_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].sort(dim=1).values}


SPEC = KernelSpec(
    name="sort_bitonic_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE DEFAULT IS THE CLAIM: upstream's descending layout puts the lanes on
    # the rows, so every exchange stays in its lane. On the sorted axis: 13 trips.
    expect_ir={"08-passes/04-pytorchsim-materialize-layouts.mlir": {"pytorchsim.convert_layout": 0}},
)
