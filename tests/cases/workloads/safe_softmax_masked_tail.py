"""`_safe_softmax` over a row that does NOT fill the block: two custom reduces.

Attention's softmax as Inductor writes it, and the shape ViT's arrives in -- a
197-long row (196 patches plus the class token) in a 256-wide block, so 59 lanes
of every tile are mask. Both of its reductions have a body the linalg lowering's three
named converters decline, so both take the general path added for Welford:

    triton_helpers.max2   tl.reduce(x, 1, maximum)   3 ops: `a > b`, `|= a != a`,
                                                     `where` -- a max that lets a
                                                     NaN operand win, which
                                                     `arith.maximumf` does not
    triton_helpers.any    tl.reduce(x, 1, a | b)     an i1 fold

`_safe_softmax` is not `softmax`: a row that is ENTIRELY masked has no finite
max, and `exp(-inf - -inf)/sum` is nan, so torch defines the output as zero
there. That is what `any` is for -- it asks whether any element of the row is
not -inf -- and it is why this kernel has two reductions rather than one.

WHAT THE MASK DOES TO THE SEED, WHICH IS THE POINT OF THE FILE. The general
lowering seeds the accumulator from element 0 along the reduced axis and folds
elements 1..255. Every one of those, real or tail, has already been through

    tl.where(r0_mask, x, -inf)      for the max
    tl.where(r0_mask, x, False)     for the any

so the tail is the combiner's identity by construction and the fold is exact --
PROVIDED the seed and the fold see the same masked tile. A seed taken before the
where, or a fold that starts at 0 instead of 1, or a rest slice that runs to 255
instead of 256, all give an answer that is close but wrong, and a softmax that
is close but wrong is the hardest thing in this suite to notice: every element
is off by a few percent and nothing is nan.

THE ROWS ARE CHOSEN SO THAT IT SHOWS
  * row 0 is entirely -inf              -- exercises `any` returning false, and
                                           is the only row whose answer is not a
                                           distribution
  * row 1 has its ONLY finite value at  -- if the seed were the tail's -inf and
    the last real column, 196              the fold skipped element 0, the max
                                           would still be right here; putting
                                           the peak at the far end instead makes
                                           a fold that stops early wrong by the
                                           whole row
  * row 2 peaks at column 0             -- the seed itself. A fold that
                                           double-counts the seed is invisible
                                           in a max (idempotent) and shows in
                                           the sum below it
  * the rest are ordinary               -- 0.064 was the measured abs error on
                                           ViT's buf18, over 416920 of 465708
                                           elements, which is what an ordinary
                                           row looks like when this is wrong

WHERE IT CAME FROM. test_vit's `triton_npu_fused__safe_softmax_view_10`, the
first buffer its per-kernel functional verify rejects.

Attention's safe softmax with a masked block tail: a NaN-aware max and an i1 any, both through the general reduce lowering, both over a row shorter than the block.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 197
BLOCK = 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _maximum(a, b):
        # triton_helpers.maximum, transcribed: a NaN operand wins, which is what
        # torch.maximum promises and `tl.maximum` does not do.
        mask = (a > b) | (a != a)
        return tl.where(mask, a, b)

    @triton.jit
    def _any_combine(a, b):
        return a | b

    @triton.jit
    def safe_softmax_masked_tail(x_ptr, o_ptr, M: tl.constexpr,
                                 N: tl.constexpr, BLOCK: tl.constexpr):
        rm = tl.arange(0, M)[:, None]
        rn = tl.arange(0, BLOCK)[None, :]
        mask = rn < N
        x = tl.load(x_ptr + rm * N + rn, mask, other=0.0)

        finite = tl.reduce(tl.where(mask, x != float("-inf"), False), 1,
                           _any_combine)[:, None]
        mx = tl.reduce(tl.where(mask, x, float("-inf")), 1, _maximum)[:, None]
        e = tl.exp(x - mx)
        s = tl.sum(tl.where(mask, e, 0.0), 1)[:, None]
        tl.store(o_ptr + rm * N + rn, tl.where(finite, e / s, 0.0), mask)

    return safe_softmax_masked_tail


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, N, generator=g, dtype=torch.float32)
    ninf = float("-inf")
    x[0, :] = ninf                      # no finite element at all
    x[1, :] = ninf
    x[1, N - 1] = 1.0                   # the only finite value, at the far end
    x[2, 0] = 10.0                      # the peak IS the seed
    return {"x": x}


def reference(inputs):
    import torch
    x = inputs["x"]
    finite = torch.isfinite(x).any(dim=1, keepdim=True)
    mx = x.amax(dim=1, keepdim=True)
    e = torch.exp(x - mx)
    out = e / e.sum(dim=1, keepdim=True)
    return {"o": torch.where(finite, out, torch.zeros_like(out))}


SPEC = KernelSpec(
    name="safe_softmax_masked_tail", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr",
               "N": "constexpr", "BLOCK": "constexpr"},
    constexprs={"M": M, "N": N, "BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
