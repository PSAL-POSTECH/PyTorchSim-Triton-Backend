"""Welford's combiner: THREE accumulators, and a body that captures a constant.

    combine((mean1, m2_1, w1), (mean2, m2_2, w2)):
        delta = mean2 - mean1
        w     = w1 + w2
        r     = 0.0 if w == 0.0 else w2 / w        <- the capture
        return mean1 + delta * r,
               m2_1 + m2_2 + delta * delta * w1 * r,
               w

This is `torch._inductor`'s `welford_combine`, transcribed. It is what a
`var_mean` becomes whenever Inductor does not take the two-pass fallback, and
ViT's first LayerNorm -- fused with the patch convolution and a permute -- is
exactly that: `triton_npu_fused_add_cat_convolution_expand_native_layer_norm_
permute_view_3`.

WHAT IT FIXES, AND IT IS NOT THE THIRD ACCUMULATOR. `_bank_reduce_general` has
always been generic in the number of results (`n = len(op.operands) // 2`), so
three folds like two. What it could not do is read an operand its body did not
define:

    %10 = arith.cmpf oeq, %9, %cst : f32       %cst = arith.constant 0.0,
    %12 = arith.select %10, %cst, %11 : f32    HOISTED out of the region

Welford guards `w2 / w` against a zero weight, so the body tests against a
constant and selects it. A constant is loop invariant, canonicalize parks it in
the enclosing block, and the region then refers to a value defined outside it.
`_widen_body` indexed its `env` directly and said "no captures" in its own
docstring, so this arrived as

    KeyError: <mlir._mlir_libs._mlir.ir.OpResult object at 0x...>

with nothing naming the value or the pass. It now broadcasts a capture, which is
what the generic elementwise path a thousand lines below has always done for the
same thing -- one element per lane, in every lane.

WHY THE GUARD HAS TO STAY IN THE KERNEL. Dropping the `tl.where` removes the
capture and the kernel passes without the fix -- that is the whole defect. Every
weight here is 1, so the guarded branch never fires and the numbers do not
depend on it; it is in the source because Inductor puts it there, and because it
is the only thing in this file that makes the body reach outside itself.

WHAT ELSE WOULD BREAK THIS
  * seeding from slice 0 and then folding slice 0 again: `w` comes out N+1
    instead of N, so the variance is scaled wrong. That is why `w` is returned
    and checked rather than left implicit.
  * an init that is not rank-reduced, or an operand order that pairs the wrong
    accumulator with the wrong element -- `m2` is the only one of the three that
    is sensitive to both, and it is checked against torch's biased variance.

Inductor's welford_combine as a three-operand tl.reduce. Fixes _widen_body, which could not re-emit a body that refers to a value defined outside its region.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _welford_combine(mean1, m2_1, w1, mean2, m2_2, w2):
        delta = mean2 - mean1
        w = w1 + w2
        # THE CAPTURE. Inductor's own guard against an empty partial; with every
        # weight 1 it never fires, and it is the reason this file exists.
        r = tl.where(w == 0.0, 0.0, w2 / w)
        return (mean1 + delta * r,
                m2_1 + m2_2 + delta * delta * w1 * r,
                w)

    @triton.jit
    def reduce_welford_captured_const_kernel(x_ptr, mean_ptr, m2_ptr, w_ptr,
                                             M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
        zero = tl.zeros((M, N), dtype=tl.float32)
        one = tl.full((M, N), 1.0, tl.float32)
        mean, m2, w = tl.reduce((x, zero, one), 1, _welford_combine)
        tl.store(mean_ptr + rm, mean)
        tl.store(m2_ptr + rm, m2)
        tl.store(w_ptr + rm, w)

    return reduce_welford_captured_const_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    mean = x.mean(dim=1)
    # m2 is the SUM of squared deviations -- the biased variance times the count
    # -- which is what the fold accumulates before anyone divides.
    m2 = ((x - mean[:, None]) ** 2).sum(dim=1)
    return {"mean": mean, "m2": m2,
            "w": torch.full((M,), float(N), dtype=torch.float32)}


SPEC = KernelSpec(
    name="reduce_welford_captured_const_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "mean_ptr": "*fp32", "m2_ptr": "*fp32",
               "w_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)),
          Arg("mean", OUT, "float32", (M,)),
          Arg("m2", OUT, "float32", (M,)),
          Arg("w", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
