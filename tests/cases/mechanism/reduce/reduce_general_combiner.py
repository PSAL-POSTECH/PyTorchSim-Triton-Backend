"""`tl.reduce` with a combiner that is not one arith op: the online-softmax pair.

    combine((m1, l1), (m2, l2)):
        m = max(m1, m2)
        l = l1 * exp(m1 - m) + l2 * exp(m2 - m)
        return m, l

Associative, and the two results GENUINELY INTERACT -- `l` is rescaled by how
much the running max moved, so neither half can be computed without the other.
It is flash attention's combiner, and it is the shape `var_mean`'s Welford
combiner has too (three tensors instead of two, same structure), which is what
ViT's fused LayerNorm arrives as.

WHY IT DID NOT LOWER, AND IT IS NOT THE FOLD THAT WAS MISSING. vectorize
has folded an arbitrary body one element at a time since argmax
(`_bank_reduce_general`, generic in the number of results, `n = len(op.operands)
// 2`). What was missing is one step earlier: triton-shared's `ReduceConverter`
declined to make a `linalg.reduce` at all --

    if (reductionOps.size() != 1 || !isReductionOpSupported(...))
      return notifyMatchFailure(op, "Only support lowering reduction with body
                                     containing 1 max(i/f), addf, ori, or mulf.")

so the `tt.reduce` survived stage 3 as an unregistered op holding tensors and
died below it. The refusal is about the INIT, not the body: the converter starts
the accumulator from a constant it looks up per op (0 for addf, -inf for maximumf
...) and an arbitrary body has no such table. Upstream's own comment says what
the general answer is -- "using the first element across the reduction dimension
requires us to iterate over a subview that skips over each first element" -- and
that is what now happens: two `tensor.extract_slice`s per operand, one taking
slice 0 as the init and one taking the remaining extent as the input.

Seeding matters HERE and this kernel is chosen so that it does. The pair's
identity is `(-inf, 0)`, and folding it through this body computes
`0 * exp(-inf - -inf)` = `0 * exp(nan)` = **nan** -- an identity that exists
mathematically and is unusable numerically. Only the first slice works.

WHAT WOULD BREAK THIS
  * seeding from slice 0 but reducing over the WHOLE extent -- element 0 folded
    twice. `l` comes out with row 0's term doubled, `m` is unchanged (max is
    idempotent), so a max-only kernel would NOT catch it. The sum is why the
    pair is here.
  * dropping the rank reduction on the init slice, so the init is [M, 1] where
    the reduce wants [M].
  * a reduced extent of 1, where the "rest" is empty. N is 16 here; that case is
    handled by replacing the reduce with the slices and is not covered by this
    file.

A two-result tl.reduce whose combiner is four arith ops and whose results depend on each other. Fixes the general path in triton-shared's ReduceConverter (init = first slice along the reduced axis) feeding vectorize._bank_reduce_general.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _online_softmax(m1, l1, m2, l2):
        m = tl.maximum(m1, m2)
        l = l1 * tl.exp(m1 - m) + l2 * tl.exp(m2 - m)
        return m, l

    @triton.jit
    def reduce_general_combiner_kernel(x_ptr, m_ptr, l_ptr, M: tl.constexpr,
                                       N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
        ones = tl.full((M, N), 1.0, tl.float32)
        m, l = tl.reduce((x, ones), 1, _online_softmax)
        tl.store(m_ptr + rm, m)
        tl.store(l_ptr + rm, l)

    return reduce_general_combiner_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    # Folding this combiner over a row with every weight 1 IS the softmax
    # denominator taken in a numerically stable way: m is the row max and l is
    # sum(exp(x - m)).
    import torch
    x = inputs["x"]
    m = x.amax(dim=1)
    return {"m": m, "l": torch.exp(x - m[:, None]).sum(dim=1)}


SPEC = KernelSpec(
    name="reduce_general_combiner_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "m_ptr": "*fp32", "l_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)),
          Arg("m", OUT, "float32", (M,)),
          Arg("l", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
