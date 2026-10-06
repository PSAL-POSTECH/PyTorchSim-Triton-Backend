"""A general-combiner reduce whose INPUTS come out of an `scf.for`.

The split-reduction shape, which is what Inductor emits whenever a row does not
fit one tile: a loop folds the row into R partial columns, and a second
reduction combines those R partials. Here the combine is flash attention's
(max, sum) pair, so the second reduction is the general path -- seed from slice
0 along the reduced axis, fold the rest.

WHAT IS NEW ABOUT IT, AND IT IS NOT THE COMBINER. reduce_general_combiner.py
already covers the body. What this adds is WHERE THE OPERANDS COME FROM: `accm`
and `accl` cross the loop's back edge, so the seed slice bufferizes into

    memref.copy %subview, %alloc : memref<128xf32, strided<[512]>> to ...

with the subview taken from an `scf.for` RESULT. bufferize's `_internal_buffer`
walks a loop's block argument back to its init and stopped at a loop result,
reporting the op name `scf.for` as if it were a storage -- so `_bank_copies`
refused a copy whose buffer it could in fact name:

    _Fatal: memref.copy between storage this pass cannot place (scf.for and
            memref.alloc)

Its own docstring states the identity as four spellings -- "init i, block
argument i+1, result i and yielded value i are one buffer" -- and read three of
them. Reading the fourth is the fix, and it is the same arithmetic from the
other side.

WHERE IT CAME FROM. ViT's LayerNorm, fused with the patch convolution and a
transpose, split over 512-wide partials:
`triton_npu_fused_add_addmm_cat_convolution_expand_native_layer_norm_permute_
transpose_view_15`.

WHAT WOULD BREAK THIS
  * the loop-result hop going away again -- adapt dies in bufferize.
  * off-by-one in that hop (operand `i + 3`: lb, ub, step, then the inits). A
    wrong operand roots at some OTHER init, and with two carried tensors here
    that is a different buffer of the same type, so the refusal does not fire
    and the copy is banked against the wrong storage.
  * R must stay > 1: at R == 1 the general converter yields the slices with no
    linalg.reduce at all and the copy never appears.

A split reduction: loop-carried partials combined by a general tl.reduce. Fixes bufferize._internal_buffer, which walked a loop's block argument to its init but stopped at a loop result.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, R = 128, 64, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _online_softmax(m1, l1, m2, l2):
        m = tl.maximum(m1, m2)
        l = l1 * tl.exp(m1 - m) + l2 * tl.exp(m2 - m)
        return m, l

    @triton.jit
    def reduce_general_over_loop_partials_kernel(
            x_ptr, m_ptr, l_ptr, M: tl.constexpr, N: tl.constexpr,
            R: tl.constexpr):
        rm = tl.arange(0, M)
        rr = tl.arange(0, R)
        accm = tl.full((M, R), -1e30, tl.float32)
        accl = tl.zeros((M, R), dtype=tl.float32)
        # THE LOOP. Both accumulators cross the back edge, so the tile the
        # reduction below slices is an scf.for result.
        for k in range(0, N, R):
            x = tl.load(x_ptr + rm[:, None] * N + (k + rr)[None, :])
            newm = tl.maximum(accm, x)
            accl = accl * tl.exp(accm - newm) + tl.exp(x - newm)
            accm = newm
        m, l = tl.reduce((accm, accl), 1, _online_softmax)
        tl.store(m_ptr + rm, m)
        tl.store(l_ptr + rm, l)

    return reduce_general_over_loop_partials_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    # Folding this pair over a whole row, however it is partitioned, IS the row
    # max and the stable softmax denominator.
    import torch
    x = inputs["x"]
    m = x.amax(dim=1)
    return {"m": m, "l": torch.exp(x - m[:, None]).sum(dim=1)}


SPEC = KernelSpec(
    name="reduce_general_over_loop_partials_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "m_ptr": "*fp32", "l_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "R": "constexpr"},
    constexprs={"M": M, "N": N, "R": R},
    args=[Arg("x", IN, "float32", (M, N)),
          Arg("m", OUT, "float32", (M,)),
          Arg("l", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
