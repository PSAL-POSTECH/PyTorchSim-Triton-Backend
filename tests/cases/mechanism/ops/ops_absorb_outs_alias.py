"""The outs of an elementwise generic ALIASES one of its ins, and a transpose is
absorbed into that generic's outs map.

FAILS TODAY, IN absorb_layout_ops, AND THE NUMBERS ARE THE ONLY PLACE IT SHOWS.
Found while making vectorize accept a transposed indexing map: until then
every generic with a non-identity map was refused before anything ran, so this
could not produce a wrong answer. It can now, and this is the smallest form of
it. tests/cases/mechanism/ptr/ptr_memref_masked_fill.py and
tests/cases/mechanism/ops/ops_shape.py fail the same way and for the same reason,
buried under a mask and a shape-op family respectively.

WHAT THE IR DOES. `tl.sum(a + b, axis=1)` arrives as

    alloc_1 <- MVIN dram_stride [C, 1]           a
    alloc_2 <- MVIN dram_stride [C, 1]           b
    %8 = linalg.generic ins(%6, %7) outs(%6)     maps all identity, %6 = alloc_1
    %t = linalg.transpose %8 permutation [1, 0]
         linalg.reduce ins(%t) dimensions = [0]

The `outs(%6)` is destination-passing -- linalg's way of naming a buffer to
write into -- and it is alloc_1, which is also ins #0. absorb_layout_ops folds the
transpose into the generic's outs map, which is correct and complete on its own:
`%8[d1,d0] = f(%6[d0,d1])` IS the transpose of `f(%6)`, and no other operand
needs to move. But it then follows the chain from the outs to what produced that
buffer (rule 4) and flips alloc_1's MVIN to dram_stride [1, C] as well -- while
ins #0's map, reading the same buffer, stays the identity. One buffer, two
layouts, and the op reads it in the one it was not staged in:

    after absorb   ins  %6 (staged transposed), map (d0,d1)
                   ins  %7 (staged plain),      map (d0,d1)
                   outs                         map (d1,d0)

so the result is `y[d1,d0] + y[d0,d1]` where it should be `2*y[d1,d0]`. On a
SQUARE tile that verifies, lowers and runs; the answer is a column sum added to
a row sum. Measured on ptr_memref_masked_fill: max_abs_err 7.02.

WHAT FIXES IT: absorbing into a generic's outs map does not change the layout of
anything the generic READS, so the transfer that staged an aliased ins must be
left alone -- or, if the chain really has to move, ins #0's map has to move with
it. Either is a change in p06_absorb_layout_ops.py, not here.

WHAT BREAKS IT AGAIN: any rewrite that treats a linalg destination operand as
"the buffer this op produces" without asking whether the same buffer is also
read. The square tile is deliberate -- a rectangular one would fail the verifier
and be caught for the wrong reason.

Reproducer: absorb_layout_ops folds a transpose into a generic's outs map AND flips the transfer that staged the ins aliasing that outs, leaving that ins' own map identity. Wrong numbers, no diagnostic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BS = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_absorb_outs_alias_kernel(y_ptr, o_ptr, BS: tl.constexpr):
        r = tl.arange(0, BS)
        p = y_ptr + r[:, None] * BS + r[None, :]
        a = tl.load(p)
        b = tl.load(p)
        # `a + b` writes into a's own buffer, and the axis-1 fold puts a
        # transpose in front of the reduce for absorb_layout_ops to place.
        tl.store(o_ptr + r, tl.sum(a + b, axis=1))

    return ops_absorb_outs_alias_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # NOT symmetric: y + y^T is what the bug computes, and on a symmetric input
    # that is 2*y and the kernel would pass while still being wrong.
    return {"y": torch.randn(BS * BS, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    y = inputs["y"].reshape(BS, BS)
    return {"o": (y + y).sum(dim=1).contiguous()}


SPEC = KernelSpec(
    name="ops_absorb_outs_alias_kernel", kernel=kernel,
    signature={"y_ptr": "*fp32", "o_ptr": "*fp32", "BS": "constexpr"},
    constexprs={"BS": BS},
    args=[Arg("y", IN, "float32", (BS * BS,)),
          Arg("o", OUT, "float32", (BS,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
