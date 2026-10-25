"""The row gather x[i[:, None]*N + arange(N)] -- where the two passes MERGE.

This is the canonical indirect-row-read (an embedding lookup). It is the only
shape in which PtrAnalysis has to combine a structured dim with a
non-structured one, so it is the only shape that exercises
mergeUnstructuredState and addState's two "the other side is all zeros" fast
paths at once.

SIBLINGS
  ptr_gather_1d.py / ptr_gather_1d_arith.py -- rank 1, nothing to merge.
  ptr_gs_memref_trips.py -- the SAME pointer shape carried further, into
  the four values the gather dim's mask entry can take. This file stops at
  01-structured.mlir on purpose; that one is about what b11 does with it.

COVERS (4)
  t2u-makegatherscattertptr-noop   TritonToUnstructuredPass.cpp:397
      The Case<tts::MakeGatherScatterTensorPtrOp> arm: unconditional
      success() that does NOTHING, because triton-to-structured already built
      the gather/scatter tensor pointer and TritonToUnstructured must leave it
      alone. Measured at 3 hits from this kernel -- the two
      make_gather_scatter_tptr ops below plus the tts.load's operand walk.
      Proof is an ABSENCE: %14's base stays %arg0 rather than being rewritten
      to base+offset, and no tts.gather is created for it.
      (Inherited from ptr_gathered.py, which claimed this id and was the last
      case in it still to be re-homed.)

             lib/AnalysisStructured/PtrAnalysis.cpp
  merge-success                    PtrAnalysis.cpp:641
      mergeUnstructuredState: ONLY the gather dim's offset/stride/shape are
      taken from the unstructured pass; every other dim keeps what the
      structured pass computed. That is why the emitted op can have a tensor
      offset on dim 0 and a plain literal stride on dim 1.
  addstate-unstruct-lhs-zero       PtrAnalysis.cpp:317
      addState, unstructured pass, dim 1: the LHS (`i[:, None]*16`) has
      offset 0 AND stride 0 there -- rebuildAsUnsupportedOp zeroed every
      size-1 dim -- so the RHS's offset/stride are taken verbatim, no arith.
  addstate-unstruct-rhs-zero       PtrAnalysis.cpp:322
      the mirror, dim 0: the RHS (`arange(16)[None, :]`) is the size-1 side
      there, so the LHS wins. ONE load takes both arms, on different dims --
      which is the cheapest possible demonstration that these two `if`s are
      per-dimension, not per-operand.

WHAT PROVES IT (01-structured.mlir, verbatim)
      %5  = "tts.load"(%3) : (tensor<8x!tt.ptr<i32>>) -> tensor<8xi32>
      %9  = tts.make_gather_scatter_tptr %arg0 to sizes: [8, 1]
            gather_scatter_dim: 0 gather_scatter_offset: %5,
            strides: [%c16_0, 0], offsets: [0, 0]
      %14 = tts.make_gather_scatter_tptr %arg0 to sizes: [8, 16]
            gather_scatter_dim: 0 gather_scatter_offset: %5,
            strides: [%c16_1, 1], offsets: [0, 0]
            : tensor<8xi32>  <f32> to !tt.ptr<tensor<8x16xf32>>
      %16 = "tts.load"(%14) <{static_mask_dims = array<i64>}>
            : (!tt.ptr<tensor<8x16xf32>>) -> tensor<8x16xf32>
  Read %9 and %14 together. %9 is the state after `i[:, None] * 16` alone: dim 1
  is the size-1 axis, offset 0 and STRIDE 0. %14 is after adding
  `arange(16)[None, :]`, and dim 1 is now `1` -- taken verbatim from the rhs
  because the lhs was all zeros there (:317). Dim 0 went the other way (:322).
  %9 has no consumer; it stays until --remove-dead-values.
  `strides: [%c16_1, 1]` is also the merge in one line: slot 0 (16) came from
  the unstructured re-analysis, slot 1 from the structured pass, untouched
  (:641). The offset operand is the raw loaded %5 -- rank 1, because only the
  gather dim may carry a tensor, and there is no arith.muli/addi at all between
  %5 and %14: both zero fast paths fired, so neither dim needed a computation.
  The store takes the same shape through rewriteStoreOp, giving a second
  tts.make_gather_scatter_tptr on %arg2 and a tts.store on it -- i.e. this file
  also shows the ptr operand of tts.store in its block-pointer form
  (`!tt.ptr<tensor<8x16xf32>>`), not the tensor-of-pointers form.

WHY IT MATTERS
  Every gather that is not rank 1 goes through mergeUnstructuredState, and the
  merge is deliberately narrow: it copies exactly one dim. If the two passes
  disagree about WHICH dim that is, the merge refuses -- see
  tests/cases/ptr_merge_dim_mismatch.py for that failure, and
  ptr_gather_multidim_refused.py for the case where no single dim exists.

torch equivalent: o.view(8, 16)[j, :] = x[i[j]*16 : i[j]*16 + 16], i.e. an
8-row gather from x and the same 8 rows scattered into o.

Row gather + row scatter: mergeUnstructuredState and addState's two zero fast paths, one per dimension of a single load.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NX = 4096
NI = 8
BM = 8
BN = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_2d_row_kernel(x_ptr, i_ptr, o_ptr,
                                 BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        i = tl.load(i_ptr + rm)                     # rows to read, from memory

        # dim 0 unstructured (offset = i, stride = 16), dim 1 structured.
        a = tl.load(x_ptr + i[:, None] * BN + rn[None, :])
        tl.store(o_ptr + i[:, None] * BN + rn[None, :], a)

    return ptr_gather_2d_row_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            # duplicate-free so the scatter is a well-defined permutation
            "i": torch.randperm(NX // BN, generator=g)[:NI].to(torch.int32)}


def reference(inputs):
    import torch
    x, i = inputs["x"], inputs["i"].long()
    o = torch.zeros(NX, dtype=torch.float32)
    for j in range(NI):
        o[i[j] * BN:i[j] * BN + BN] = x[i[j] * BN:i[j] * BN + BN]
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_gather_2d_row_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int32", (NI,)),
          Arg("o", OUT, "float32", (NX,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
