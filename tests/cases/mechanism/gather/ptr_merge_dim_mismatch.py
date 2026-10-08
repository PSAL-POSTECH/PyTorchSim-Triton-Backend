"""the two analysis passes disagree about WHICH axis is the gather.

Split out of the old tests/cases/ptr_reject.py (one case per file).
The merge failure IS the observable; do not try to make this address lower.
(The MODULE still lowers -- see the end of "WHAT PROVES IT FIRED".)

COVERS  (source root /workspace/triton_shared/, worktree lines)

  merge-diff-dim-fail        lib/AnalysisStructured/PtrAnalysis.cpp:629
      PtrState::mergeUnstructuredState() rejects a merge when
      `other.getNonStructuredDim() != getNonStructuredDim()`.
      Observable: remark "PtrAnalysis: do not support merging pointer states
      with different non-structured dimensions".

  addptr-merge-unstruct-fail lib/AnalysisStructured/PtrAnalysis.cpp:1429
      The caller: the unstructured RE-ANALYSIS of the tt.addptr succeeded on
      its own, but `state.mergeUnstructuredState(unstructuredState, op)`
      failed, so visitAddptr gives up.
      Observable: both passes' side-effect arith ops are left in the IR but
      NO tts.make_gather_scatter_tensor_ptr is created for this address, plus
      the remark "PtrAnalysis: Failed to merge unstructured state for
      tt.addptr".

THE DELIBERATE SHAPE -- keep both halves
  The address is

      tl.minimum(r8[:, None] % M, j8[None, :])

  and it is built so that the two passes pick DIFFERENT axes:

    * the STRUCTURED pass runs visitOperandMinSI, sees a structured LHS
      (`r8[:, None] % M`, a modulo of an arange) against an unstructured RHS
      (`j8[None, :]`, loaded from idx[]), takes the rhs-unstructured arm and
      records non-structured dim = 1;
    * the UNSTRUCTURED re-analysis instead latches onto the modulo LHS, which
      lives on dim 0.

  1 != 0, so :629 refuses and :1429 propagates the refusal. Remove the `% M`
  and both passes agree on dim 1; drop the loaded `j8` and there is nothing
  unstructured to merge at all. Both halves are load-bearing.

  Note this address ALSO walks minsi-rhs-unstructured (:931) on the way in --
  that is unavoidable, it is how dim 1 gets chosen. That branch is claimed by
  tests/cases/ptr_minmax_unstructured_operand.py; this file claims only
  the two merge branches.

WHAT PROVES IT FIRED
  `tools/lower_steps.py --debug` prints both remarks, in this order, and they
  map one-to-one onto the two claims:

      PtrAnalysis: encountered addptr operand produced by an unsupported
        operation: %4 = "tts.load"(%2) ... -> tensor<8xi32>
      :77:25: remark: PtrAnalysis: do not support merging pointer states
                      with different non-structured dimensions      <- :629
      :77:25: remark: PtrAnalysis: Failed to merge unstructured state
                      for tt.addptr                                 <- :1429
      :77:25: remark: PtrAnalysis: Failed to rewrite AddPtrOp
      :77:16: remark: PtrAnalysis: pointer is not replace with tts.make_tptr
                      so loadOp cannot be rewritten
      :77:16: remark: PtrAnalysis: Failed to rewrite LoadOp

  and 01-structured.mlir keeps BOTH passes' side-effect arith ops while
  producing no pointer op at all for this address:

      %6  = arith.remsi %5, %cst_0 : tensor<8x1xi32>       // the dim-0 modulo
      %10 = arith.minsi %8, %9 : tensor<8x8xi32>
      %11 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>>
      %c64 = arith.constant 64 : index                     // the modulo state,
                                                           // materialised and
                                                           // then orphaned
      %12 = tt.addptr %11, %10 : tensor<8x8x!tt.ptr<f32>>, tensor<8x8xi32>
      %13 = tt.load %12 : tensor<8x8x!tt.ptr<f32>>

  `%c64` is the tell: the unstructured re-analysis got far enough to emit the
  modulo's extent as an `index` before mergeUnstructuredState refused it, and
  nothing consumes it.  There is no tts.make_tptr and no
  tts.make_gather_scatter_tptr for `%12` -- compare the store on the next
  line, which gets `tts.make_tptr %arg2 to sizes: [8, 8], strides: [%c8, 1]`.

  The rest of the module lowers normally (the index load and the store both
  become tts ops, and `run.py --to triton-compile` reports ALL STAGES PASSED); it is
  this ONE address that stays a raw tt.load.

ARITHMETIC
  8x8 only, so the two axes are easy to tell apart in the IR. M = 64 makes
  `r8 % 64` the identity on 0..7, which keeps the intent (a modulo state on
  dim 0) without adding arithmetic to reason about.

reference() states the intended math (idx[] is zeroed, so it is in[0]
broadcast over the 8x8 tile) but nothing calls it: like the rest of the
coverage set this file is exercised with `run.py --to triton-compile`, and stage 3
(pytorchsim_triton_compiler/lowering/passes/) does not handle this shape.

merge-diff-dim-fail + addptr-merge-unstruct-fail. The structured and unstructured passes choose different gather axes for the same address, so this ONE load stays a raw tt.load.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
M = 64
R = 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_merge_dim_mismatch_kernel(in_ptr, idx_ptr, o_ptr,
                                      M: tl.constexpr, R: tl.constexpr):
        r8 = tl.arange(0, R)

        # structured pass picks the unstructured RHS (dim 1), unstructured pass
        # picks the modulo LHS (dim 0) -> the two states cannot be merged
        j8 = tl.load(idx_ptr + r8)
        s = tl.load(in_ptr + tl.minimum((r8[:, None] % M), j8[None, :]))
        tl.store(o_ptr + r8[:, None] * R + r8[None, :], s)

    return ptr_merge_dim_mismatch_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"in": torch.randn(N, generator=g, dtype=torch.float32),
            "idx": torch.zeros(R, dtype=torch.int32)}


def reference(inputs):
    """NEVER CALLED -- the coverage set stops at stage 2 (see the docstring).

    IT ALSO USED TO PASS BY ACCIDENT. `idx` is all zeros, so the address term
    j8[c] is the same value for every column -- which is exactly what hid a
    broadcast-axis bug in vectorize: the [1, 8] operand was read one
    element per lane and splatted, giving lane r the value j8[r] where the tile
    wanted j8[c]. With a zeroed idx the two are indistinguishable. Fill idx with
    anything varying and the numbers separate.

    vectorize now refuses the row-broadcast rather than emitting it, so
    this stops at stage 3 with a named reason instead of matching by luck.

    The intended math is in[minimum(i % 64, idx[j])] for i, j in 0..7, which
    with a zeroed idx[] is in[0] broadcast over the whole 8x8 tile.
    """
    import torch
    return {"o": inputs["in"][0].expand(R * R).clone()}


SPEC = KernelSpec(
    name="ptr_merge_dim_mismatch_kernel",
    kernel=kernel,
    signature={"in_ptr": "*fp32", "idx_ptr": "*i32", "o_ptr": "*fp32",
               "M": "constexpr", "R": "constexpr"},
    constexprs={"M": M, "R": R},
    args=[Arg("in", IN, "float32", (N,)),
          Arg("idx", IN, "int32", (R,)),
          Arg("o", OUT, "float32", (R, R))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
