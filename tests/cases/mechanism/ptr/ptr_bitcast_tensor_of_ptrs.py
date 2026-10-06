"""bitcast of a TENSOR of pointers -- refused because the element type is a ptr.

Split out of the old tests/cases/ptr_reject.py (one case per file).
This is the SHAPED counterpart of ptr_bitcast_scalar_base.py: same tt.bitcast,
but applied to `in_ptr + r` (a tensor<128x!tt.ptr<f32>>) instead of to the bare
scalar pointer, and that one difference sends it down a completely different
arm and makes it fail.

COVERS  (source root /workspace/triton_shared/, worktree lines)

  rebuild-unsupported-ptr-elt-fail  lib/AnalysisStructured/PtrAnalysis.cpp:223
      rebuildAsUnsupportedOp() refuses to rebuild an operand whose element
      type is a triton::PointerType -- doing so would manufacture a
      non-structured SOURCE, which the rest of the analysis has no
      representation for.  It returns failure instead.
      Observable: the tt.bitcast on the tensor of pointers never becomes a
      gather base; the tt.load survives unlowered.

WHAT PROVES IT FIRED
  `tools/lower_steps.py --debug` prints, from pass 1:

      PtrAnalysis: encountered addptr operand produced by an unsupported
        operation: %4 = tt.bitcast %3
          : tensor<128x!tt.ptr<f32>> -> tensor<128x!tt.ptr<i32>>
      :70:37: remark: PtrAnalysis: Failed to rewrite AddPtrOp
      :70:32: remark: PtrAnalysis: pointer is not replace with tts.make_tptr
                      so loadOp cannot be rewritten
      :70:32: remark: PtrAnalysis: Failed to rewrite LoadOp

  and in 01-structured.mlir the shaped bitcast and its tt.load are still
  there, untouched, while the store next to them did get a make_tptr:

      %4 = tt.addptr %3, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %5 = tt.bitcast %4 : tensor<128x!tt.ptr<f32>> -> tensor<128x!tt.ptr<i32>>
      %6 = tt.addptr %5, %0 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      %7 = tt.load %6 : tensor<128x!tt.ptr<i32>>

  No tts.make_tptr and no tts.make_gather_scatter_tptr is produced for this
  address.

WHY IT IS INTERESTING
  visitOperandBitcast has two arms and they disagree about the same op:
  :1281 (scalar, ptr_bitcast_scalar_base.py) accepts the bitcast as a new
  base, while the shaped arm recurses and then dies in the generic rebuild
  path at :223.  Reading the two files side by side is the point -- a
  reinterpret cast of one pointer is fine, a reinterpret cast of a hundred
  and twenty-eight of them is not.

WHAT HAPPENS AFTERWARDS -- measured, and not what you would guess
  --triton-to-unstructured does NOT reject the tensor-of-pointers bitcast; it
  leaves it alone (02-unstructured.mlir still shows `tt.bitcast` + `tt.load`)
  and the later passes lower it elementwise through the `tptr` dialect: the
  composite output is a linalg.generic per addptr, each doing
  `tptr.ptradd`, and a final one doing `tptr.to_memref` + `memref.load`.
  Every triton-shared pass succeeds.

  `run.py --to triton-shared` still reports FAIL, but for an unrelated reason:

      error: "-":5:10: Dialect `tptr' not found for custom op
             'tptr.type_offset'

  which is the DRIVER's MLIR refusing to parse triton-shared's output -- the
  two LLVM builds have drifted.  That is a toolchain gap, not a rejection, and
  it means reference() is never called.  Everything claimed above is read out
  of 01-structured.mlir regardless.

ARITHMETIC
  None: reinterpret the input's bits as i32 and store them.  The address is
  `bitcast(in_ptr + r)` advanced by another `r` i32 slots, i.e. byte offset
  4r + 4r, i.e. element 2r of the i32 view -- which is what reference()
  computes.

rebuild-unsupported-ptr-elt-fail, read out of pass 1. The shaped twin of ptr_bitcast_scalar_base.py -- same op, opposite outcome. Lowers elementwise through tptr; run.py stops at stage 2 only because the driver's MLIR has no `tptr` dialect.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_bitcast_tensor_of_ptrs_kernel(in_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)

        # bitcast of a TENSOR of pointers: visitOperand's default arm refuses
        # to rebuild it because the element type is a pointer
        pc = tl.cast(in_ptr + r, tl.pointer_type(tl.int32))
        tl.store(o_ptr + r, tl.load(pc + r))

    return ptr_bitcast_tensor_of_ptrs_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"in": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    """in[2*i] reinterpreted as i32, because the address is
    `in_ptr + r` bitcast to i32* and then advanced by another `r` i32 slots.
    """
    import torch
    x = inputs["in"].contiguous().view(torch.int32)
    return {"o": x[0:2 * BLOCK:2].contiguous()}


SPEC = KernelSpec(
    name="ptr_bitcast_tensor_of_ptrs_kernel",
    kernel=kernel,
    signature={"in_ptr": "*fp32", "o_ptr": "*i32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("in", IN, "float32", (N,)),
          Arg("o", OUT, "int32", (BLOCK,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    # WAS `expect="refuse"`, and the refusal is gone. convert_unsupported_pointers now
    # takes the whole tensor-of-pointers family down to i64 + a gather, and this
    # file's own reference -- which was written out but never run -- matches to
    # the bit. The upstream branch ids it covers still fire; what changed is
    # that giving up is no longer the end of the pipeline for us.
)
