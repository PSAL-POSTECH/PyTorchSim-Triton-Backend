"""int_to_ptr -- a pointer conjured out of an i64 loaded from memory.

Split out of the old tests/cases/ptr_reject.py (one case per file).

COVERS  (source root /workspace/triton_shared/, worktree lines)

  visitoperand-ptr-inttoptr   lib/AnalysisStructured/PtrAnalysis.cpp:1326
      The scalar !tt.ptr operand's defining op is a triton::IntToPtrOp, so
      visitOperand takes the IntToPtr arm of the provenance chain instead of
      falling through to the "Unexpected defining op" failure at :1334.

  inttoptr-source             lib/AnalysisStructured/PtrAnalysis.cpp:1265
      visitOperandIntToPtr's body: `state.source = op.getResult()`, rank stays
      0.  The int_to_ptr is treated as an opaque base, never looked through.

WHAT PROVES IT FIRED  (quoted from
  work/ptr_inttoptr_base_kernel/triton-shared-detail/01-structured.mlir)

      %1 = tt.load %arg0 : !tt.ptr<i64>
      %2 = tt.int_to_ptr %1 : i64 -> !tt.ptr<f32>
      %7 = tts.make_tptr %2 to sizes: [128], strides: [1], offsets: [0],
             shape: [0], order: [] : <f32> to tensor<128x!tt.ptr<f32>>
      %9 = "tts.load"(%7) ... -> tensor<128xf32>

  The make_tptr base is `%2`, the int_to_ptr RESULT, with rank-0 state
  (offsets: [0], strides: [1] come from the `+ r` alone).  If :1326 had not
  matched, this load would instead have produced the "Unexpected defining op
  for triton pointer operand" remark -- the arm three lines further down at
  :1334 -- and the tt.addptr chain would have survived unlowered.  It does
  not: the only remark pass 1 emits here is about the SCALAR i64 load
  `tl.load(pp_ptr)`, "pointer is not replace with tts.make_tptr so loadOp
  cannot be rewritten", which is a different op on a different line.

WHY IT IS INTERESTING
  This is the one base-provenance arm where the base is a RUNTIME VALUE read
  out of a buffer, so triton-shared has to accept an address it cannot reason
  about at all -- rank 0, no offsets, no strides, no aliasing information.

NEGATIVE FROM PASS 2 ONWARD: A C++ ASSERT, AND THAT IS THE EXPECTED OUTCOME
  Do not try to "fix" this.  Pass 1 succeeds and is where the claim is read;
  `--triton-to-unstructured` then SIGABRTs (rc=134):

      triton-shared-opt: .../mlir/lib/IR/OperationSupport.cpp:531:
        mlir::OpOperand &mlir::MutableOperandRange::operator[](unsigned int)
        const: Assertion `index < length && "index is out of bounds"' failed.
      #12 TritonToUnstructuredPass::processUnstructuredPtrs(unsigned int)
          ::'lambda0'(auto)::operator()<mlir::tts::MakeTensorPtrOp>
          .../lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp:523

  The trigger is the leftover scalar `tt.load %arg0 : !tt.ptr<i64>`: it is an
  unstructured pointer use, so processUnstructuredPtrs walks the make_tptr
  ops looking for an offset operand to rewrite, and the int_to_ptr-based one
  has none.  `run.py --to triton-shared` therefore exits non-zero and stage 2 is as
  far as this file ever gets.  (This is why the old monolithic ptr_reject.py
  never saw the assert: an earlier rejection killed the module first.)

COMPILE-ONLY BY CONSTRUCTION -- reference() is an unreachable stub
  Even without the assert the kernel dereferences whatever 64-bit number
  happens to sit in pp[0].  make_inputs zeroes that buffer, so running it
  would be a null dereference, and no torch expression can name the address.
  Use it with `run.py ... --to triton-shared` / tools/lower_steps.py only.

ARITHMETIC
  None. One load through the conjured pointer, one store.

visitoperand-ptr-inttoptr + inttoptr-source, read out of pass 1. COMPILE-ONLY: the base is an address read out of a buffer, and --triton-to-unstructured then SIGABRTs at TritonToUnstructuredPass.cpp:523 -- expected, see the docstring.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_inttoptr_base_kernel(pp_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)

        # int_to_ptr -> the tt.int_to_ptr result becomes the base
        addr = tl.load(pp_ptr)
        pd = tl.cast(addr, tl.pointer_type(tl.float32))
        tl.store(o_ptr + r, tl.load(pd + r))

    return ptr_inttoptr_base_kernel


def make_inputs(torch, seed=0):
    return {"pp": torch.zeros(4, dtype=torch.int64)}


def reference(inputs):
    """UNREACHABLE STUB -- see the module docstring.

    The kernel loads through int_to_ptr(pp[0]); pp is zeroed, so executing it
    would fault, and no torch expression can name the address anyway. This
    exists only to satisfy KernelSpec's type contract.
    """
    import torch
    return {"o": torch.zeros(BLOCK, dtype=torch.float32)}


SPEC = KernelSpec(
    name="ptr_inttoptr_base_kernel",
    kernel=kernel,
    signature={"pp_ptr": "*i64", "o_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("pp", IN, "int64", (4,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    expect="refuse",
    expect_why="a C++ assert in triton-shared from pass 2 onward, which IS the expected outcome",
)
