"""SINCE triton-torchsim-sink-into-pointer-ifs THIS COMPILES AND VERIFIES: the body after the if is
moved into each branch, so the make_tptr base is the branch's own argument and PtrAnalysis never
sees the if result. The account below is of the refusal this used to pin.

an scf.if that really YIELDS a pointer -- the if result is the opaque base.

Split out of the old tests/cases/ptr_reject.py (one case per file).

COVERS  (source root /workspace/triton_shared/, worktree lines)

  visitoperand-ptr-ifop   lib/AnalysisStructured/PtrAnalysis.cpp:1330
      The scalar !tt.ptr operand's defining op is an scf::IfOp.  visitOperand
      does NOT recurse into either region; it does `state.source = operand`
      and treats the if result as an opaque base.

WHAT PROVES IT FIRED  (quoted from
  work/ptr_scf_if_base_kernel/triton-shared-detail/01-structured.mlir)

      %4:2 = scf.if %3 -> (!tt.ptr<f32>, index) {
               %12 = tt.addptr %arg1, %0 : !tt.ptr<f32>, i32
               scf.yield %12, %1 : !tt.ptr<f32>, index
             } else {
               scf.yield %arg0, %c0 : !tt.ptr<f32>, index
             }
      %9 = tts.make_tptr %4#0 to sizes: [128], strides: [1], offsets: [0],
             shape: [0], order: [] : <f32> to tensor<128x!tt.ptr<f32>>

  A base of `%arg0` or `%arg1` would mean the walk had recursed; `%4#0`, the
  if's own result, means it stopped at the if.  Note the prepass gave the
  scf.if a SECOND result of type `index` (the offset state) -- but the base
  the load uses is still result #0, opaque.

WHY THE `if` HAS TO BE WRITTEN THIS WAY
  The pointer for the taken branch is BUILT INSIDE the region (`b_ptr + pid`).
  If both branches merely selected an existing value, MLIR's
  ConvertTrivialIfToSelect canonicalisation would replace the whole scf.if
  with an arith.select before PtrAnalysis ever ran, and :1330 would never be
  reached -- the operand would be a select, which lands on the "Unexpected
  defining op" arm instead.  Keep the addptr inside the region.

TRITON-SHARED ACCEPTS THIS -- the harness is what stops, and for another reason
  Measured, not assumed: every pass of the linalg lowering succeeds.  The addptr in the
  region survives --triton-to-unstructured untouched (02-unstructured.mlir
  still has the scf.if yielding a !tt.ptr<f32>) and the composite output is
  fully lowered:

      %4 = scf.if %3 -> (!ptr.ptr<#tptr.default_memory_space>) {
             %8 = tptr.ptradd %1 %7 : ...
             scf.yield %8 : !ptr.ptr<#tptr.default_memory_space>
           } else { scf.yield %2 : ... }
      %5 = tptr.to_memref %4 : ... to memref<1xf32>

  `run.py --to triton-compile` nevertheless reports FAIL, and the message says why:

      error: "-":3:10: Dialect `tptr' not found for custom op
             'tptr.type_offset'

  That is the DRIVER's MLIR failing to parse the linalg lowering's output -- the two
  LLVM builds have drifted and the driver does not register the linalg lowering's
  `tptr` dialect.  It is a toolchain gap, not a rejection, and it hits every
  kernel whose lowering goes through the unstructured pointer path.  Nothing
  about branch :1330 depends on it; the claim is read from 01-structured.mlir.

  (tests/cases/mechanism/unstructured/unstructured_fail_addptr_in_if.py claims
  t2u-addptr-in-scf-if-fail, TritonToUnstructuredPass.cpp:324, for the shape
  that DOES trip that guard.  This one does not trip it, and does not claim
  it.)

ARITHMETIC
  None: pick one of two base pointers, read BLOCK contiguous floats.
  reference() states the intended math -- with grid=(1,) pid is 0, so the else
  arm wins and the answer is in[:BLOCK] -- but nothing calls it, because the
  harness stops at the parse above.

visitoperand-ptr-ifop: the scf.if result is used verbatim as the make_tptr base. Every pass of the linalg lowering succeeds; run.py stops at stage 2 only because the driver's MLIR has no `tptr` dialect.
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
    def ptr_scf_if_base_kernel(in_ptr, b_ptr, o_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, BLOCK)

        # scf.if that really yields a pointer (the value is built inside the
        # region, so ConvertTrivialIfToSelect cannot replace the result)
        base = in_ptr
        if pid > 0:
            base = b_ptr + pid
        tl.store(o_ptr + r, tl.load(base + r))

    return ptr_scf_if_base_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"in": torch.randn(N, generator=g, dtype=torch.float32),
            "b": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    """NEVER CALLED -- the harness cannot parse the linalg lowering's tptr output.

    The math is real though: with grid=(1,) pid is always 0, so the else arm
    wins and this is `in[:BLOCK]`.
    """
    return {"o": inputs["in"][:BLOCK]}


SPEC = KernelSpec(
    name="ptr_scf_if_base_kernel",
    kernel=kernel,
    signature={"in_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("in", IN, "float32", (N,)), Arg("b", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
