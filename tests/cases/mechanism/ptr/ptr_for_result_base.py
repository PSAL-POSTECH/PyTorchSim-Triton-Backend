"""an scf.for RESULT used as a scalar pointer -- unsupported base, pass 1 bails.

Split out of the old tests/cases/ptr_reject.py (one case per file).
PtrAnalysis is SUPPOSED to reject this address; the rejection remark IS the
observable.  Do not "fix" it.  (The module as a whole still lowers -- see
"WHAT HAPPENS AFTERWARDS" below.)

COVERS  (source root /workspace/triton_shared/, worktree lines)

  visitoperand-ptr-unsupported-fail  lib/AnalysisStructured/PtrAnalysis.cpp:1334
      The default arm of visitOperand's scalar-!tt.ptr provenance chain: the
      defining op is none of AddPtr / Bitcast / IntToPtr / MakeTensorPtr /
      scf.If.  Here it is an scf.ForOp result.
      Observable: remark "PtrAnalysis: Unexpected defining op for triton
      pointer operand" and the tt.addptr chain survives unlowered.

  visitaddptr-ptr-visit-fail         lib/AnalysisStructured/PtrAnalysis.cpp:1124
      visitOperandAddptr bails because visiting the PTR operand failed --
      BEFORE it ever visits the OFFSET operand.  This is the branch the
      multiplier trick below exists to prove.

THE UNIQUE MULTIPLIER 37 -- keep it
  :1124 (ptr visit failed) and its sibling visitaddptr-offset-visit-fail
  (:1128, offset visit failed) both end in the same failure() and leave the
  same tt.addptr behind, so the IR alone cannot tell them apart.  What CAN
  tell them apart is what the offset walk leaves behind: visiting the offset
  materialises its constants as `index`-typed values.  The offset here is
  `r * 37`, and 37 appears nowhere else in the kernel, so:

      $ grep -c 'arith.constant 37 : index' 01-structured.mlir
      0
      $ grep -n 37 01-structured.mlir
      5:    %cst = arith.constant dense<37> : tensor<128xi32>

  MEASURED, both lines above.  The one 37 that exists is the frontend's
  tensor splat; the `: index` form -- which is what visitOperand would have
  emitted had it walked the offset -- is absent.  That positively proves the
  failure came from the PTR side (:1124), not the offset side.  Contrast the
  scalar-pointer state the ptr walk DID build before giving up, which is
  visible as `%2 = arith.index_cast %1 : i32 to index` for `pid`.
  Change 37 to a value used elsewhere and the proof silently evaporates.

  The exact remarks, from `tools/lower_steps.py --debug`:

      :79:22: remark: Unexpected defining op for triton pointer operand
      :81:37: remark: PtrAnalysis: Failed to rewrite AddPtrOp
      :81:32: remark: PtrAnalysis: pointer is not replace with tts.make_tptr
                      so loadOp cannot be rewritten
      :81:32: remark: PtrAnalysis: Failed to rewrite LoadOp

  and 01-structured.mlir has NO tts.make_tptr for this address -- just the
  raw `%11 = tt.addptr %10, %9` and `%12 = tt.load %11` left standing, while
  the output store next to them did get its make_tptr.

WHY IT IS INTERESTING
  Carrying a POINTER (not an offset) across an scf.for is the natural way to
  write a streaming kernel, and the linalg lowering's prepass DOES rewrite this
  loop's iter_args -- 01-structured.mlir shows it already carrying a second
  `index` iter-arg next to the pointer:

      %5:2 = scf.for %arg2 = %c0_i32 to %c4_i32 step %c1_i32
               iter_args(%arg3 = %4, %arg4 = %2) -> (!tt.ptr<f32>, index)

  and yet the pointer still arrives at visitOperand typed as an scf.for
  RESULT, which is on none of the five accepted provenance arms.  The prepass
  and the provenance walk disagree about the same loop; that is the case.

WHAT HAPPENS AFTERWARDS -- the module is NOT killed
  Only the STRUCTURED analysis gives up.  --triton-to-unstructured then picks
  the address up as a gather:

      %8 = tts.gather %arg0[%7] : (<f32>, tensor<128xi32>) -> tensor<128xf32>

  so `run.py --to triton-compile` reports ALL STAGES PASSED.  This file is a
  witness for two pass-1 branches, not a pipeline failure.

ARITHMETIC
  Deliberately trivial: bump a pointer by 1 four times, then read a strided
  window, in[STEPS + 37*r].  N = 8192 keeps 4 + 37*127 = 4703 in bounds, so
  reference() below is the real math and not a stub.  (Nothing calls it
  anyway: stage 3, pytorchsim_triton_compiler/lowering/passes/, is where the coverage set stops.)

visitoperand-ptr-unsupported-fail + visitaddptr-ptr-visit-fail, both read out of pass 1. The absence of `arith.constant 37 : index` is what distinguishes the ptr-visit failure from the offset-visit failure. The module still lowers, as a tts.gather.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 8192
BLOCK = 128
STEPS = 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_for_result_base_kernel(in_ptr, o_ptr, BLOCK: tl.constexpr,
                                   STEPS: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, BLOCK)

        # scf.for result used as a scalar pointer -> unsupported defining op.
        # The offset carries a UNIQUE multiplier (37) so that the absence of
        # `arith.constant 37 : index` in 01-structured.mlir positively proves
        # visitOperandAddptr bailed on the PTR operand before ever visiting
        # the OFFSET operand (visitaddptr-ptr-visit-fail, not -offset-visit-fail).
        qf = in_ptr + pid
        for k in range(0, STEPS):
            qf = qf + 1
        tl.store(o_ptr + r, tl.load(qf + r * 37))

    return ptr_for_result_base_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"in": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    """The real math: in[pid + STEPS + 37 * arange(BLOCK)], pid = 0.

    PtrAnalysis rejects the STRUCTURED form of this address, but
    --triton-to-unstructured rebuilds it as a tts.gather, so the arithmetic
    below is what the emitted code computes.
    """
    import torch
    x = inputs["in"]
    idx = STEPS + 37 * torch.arange(BLOCK)
    return {"o": x[idx]}


SPEC = KernelSpec(
    name="ptr_for_result_base_kernel",
    kernel=kernel,
    signature={"in_ptr": "*fp32", "o_ptr": "*fp32",
               "BLOCK": "constexpr", "STEPS": "constexpr"},
    constexprs={"BLOCK": BLOCK, "STEPS": STEPS},
    args=[Arg("in", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
