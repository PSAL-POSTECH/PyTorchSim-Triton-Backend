"""PtrAnalysis coverage -- NEGATIVE: reconcileLoopPtrState rank-0 no-scalar abort.

Claim under test: reconcile-rank0-no-scalar-assert -- not in the branch census.
  lib/AnalysisStructured/PtrAnalysis.cpp, the rank-0 arm of
  `PtrAnalysis::reconcileLoopPtrState`.  Working-tree line numbers:

      1574  PtrState PtrAnalysis::reconcileLoopPtrState(
      1577    PtrState newState = state;
      1578    int cnt = iterArgIndex + 1;
      1579    if (newState.getRank() == 0) {
      1580      assert(newState.scalar);
      1583      newState.scalar = getReplacementVal(forOp, cnt);

READ THE LINE NUMBER CAREFULLY -- 1571 means two different lines.
  The critique filed this as "PtrAnalysis.cpp:1571", and the abort message
  printed by the binary does say 1571.  But the census's `loopinit-fail` entry
  (the branch census) ALSO says PtrAnalysis.cpp:1571, and it is a
  different line.  They are not the same numbering:

    * the linalg lowering's working tree has UNCOMMITTED comment-only edits (a run of
      `// 분석 완료` reading-annotations, `git -C /workspace/triton_shared
      diff lib/AnalysisStructured/PtrAnalysis.cpp`) that push this
      region ~9 lines down.  The census was written against the working tree,
      where :1571 is `return failure();` -- the last line of
      getLoopInitArgPtrState -- and the assert has moved to :1580.
    * triton-shared-opt was built from HEAD, so its `__FILE__:__LINE__` uses the
      pre-edit numbering, where the assert IS :1571 and that `return failure()`
      is :1562.

  So the number in the abort message is HEAD-relative and the number in the
  census is worktree-relative, and they collide by accident.  This kernel is
  the ASSERT, not the `return failure()`.

PRECONDITION (read off the source; this is what actually has to hold)
  `rewriteForOp` (:1636 worktree) runs getLoopIterArgPtrState ->
  getLoopInitArgPtrState -> reconcileLoopPtrState for every region iter-arg in
  `maybeStructuredArgs`.  The assert fires when that chain hands back a PtrState
  with BOTH

      getRank() == 0   (offsets and strides both empty)   AND   scalar == null

  Rank 0 alone is not enough.  ptr_loops.py section 3 (`q = x_ptr + pid`) already
  reaches this exact line at rank 0 and takes the assert-PASSES side (census
  `reconcile-rank0-scalar`), because visitOperandAddptr folded the `pid` offset
  into `state.scalar`.  The only construct that yields rank 0 with `scalar`
  unset is a pointer that never passed through a tt.addptr: visitOperand
  (:1317-1341 worktree) handles a bare `!tt.ptr<f32>` BLOCK ARGUMENT with
  `state.source = operand; return success();` and nothing else -- no offsets, no
  strides, no scalar.  That state is put into knownPtrs by the
  tts.GetStructuredStateOp arm of the rewrite walk (:2034-2040), and
  getLoopInitArgPtrState (:1526-1532) reads it straight back out through the
  tts.get_structured_state the prepass inserted in front of the loop.

  Hence the kernel below carries the BARE kernel argument `x_ptr` into the
  scf.for -- no `+ pid`, no `+ offs`, no tt.addptr before the loop.  Dropping
  that addptr is the whole trick; adding any pre-loop offset moves it to the
  already-covered arm.

  Discriminating check, measured: change the one line `p = x_ptr` to
  `p = x_ptr + tl.program_id(0)` and leave everything else alone.  That kernel
  compiles clean -- ttir ok, ttshared ok -- because visitOperandAddptr now sets
  `state.scalar` and the assert holds.  Nothing else about the loop matters.

EXPECTED RESULT: THE PIPELINE ABORTS.  Same class as mask_assert_revcmp.py and
mask_assert_expand3d.py -- the SIGABRT *is* the observable.  Do not "fix" it.
Stage 1 (TTIR) succeeds; triton-shared-opt then dies, exit 134 (SIGABRT, core
dumped) with no output file written, under --triton-to-structured alone -- the
first pass of stage 3, so nothing after it matters:

  triton-shared-opt: /workspace/triton_shared/lib/AnalysisStructured/PtrAnalysis.cpp:1571:
  mlir::tts::PtrState mlir::tts::PtrAnalysis::reconcileLoopPtrState(scf::ForOp,
  size_t, const mlir::tts::PtrState &, llvm::function_ref<Value (scf::ForOp,
  size_t)>): Assertion `newState.scalar' failed.

  #11 PtrAnalysis::reconcileLoopPtrState      PtrAnalysis.cpp:0:5   (inlined)
  #12 PtrAnalysis::getLoopIterArgPtrState     PtrAnalysis.cpp:1600
  #13 PtrAnalysis::rewriteForOp               PtrAnalysis.cpp:1628
  #23 PtrAnalysis::rewriteOp                  PtrAnalysis.cpp:1955
  #24 TritonToStructuredPass::runOnOperation  TritonToStructuredPass.cpp:323

  (those frame numbers are HEAD-relative too: :1628 is the
  `getLoopIterArgPtrState(op, i)` call in rewriteForOp, :1636 in the worktree.)

`python tools/lower_steps.py tests/cases/loop_assert_bare_arg.py --debug`
therefore prints FAILED for all six passes, and every file under
work/loop_assert_bare_arg_kernel/triton-shared-detail/ is a
`// PASS FAILED at triton-to-structured` stub holding the untouched TTIR.

The prepass IR, dumped by --debug-only=triton-ptr-analysis just before the
abort, is the crispest proof of the precondition -- the bare `%arg0` is the
init operand, and the prepass has already reserved the `index` offset slot that
the assert dies trying to fill:

  %structured, %offsets = "tts.get_structured_state"(%arg0)
      <{resultSegmentSizes = array<i32: 1, 1, 0>}> : (!tt.ptr<f32>) -> (!tt.ptr<f32>, index)
  %0:3 = scf.for %arg2 = %c0_i32 to %c4_i32 step %c1_i32
      iter_args(%arg3 = %structured, %arg4 = %offsets, %arg5 = %cst)
      -> (!tt.ptr<f32>, index, f32) : i32 {
    %1 = tt.load %arg3 : !tt.ptr<f32>
    ...
No PtrAnalysis remark is emitted before the abort -- the process just dies.

COROLLARY, worth recording: census id `getstate-rank0-kernel-arg-zero`
(rewriteGetStructuredStateOp :1730-1735 worktree, "This operand is a pointer
directly from the kernel arguments.  Use offset 0.") is UNREACHABLE, and this
abort is what is actually there instead.  That arm is guarded by
`state.getRank() == 0 && !state.scalar` -- precisely the state built here -- and
it only ever runs on a value the prepass wrapped in tts.get_structured_state,
which the prepass only does for values crossing an scf.for boundary
(TritonToStructuredPass.cpp:210-217, via the SCF structural 1->N type
conversion).  But rewriteGetStructuredStateOp is called from a SECOND walk
(TritonToStructuredPass.cpp:329) that runs only after `rewriteOp` at :323 has
returned -- and :323 never returns, because rewriteForOp aborts the process.
So the `arith.constant 0 : index` that census entry describes can never be
emitted for a bare kernel argument.

Aborts triton-shared-opt by design. Keep alone.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

N = 8
K = 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_assert_bare_arg_kernel(x_ptr, o_ptr, K: tl.constexpr):
        p = x_ptr                 # bare kernel arg: rank-0 PtrState, no scalar
        s = 0.0
        for k in range(0, K):     # p becomes an scf.for iter-arg
            s += tl.load(p)
            p = p + 1
        tl.store(o_ptr, s)

    return loop_assert_bare_arg_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    # o[0] = x[0] + x[1] + ... + x[K-1].  Unreachable -- the compiler aborts
    # before any code is produced -- but stated correctly so the file is not
    # lying about what the kernel means.
    return {"o": inputs["x"][:K].sum().reshape(1)}


SPEC = KernelSpec(
    name="loop_assert_bare_arg_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "K": "constexpr"},
    constexprs={"K": K},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (1,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    expect="open",
    expect_stop=Stop("triton-compile", Reason.TRITON_PASS_FAILED),
    expect_why="a C++ assert in PtrAnalysis: a crash, not a refusal, so ours to fix (#260)",
)
