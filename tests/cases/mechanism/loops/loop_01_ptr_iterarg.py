"""A structured tensor-of-pointers carried through scf.for, then read after it.

This is the loop census's happy path, and the file every other loop_* file is a
deviation from. `p = x_ptr + r` is affine, so PtrAnalysis recovers a full
PtrState for it, threads that state through the loop as extra iter-args, and
rebuilds the pointer at the top of the body.

Line numbers below are the WORKTREE numbers the census records; where a
measurement is quoted the HEAD number gdb actually broke on is given too, since
triton-shared-opt is built from HEAD and the worktree has uncommitted edits.
"PtrAnalysis.cpp" always means lib/AnalysisStructured/PtrAnalysis.cpp -- there is
a second, unrelated PtrAnalysis.cpp under lib/Analysis/.

COVERS (9 ids, all in lib/AnalysisStructured/PtrAnalysis.cpp unless noted)

  maybestructured-forop-neighbors   :1875 (HEAD :1864)
      initializeMaybeStructuredArgs' BFS sees an scf::ForOp user of `p` and adds
      BOTH the region iter-arg and the tied loop result to maybeStructuredArgs.
      Without this the arg is skipped at :1632 and nothing below happens.

  forop-ptr-make-tptr               :1679 (HEAD :1670)
      isPointerType(arg) && rank != 0 -> a tts.make_tptr is inserted as the
      FIRST op of the loop body. Proof, from 01-structured.mlir:
          %4:4 = scf.for ... iter_args(%arg3 = %2, %arg4 = %c0, %arg5 = %c1,
                                       %arg6 = %cst) -> (...)  : i32 {
            %13 = tts.make_tptr %arg0 to sizes: [128], strides: [%arg5],
                                offsets: [%arg4], ...
      -- make_tptr is literally the first statement inside the region, and its
      offset/stride operands are the loop's own block arguments.

  loopinit-from-getstate            :1527 (HEAD :1518)
      getLoopInitArgPtrState found the init arg defined by a
      tts::GetStructuredStateOp whose operand is in knownPtrs (the pre-loop
      `tt.addptr %1, %0`), so the state came back without recursion.

  reconcile-offsets-strides         :1585 (HEAD :1576)
      rank > 0, so every offset and then every stride is replaced by a loop
      value. Witness is the scf.for TYPE: `-> (tensor<128x!tt.ptr<f32>>, index,
      index, tensor<128xf32>)` -- 2*rank == 2 extra index iter-args
      (%arg4 = offset, %arg5 = stride) that the source never wrote.

  getstate-offsets-strides-const    :1740 (HEAD :1731)
      The offset (0) and stride (1) are int attrs, so arith.constants are
      materialized just before the loop and fed in:
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
      loop_02 is the same kernel with those two positions turned into SSA
      values, which takes the OTHER arm (:1744).

  getstate-success-erase            :1761 (HEAD :1752)
      rewriteGetStructuredStateOp's tail RAUWs and erases the op. Witness is an
      absence with a shape: grep 01-structured.mlir for get_structured_state and
      there is none, while the scf.for signature grew by exactly (index, index).
      Contrast loop_05/loop_06/loop_13/loop_16, where the op SURVIVES because
      the rewrite failed.

  visitforop-success                :1261 (HEAD :1257)
      `state = newState.value()` -- the post-loop `tl.load(p + r)` resolves
      through the scf.for RESULT. Witness:
          %5 = arith.addi %4#2, %c1_1 : index
          %6 = tts.make_tptr %arg0 to sizes: [128], strides: [%5],
                             offsets: [%4#1], ...
      The offsets/strides are %4#1 / %4#2, results of the loop, not pre-loop
      constants. This is why the post-loop read is here and not deleted:
      without it visitOperandForOp is never called at all.

  dispatch-forop-always-skip        :2019 (HEAD :2006)
      rewriteOp's ForOp case returns WalkResult::skip() unconditionally.
      Observable as a negative: the body contains ONE tts.make_tptr per address,
      not two. Any loop reaches this; it is claimed here because this is the
      simplest loop in the set.

  tts-get-structured-state-tensor-of-ptrs
      TritonStructuredDialect.td:243, implemented at
      lib/Dialect/TritonStructured/IR/TritonStructuredOps.cpp:397-402.
      Input is tensor<128x!tt.ptr<f32>> -> (rank offsets, rank strides) = (1,1).
      gdb-measured on TritonStructuredOps.cpp:402: 5 hits for this file.
      TableGen has no code, so the census line itself cannot carry a
      breakpoint; :402 is the arm that computes the segment sizes the op is
      declared with, which is the same fact. The erased op's replacements are
      what the scf.for signature (ptr, index, index) is made of.

VERIFIED
  All 9 above hit. visitforop-loopresult-fail also shows up in that tool's
  output for this file with 1 hit -- that is a MEASUREMENT ARTIFACT, not a
  branch this file takes. Its census line :1254 maps to HEAD :1250, which is
  `if (failed(newState)) {`, the condition, evaluated on every call regardless
  of outcome. Breaking on HEAD :1254, the actual `return failure();`, gives 0
  for this file and 1 for loop_05. The failing arm belongs to loop_05.

RUNNABLE. o = x[0:128] + x[128:256] + x[256:384] + x[384:512] + x[512 + 2*j],
the last term because the post-loop `p + r` adds the lane index a second time.

POSITIVE. Structured tensor-of-pointers carried through scf.for, plus a post-loop read of the loop result.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
K = 4
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop01_ptr_iterarg_kernel(x_ptr, o_ptr, K: tl.constexpr,
                                  BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        p = x_ptr + r                                   # tensor<128x!tt.ptr>
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(0, K):
            acc += tl.load(p)
            p = p + BLOCK
        z = tl.load(p + r)                              # post-loop use
        tl.store(o_ptr + r, acc + z)

    return loop01_ptr_iterarg_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    acc = torch.zeros(BLOCK, dtype=torch.float32)
    for k in range(K):
        acc += x[k * BLOCK:(k + 1) * BLOCK]
    # post-loop: p is x_ptr + K*BLOCK + j, and `p + r` adds j again -> stride 2
    j = torch.arange(BLOCK)
    return {"o": acc + x[K * BLOCK + 2 * j]}


SPEC = KernelSpec(
    name="loop01_ptr_iterarg_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "K": "constexpr", "BLOCK": "constexpr"},
    constexprs={"K": K, "BLOCK": BLOCK},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (BLOCK,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
