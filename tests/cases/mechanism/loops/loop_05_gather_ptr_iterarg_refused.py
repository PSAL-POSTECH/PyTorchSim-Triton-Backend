"""A GATHER pointer carried through scf.for: the state exists and is REFUSED.

The distinction this file exists to draw is between "no PtrState" and "a
PtrState that is not structured". `g = x_ptr + i` with `i` loaded from memory
analyses fine -- PtrAnalysis builds a state and even emits a
tts.make_gather_scatter_tptr for the pre-loop use -- but that state fails
isStructured(), and every loop entry point refuses it. loop_06 is the other
case, where analysis fails outright and there is no state at all.

The file lowers to the end. It is not a negative kernel: PtrAnalysis declines
the loop, prints two warnings, and TritonToUnstructured then handles the loop
its own way, which is where the last two branches come from.

COVERS (6 ids)

lib/AnalysisStructured/PtrAnalysis.cpp

  loopstate-unstructured-fail       :1604 (HEAD :1595)
      getLoopIterArgPtrState found a state but !isStructured() -> refuse. This
      is the branch that separates this file from loop_06.

  forop-iterarg-state-fail          :1636 (HEAD :1627)
      getLoopIterArgPtrState failed, so rewriteForOp `continue`s WITHOUT
      failing the whole rewrite. Proof -- the loop body keeps raw triton ops
      where loop_01 has tts ops:
          %15 = tt.load %arg4 : tensor<128x!tt.ptr<f32>>
          %17 = tt.addptr %arg4, %cst_0 : tensor<128x!tt.ptr<f32>>, ...
      and there is no tts.make_tptr anywhere inside the region.

  getstate-not-structured           :1713 (HEAD :1704)
      knownPtrs HAS the value but the state is not structured, so the
      GetStructuredStateOp is NOT erased -- result 0 is RAUW'd to the original
      triton value and the offset/stride results dangle. This is the one place
      in the set where you can read the op's literal text:
          %structured, %offsets, %strides = "tts.get_structured_state"(%7)
            <{resultSegmentSizes = array<i32: 1, 1, 1>}>
            : (tensor<128x!tt.ptr<f32>>)
              -> (tensor<128x!tt.ptr<f32>>, index, index)
      Accompanied by the warning, twice, at py:23 and py:25:
          warning: Rewriting GetStructuredStateOp failed.

  visitforop-loopresult-fail        :1254 (HEAD :1250 cond / :1254 stmt)
      getLoopResultPtrState failed for a post-loop use of the refused result.
      THE `+ r` IN `tl.load(g + r)` IS LOAD-BEARING and there is a comment on it
      in the body: only an addptr whose base is the scf.for result routes
      visitOperandAddptr into visitOperandForOp. With a bare `tl.load(g)`,
      measured, HEAD :1254 gets 0 hits and this branch is silently lost --
      which is exactly what an earlier version of this split did.
      Measured now: HEAD :1254 = 1 hit here, 0 for every other loop_* file,
      and 1 for the original ptr_loops.py, so the count is preserved exactly.
      NOTE the census line :1254 -> HEAD :1250 is the `if (failed(newState))`
      CONDITION, which runs on every call; the branch census therefore
      also credits this id to loop_01. The real arm is only taken here.

lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp

  t2u-scf-for-iterarg-retype        :414
      Case<scf::ForOp>: the iter-arg and the tied result are MUTATED IN PLACE
      to the offset type. This file is the census's trigger verbatim. Proof,
      from 02-unstructured.mlir -- the iter-arg that was
      tensor<128x!tt.ptr<f32>> in 01-structured.mlir is now an offset tensor and
      the base has been hoisted out of the loop into the gather:
          %3:2 = scf.for ... iter_args(%arg4 = %2, %arg5 = %cst)
                     -> (tensor<128xi32>, tensor<128xf32>)  : i32 {
            %8 = tts.gather %arg0[%arg4] : (<f32>, tensor<128xi32>)
                                        -> tensor<128xf32>

  t2u-scf-yield-noop                :460
      Case<scf::YieldOp> returns success() and does nothing; the operand type
      is fixed up implicitly by the RAUW loop at :567. Proof, same region:
          scf.yield %10, %9 : tensor<128xi32>, tensor<128xf32>
      -- the yield now carries tensor<128xi32> although no pattern rewrote it.
      The census names this same loop as its trigger, so it is claimed here.

VERIFIED
      tests/cases/loop_05_gather_ptr_iterarg_refused.py
  All six hit. visitforop-loopresult-fail additionally confirmed with a
  breakpoint on HEAD :1254 alone (1 hit).

RUNNABLE. o = sum over k of x[idx + k*128] + x[idx + 512 + arange(128)].

POSITIVE lowering, NEGATIVE analysis: PtrAnalysis refuses the carried gather pointer (two warnings) and TritonToUnstructured retypes the scf.for iter-arg to an offset tensor instead.
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
    def loop05_gather_ptr_kernel(x_ptr, idx_ptr, o_ptr, K: tl.constexpr,
                                 BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        i = tl.load(idx_ptr + r)
        g = x_ptr + i                       # gather: state exists, not structured
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(0, K):
            acc += tl.load(g)
            g = g + BLOCK
        # The post-loop use must be `g + r`, NOT a bare `tl.load(g)`. Only an
        # addptr whose BASE is the scf.for result sends visitOperandAddptr into
        # visitOperandForOp; a bare load on the result never calls it, and the
        # visitforop-loopresult-fail arm is then never executed. Measured: with
        # `tl.load(g)` PtrAnalysis.cpp:1254 `return failure()` gets 0 hits.
        acc += tl.load(g + r)               # post-loop use of the refused result
        tl.store(o_ptr + r, acc)

    return loop05_gather_ptr_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "idx": torch.zeros(BLOCK, dtype=torch.int32)}


def reference(inputs):
    import torch
    x, idx = inputs["x"], inputs["idx"].long()
    acc = torch.zeros(BLOCK, dtype=torch.float32)
    for k in range(K):
        acc += x[idx + k * BLOCK]
    acc += x[idx + K * BLOCK + torch.arange(BLOCK)]
    return {"o": acc}


SPEC = KernelSpec(
    name="loop05_gather_ptr_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "idx_ptr": "*i32", "o_ptr": "*fp32",
               "K": "constexpr", "BLOCK": "constexpr"},
    constexprs={"K": K, "BLOCK": BLOCK},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("idx", IN, "int32", (BLOCK,)),
        Arg("o", OUT, "float32", (BLOCK,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
