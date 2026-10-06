"""A carried pointer whose analysis failed OUTRIGHT -- no PtrState at all.

The companion to loop_05_gather_ptr_iterarg_refused.py. There the state existed
and was rejected for not being structured; here PtrAnalysis never produces one,
because the addptr operand is a rank-2 tensor loaded from memory and
rebuildAsUnsupportedOp declines multi-dimensional operands. Every loop recovery
path is then asked and every one of them misses.

Rank 2 is the whole trick. A rank-1 loaded index rebuilds into a gather (that is
loop_05); at rank 2 the rebuild refuses, so nothing lands in knownPtrs.

COVERS (3 ids, lib/AnalysisStructured/PtrAnalysis.cpp)

  loopinit-fail                     :1571 (HEAD :1562)
      getLoopInitArgPtrState tried all three recovery paths -- defined by a
      GetStructuredStateOp, defined by another scf::ForOp, an outer loop's
      iter-arg already in knownPtrs -- and none matched, so it returns
      failure(). loop_07 and loop_08 are the files where the second and third
      of those paths DO match.

  getstate-unknown-ptr              :1705 (HEAD :1696)
      rewriteGetStructuredStateOp with !knownPtrs.contains(tritonValue). Same
      RAUW-and-leave-it behaviour as loop_05's getstate-not-structured but a
      different cause, and the surviving op here is rank 2, so the segment
      sizes differ:
          %structured, %offsets:2, %strides:2 = "tts.get_structured_state"(%13)
            <{resultSegmentSizes = array<i32: 1, 2, 2>}>
            : (tensor<8x2x!tt.ptr<f32>>)
              -> (tensor<8x2x!tt.ptr<f32>>, index, index, index, index)
      and the scf.for grew four dangling index iter-args it never uses:
          %14:6 = scf.for ... iter_args(%arg4 = %13, %arg5 = %offsets#0,
                    %arg6 = %offsets#1, %arg7 = %strides#0, %arg8 = %strides#1,
                    %arg9 = %cst) -> (...)

  visitoperand-blockarg-unknown-fail :1370 (HEAD :1365)
      The silent arm: an operand with no defining op that knownPtrs does not
      contain -> failure(), no remark, no dbgs. Measured on HEAD :1367, the
      `return failure();` itself rather than the else-if the census names: 2
      hits here. loop_17 is the file that takes the OTHER side of this
      if/else, and is the only file in the set that does.

REMARKS ARE EXPECTED. --triton-to-structured prints
    PtrAnalysis: encountered addptr operand produced by an unsupported
    operation: %11 = "tts.load"(%9) ...
    warning: Rewriting GetStructuredStateOp failed.   (py:24 and py:26)
That refusal IS the observable. The module still lowers to memref -- the loads
simply stay unstructured.

VERIFIED
  loopinit-fail 2, getstate-unknown-ptr 4, and HEAD :1367 = 2 under a direct
  breakpoint.

COMPILE-TIME PROBE, not numerically runnable: make_inputs supplies an all-zero
index tensor so the addresses stay in bounds, but the point of the file is which
path the compiler takes, not the values.

POSITIVE lowering, NEGATIVE analysis: the carried pointer never enters knownPtrs at all, which is a different rejection from loop_05's 'present but unstructured'.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
K = 4
R0 = 8
R1 = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop06_failed_ptr_kernel(x_ptr, idx_ptr, o_ptr, K: tl.constexpr,
                                 R0: tl.constexpr, R1: tl.constexpr):
        two_d = tl.arange(0, R0)[:, None] * R1 + tl.arange(0, R1)[None, :]
        i2d = tl.load(idx_ptr + two_d)      # rank-2 loaded index
        pb = x_ptr + i2d
        acc = tl.zeros((R0, R1), dtype=tl.float32)
        for k in range(0, K):
            acc += tl.load(pb)
            pb = pb + 1
        tl.store(o_ptr + two_d, acc)

    return loop06_failed_ptr_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "idx": torch.zeros(R0 * R1, dtype=torch.int32)}


def reference(inputs):
    import torch
    x, idx = inputs["x"], inputs["idx"].long()
    acc = torch.zeros(R0 * R1, dtype=torch.float32)
    for k in range(K):
        acc += x[idx + k]
    return {"o": acc}


SPEC = KernelSpec(
    name="loop06_failed_ptr_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "idx_ptr": "*i32", "o_ptr": "*fp32",
               "K": "constexpr", "R0": "constexpr", "R1": "constexpr"},
    constexprs={"K": K, "R0": R0, "R1": R1},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("idx", IN, "int32", (R0 * R1,)),
        Arg("o", OUT, "float32", (R0 * R1,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
