"""Two sibling loops: the second one's iter-arg is initialised from the first one's RESULT.

getLoopInitArgPtrState's second recovery path. loop_07 covers the third (init
arg is an enclosing loop's iter-arg); this covers the one where the init arg is
defined by another scf::ForOp, which makes getLoopInitArgPtrState recurse into
getLoopResultPtrState on that other loop.

The first inner loop only advances the pointer and loads nothing. That is
deliberate: if it loaded, the canonicalizer would have grounds to fuse or
simplify, and the two-loop shape -- which is the entire content of the branch --
could disappear.

COVERS (1 id, lib/AnalysisStructured/PtrAnalysis.cpp)

  loopinit-from-nested-for          :1552 (HEAD :1543)
      Init arg defined by an scf::ForOp -> recurse into
      getLoopResultPtrState(forOp, index). Proof, from 01-structured.mlir --
      the second loop's iter_args are the first loop's results %9#0/%9#1/%9#2,
      not pre-loop constants:
          %9:3 = scf.for %arg7 = ... iter_args(%arg8 = %arg3, %arg9 = %arg4,
                                               %arg10 = %arg5)
                     -> (tensor<128x!tt.ptr<f32>>, index, index)  : i32 {
            ...advance only...
          }
          %10:4 = scf.for %arg7 = ... iter_args(%arg8 = %9#0, %arg9 = %9#1,
                                                %arg10 = %9#2, %arg11 = %arg6) {
            %11 = tts.make_tptr %arg0 to sizes: [128], strides: [%arg10],
                                offsets: [%arg9], ...
      Note that %9#1 and %9#2 are offset and stride -- the recursion recovered
      the whole state through a loop boundary, not just the pointer.

VERIFIED
  loopinit-from-nested-for: 4 hits (once per outer trip per sibling).

RUNNABLE. Each outer trip skips two BLOCKs then reads two, so
o = x[256:384] + x[384:512] + x[768:896] + x[896:1024].

POSITIVE. Two sibling loops: the second's init arg is defined by the first scf.for, forcing getLoopInitArgPtrState to recurse.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
OUTER = 2
INNER = 2
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop08_sibling_loops_kernel(x_ptr, o_ptr, OUTER: tl.constexpr,
                                    INNER: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        p = x_ptr + r
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for a in range(0, OUTER):
            for b in range(0, INNER):       # loop A: only advances the pointer
                p = p + BLOCK
            for b in range(0, INNER):       # loop B: inits from loop A's RESULT
                acc += tl.load(p)
                p = p + BLOCK
        tl.store(o_ptr + r, acc)

    return loop08_sibling_loops_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    acc = torch.zeros(BLOCK, dtype=torch.float32)
    k = 0
    for a in range(OUTER):
        k += INNER                          # loop A skips INNER blocks
        for b in range(INNER):
            acc += x[k * BLOCK:(k + 1) * BLOCK]
            k += 1
    return {"o": acc}


SPEC = KernelSpec(
    name="loop08_sibling_loops_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "OUTER": "constexpr",
               "INNER": "constexpr", "BLOCK": "constexpr"},
    constexprs={"OUTER": OUTER, "INNER": INNER, "BLOCK": BLOCK},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (BLOCK,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
