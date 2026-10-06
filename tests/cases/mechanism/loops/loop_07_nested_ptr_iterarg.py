"""Nested loops carrying a structured pointer: the inner loop inherits from the outer.

getLoopInitArgPtrState has three ways to recover an iter-arg's state. loop_01
shows the first (defined by a tts.get_structured_state). This file shows the
third: the inner loop's init arg IS the outer loop's region iter-arg, which is
already in knownPtrs because the outer rewriteForOp put it there. The recursion
is what makes nesting work at all.

COVERS (1 id, lib/AnalysisStructured/PtrAnalysis.cpp)

  loopinit-from-knownptrs-iterarg   :1566 (HEAD :1557)
      The init arg has no defining op (assert(!ptr.getDefiningOp())) and is
      found in knownPtrs. Proof, from 01-structured.mlir -- the INNER loop's
      make_tptr is built from the inner loop's own block arguments %arg9/%arg10,
      which were seeded from the OUTER loop's %arg4/%arg5:
          %4:4 = scf.for %arg2 = ... iter_args(%arg3 = %2, %arg4 = %c0,
                                               %arg5 = %c1, %arg6 = %cst) {
            %9:4 = scf.for %arg7 = ... iter_args(%arg8 = %arg3, %arg9 = %arg4,
                                                 %arg10 = %arg5,
                                                 %arg11 = %arg6) {
              %10 = tts.make_tptr %arg0 to sizes: [128], strides: [%arg10],
                                  offsets: [%arg9], ...
      Both loops carry the full (ptr, offset, stride) triple, and the inner
      one's init list is literally the outer one's block arguments. That is the
      branch.

  loop_08_sibling_loops.py is the SECOND recovery path (init arg defined by
  another scf::ForOp) and loop_09_nested_offset_tensor.py is this same nesting
  with an integer offset carried instead of the pointer.

VERIFIED
  loopinit-from-knownptrs-iterarg: 2 hits.

HISTORICAL NOTE worth keeping: tests/cases/harness/gap_fill.py's section 1 is this
exact shape, written to reach visitoperand-blockarg-known (PtrAnalysis.cpp:1378,
HEAD :1373). It does not, and neither does this file -- measured 0 hits on HEAD
:1373 for both. Nesting alone is not enough; see loop_17 for what is.

RUNNABLE. o = x[0:128] + x[128:256] + x[256:384] + x[384:512], i.e. the same
answer as loop_01 through a 2x2 nest instead of one 4-trip loop.

POSITIVE. The INNER loop's init arg is the OUTER loop's iter-arg, which is the only way getLoopInitArgPtrState reaches its third recovery path.
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
    def loop07_nested_ptr_kernel(x_ptr, o_ptr, OUTER: tl.constexpr,
                                 INNER: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        p = x_ptr + r
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for a in range(0, OUTER):
            for b in range(0, INNER):
                acc += tl.load(p)
                p = p + BLOCK
        tl.store(o_ptr + r, acc)

    return loop07_nested_ptr_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    acc = torch.zeros(BLOCK, dtype=torch.float32)
    for k in range(OUTER * INNER):
        acc += x[k * BLOCK:(k + 1) * BLOCK]
    return {"o": acc}


SPEC = KernelSpec(
    name="loop07_nested_ptr_kernel",
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
