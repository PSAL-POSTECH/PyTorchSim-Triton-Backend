"""Nested loops carrying an i32 offset TENSOR, and the iter-arg that is left alone.

loop_07 is this file with the pointer carried instead of the offset; reading
them together is the point, because they lower almost identically and the
difference is one type in the scf.for signature. What this file claims, though,
is the arm that fires on the OTHER iter-arg -- the f32 accumulator, which is not
maybeStructured and is skipped entirely.

COVERS (1 id, lib/AnalysisStructured/PtrAnalysis.cpp)

  forop-arg-not-maybestructured     :1632 (HEAD :1623)
      !maybeStructuredArgs.contains(arg) -> continue. The accumulator iter-arg
      gets no knownPtrs entry and no tts.make_tptr, and the loop rewrite steps
      straight over it. Proof, from 01-structured.mlir -- %arg6 (outer) and
      %arg11 (inner) are threaded verbatim and only ever used by the addf:
          %2:4 = scf.for ... iter_args(%arg3 = %0, %arg4 = %c0, %arg5 = %c1,
                                       %arg6 = %cst)
                     -> (tensor<128xi32>, index, index, tensor<128xf32>) {
            %6:4 = scf.for ... iter_args(%arg8 = %arg3, %arg9 = %arg4,
                                         %arg10 = %arg5, %arg11 = %arg6) {
              %10 = arith.addf %arg11, %9 : tensor<128xf32>
      Contrast %arg3, whose state was recovered and which grew the two index
      companions %arg4/%arg5.

      Any loop with a float accumulator reaches this branch -- gemm.py's
      reduction loop is the census's other example. It is CLAIMED here because
      this file has four such iter-args across two nesting levels (measured 14
      hits, the most of any loop_* file) while carrying a structured offset in
      the same scf.for, so the skipped arg and the rewritten arg sit side by
      side in one signature.

WHY THE FILE EXISTS AT ALL beyond that one claim: it is the shape everyone
expected to reach visitoperand-blockarg-known (PtrAnalysis.cpp:1378, HEAD
:1373), and it does not. the branch census credited this exact loop with
that branch, quoting the inner `tt.addptr %22, %arg12` as the observable. That
was inferred from the IR. Measured with breakpoints on the arm itself:

    HEAD :1373  `state = knownPtrs[operand];`   0 hits
    HEAD :1367  `return failure();`             0 hits

Zero on BOTH sides of the if/else, i.e. visitOperand's block-argument arm is
never entered here at all -- rewriteForOp's own mapping resolves the addptr
before it can be. tests/cases/harness/gap_fill.py's docstring recorded the same
negative for the monolithic file. loop_17_gather_uses_loop_blockarg.py is the
shape that does reach it, and its docstring explains why nesting was the wrong
guess. Keeping this file runnable on its own is what makes that comparison a
one-command check rather than an argument.

VERIFIED
      tests/cases/loop_09_nested_offset_tensor.py
  forop-arg-not-maybestructured: 14 hits.

RUNNABLE. o = x[0:128] + x[128:256] + x[256:384] + x[384:512].

POSITIVE. Same nesting as loop_07 but carrying an i32 OFFSET tensor instead of a pointer -- the inner block arg is not in knownPtrs, so visitOperand fails silently.
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
    def loop09_nested_offsets_kernel(x_ptr, o_ptr, OUTER: tl.constexpr,
                                     INNER: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        offs = tl.arange(0, BLOCK)          # i32 tensor, carried through BOTH loops
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for a in range(0, OUTER):
            for b in range(0, INNER):
                acc += tl.load(x_ptr + offs)
                offs = offs + BLOCK
        tl.store(o_ptr + r, acc)

    return loop09_nested_offsets_kernel


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
    name="loop09_nested_offsets_kernel",
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
