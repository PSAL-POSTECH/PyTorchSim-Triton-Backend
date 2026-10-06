"""A gather whose address ALSO uses the loop's structured iter-arg.

THIS IS THE 17TH FILE FOR A 16-LOOP SOURCE, and it is here because it closes a
gap the original ptr_loops.py claimed and did not have. It reaches
visitoperand-blockarg-known, which the monolithic file never executed once.

COVERS (1 id, lib/AnalysisStructured/PtrAnalysis.cpp)

  visitoperand-blockarg-known       :1378 (HEAD :1373)
      `state = knownPtrs[operand];` -- an operand with no defining op that IS in
      knownPtrs, populated during the enclosing rewriteForOp.

WHY NESTING WAS THE WRONG GUESS.  The census comment on this branch says the
operand "must be an iter-arg of an inner-loop in a multiple-level nested loop",
and both tests/cases/harness/gap_fill.py section 1 and loop_07 were written to that
description. Neither reaches it. Measured, with a breakpoint on HEAD :1373
alone:

    ptr_loops.py (the 16-loop original)   :1367 hit 6,  :1373 hit 0
    gap_fill.py                           :1367 hit 0,  :1373 hit 0
    loop_07 / loop_09 (both nested)       :1367 hit 0,  :1373 hit 0
    THIS FILE                             :1367 hit 0,  :1373 hit 1

(:1367 is the `return failure();` of the same if/else, so the middle two rows
say something stronger than "took the other arm": those kernels never enter the
block-argument arm at all.)

the branch census credited loop 9's shape with this branch, quoting the
inner loop's `tt.addptr %22, %arg12` as the observable. That was read off the
IR, not measured; loop_09's :1373 count is 0. tests/cases/harness/gap_fill.py's
docstring already recorded the correction for the monolithic file.

What actually matters is not nesting but WHO ASKS. When the addptr's operand is
the block argument itself, visitOperand is reached through the pointer-typed
path and the block-arg arm is never the one that resolves it. The block-arg arm
only runs when something else has to visit the raw integer operand -- and that
happens when a MIXED expression forces a re-visit. Here `offs + gidx` puts a
structured loop iter-arg and an unstructured loaded index into one arith.addi;
visitOperandAdd visits each side separately, and `offs` arrives as a bare block
argument that rewriteForOp already put in knownPtrs.

PROOF, from 01-structured.mlir. Inside the loop body the gather offset is built
from BOTH sources, and the structured half is made of the loop's own index
block arguments -- which only exist in that expression because knownPtrs was
read at :1373:

    %6:4 = scf.for %arg3 = ... iter_args(%arg4 = %0, %arg5 = %c0, %arg6 = %c1,
                                         %arg7 = %cst)
               -> (tensor<128xi32>, index, index, tensor<128xf32>)  : i32 {
      %11 = arith.muli %arg5, %arg6 : index          // <- knownPtrs[%arg4]
      %14 = arith.muli %4, %13 : tensor<128xi32>     // <- the loaded index
      %17 = arith.addi %16, %14 : tensor<128xi32>
      %18 = tts.make_gather_scatter_tptr %arg0 to sizes: [128]
              gather_scatter_dim: 0 gather_scatter_offset: %17,
              strides: [1], offsets: [0]
              : tensor<128xi32>  <f32> to !tt.ptr<tensor<128xf32>>
      %20 = "tts.load"(%18) ... : (!tt.ptr<tensor<128xf32>>) -> tensor<128xf32>

%arg5 and %arg6 are the offset and stride iter-args of the loop. A gather whose
offset is a function of the enclosing loop's recovered PtrState is the
observable; without the knownPtrs hit the whole chain would have taken the
silent failure at :1367 instead and %arg4 would have been folded away.

CONSEQUENCE FOR gap_fill.py: its docstring lists visitoperand-blockarg-known as
"AND ONE THIS KERNEL STILL DOES NOT REACH". That is still true of gap_fill.py,
but the branch is no longer uncovered by the set. If the branch census
is re-run over everything, this is the file that supplies it.

VERIFIED
      tests/cases/loop_17_gather_uses_loop_blockarg.py
  visitoperand-blockarg-known: hit. Confirmed independently with a breakpoint on
  HEAD :1373 alone (1 hit), not via the tool's own-line-or-next-line window --
  which matters here, because the neighbouring :1365/:1366 lines belong to the
  opposite arm and would otherwise credit this id to files that only take the
  failure path.

RUNNABLE. o = sum over k of x[arange(128) + k*128 + idx].

POSITIVE. A gather whose address also consumes the loop's structured iter-arg, which is what forces visitOperand's unstructured re-visit to look a block argument up in knownPtrs.
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
    def loop17_gather_blockarg_kernel(x_ptr, idx_ptr, o_ptr, K: tl.constexpr,
                                      BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        gidx = tl.load(idx_ptr + r)         # unstructured: comes out of memory
        offs = tl.arange(0, BLOCK)          # structured: the loop's iter-arg
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(0, K):
            acc += tl.load(x_ptr + (offs + gidx))
            offs = offs + BLOCK
        tl.store(o_ptr + r, acc)

    return loop17_gather_blockarg_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "idx": torch.zeros(BLOCK, dtype=torch.int32)}


def reference(inputs):
    import torch
    x, idx = inputs["x"], inputs["idx"].long()
    r = torch.arange(BLOCK)
    acc = torch.zeros(BLOCK, dtype=torch.float32)
    for k in range(K):
        acc += x[r + k * BLOCK + idx]
    return {"o": acc}


SPEC = KernelSpec(
    name="loop17_gather_blockarg_kernel",
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
