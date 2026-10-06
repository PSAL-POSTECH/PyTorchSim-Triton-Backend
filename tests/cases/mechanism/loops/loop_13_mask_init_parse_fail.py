"""The mask index comes out of MEMORY: the prepass wraps it, both rewrites fail.

parseLoopIterArg rejection #1 of 4. The init value is a tt.load, so
tts.get_structured_state IS inserted around it (the prepass only looks at the
type) but nothing downstream can make sense of it. Two different passes then
fail on the same value, which is why both branches are claimed here: they are
the same construct seen from PtrAnalysis and from MaskAnalysis.

COVERS (2 ids)

lib/AnalysisStructured/PtrAnalysis.cpp

  rewriteop-getstate-populate-fail  :2042 (HEAD :2029)
      rewriteOp's GetStructuredStateOp case with visitOperand failed (or an
      unstructured state), so knownPtrs is NOT populated -- which is what makes
      the later rewriteGetStructuredStateOp take its unknown-ptr arm. This file
      is the census's trigger verbatim: an index loaded from memory, then
      `idx = idx + 1` inside the loop. loop_04 is the same construct with an
      arange init and takes the succeeding arm (:2037).

lib/Analysis/MaskAnalysis.cpp

  loopiter-init-parse-fail          :672 (HEAD :672)
      parseLoopIterArg moves its insertion point BEFORE the forOp and tries to
      parse the pre-loop triton value. That parse fails on a tt.load, so the
      whole mask parse fails and anything already materialised is left behind.

PROOF, from 01-structured.mlir. The get_structured_state survives on BOTH the
init and the yielded value, and the loop body keeps raw masked triton ops
instead of tts.load/tts.store:

    %4 = "tts.load"(%2) ... : (tensor<128x!tt.ptr<i32>>) -> tensor<128xi32>
    %structured, %offsets, %strides = "tts.get_structured_state"(%4)
      <{resultSegmentSizes = array<i32: 1, 1, 1>}>
      : (tensor<128xi32>) -> (tensor<128xi32>, index, index)
    %12:3 = scf.for ... iter_args(%arg6 = %4, %arg7 = %offsets,
                                  %arg8 = %strides) {
      %13 = arith.cmpi slt, %arg6, %5 : tensor<128xi32>
      %14 = tt.load %8, %13, %cst_0 : tensor<128x!tt.ptr<f32>>
      tt.store %11, %14, %13 : tensor<128x!tt.ptr<f32>>

`tt.load` with an explicit mask operand, rather than a tts.load with
static_mask_dims, is the signature of a mask MaskAnalysis could not parse.
This is also the clearest textual instance of the int-tensor
get_structured_state shape (1,1,1 on tensor<128xi32>) anywhere in the set,
since loop_04's copy is erased on success.

Accompanied, twice, by:
    warning: Rewriting GetStructuredStateOp failed.   (py:22 and py:25)

VERIFIED
      tests/cases/loop_13_mask_init_parse_fail.py
  rewriteop-getstate-populate-fail 4, loopiter-init-parse-fail 4.

COMPILE-TIME PROBE. make_inputs supplies an all-zero index so the addresses are
in bounds; the value is not the point.

POSITIVE lowering, NEGATIVE analysis: the loop-carried mask index was loaded from memory, so both PtrAnalysis's prepass and MaskAnalysis's pre-loop reparse fail on it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 128
NBOUND = 96
ITERS = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop13_mask_init_load_kernel(x_ptr, idx_ptr, o_ptr, n, iters,
                                     BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        kk = tl.load(idx_ptr + r)           # mask index comes out of MEMORY
        for _ in range(iters):
            v = tl.load(x_ptr + r, mask=kk < n, other=0.0)
            tl.store(o_ptr + r, v, mask=kk < n)
            kk += 1

    return loop13_mask_init_load_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "idx": torch.zeros(BLOCK, dtype=torch.int32)}


def reference(inputs):
    import torch
    x, idx = inputs["x"], inputs["idx"]
    o = torch.zeros(BLOCK, dtype=torch.float32)
    keep = (idx + ITERS - 1) < NBOUND       # last iteration wins
    o[keep] = x[:BLOCK][keep]
    return {"o": o}


SPEC = KernelSpec(
    name="loop13_mask_init_load_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "idx_ptr": "*i32", "o_ptr": "*fp32",
               "n": "i32", "iters": "i32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("idx", IN, "int32", (BLOCK,)),
        Arg("o", OUT, "float32", (BLOCK,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    extra={"scalar_args": [("n", "int32_t"), ("iters", "int32_t")],
           "scalar_values": {"n": NBOUND, "iters": ITERS}},
)
