"""The loop carries an i32 OFFSET TENSOR; no pointer crosses the boundary.

`offs` is a tensor<128xi32> and the pointer is rebuilt inside the body from
`x_ptr + offs`. The prepass still has to track it -- integers are what pointer
arithmetic is made of -- so tts.get_structured_state accepts the I32Tensor
alternative of its operand type and rewriteForOp takes its non-pointer arm.

COVERS (4 ids; PtrAnalysis.cpp = lib/AnalysisStructured/PtrAnalysis.cpp)

  maybestructured-generic-user      :1893 (HEAD :1882)
      The BFS reaches `offs` through a NON-for user -- the `arith.addi %arg3,
      %cst_0` that advances it -- and only enqueues results whose type equals
      the value's. It does here (tensor<128xi32> -> tensor<128xi32>), so the
      chain propagates. The census's warning is what happens when it does not:
      an arith.extsi i32->i64 in this position breaks the chain and the
      downstream iter-arg is silently skipped. This file is the census's own
      trigger for the branch, verbatim.

  forop-nonptr-index-tensor         :1679 (HEAD :1670)
      !isPointerType(arg): knownPtrs[arg] is populated so later addptrs in the
      body analyse, but no tts.make_tptr is created for the index tensor
      itself. Proof -- the iter-arg keeps its integer type and the make_tptr in
      the body is for the LOAD, built from the offset/stride iter-args rather
      than from %arg3:
          %2:4 = scf.for ... iter_args(%arg3 = %0, %arg4 = %c0, %arg5 = %c1,
                                       %arg6 = %cst)
                     -> (tensor<128xi32>, index, index, tensor<128xf32>)  : i32 {
            %6 = tts.make_tptr %arg0 to sizes: [128], strides: [%arg5],
                               offsets: [%arg4], ...
      Contrast loop_01, whose first iter-arg is tensor<128x!tt.ptr<f32>>.

  rewriteop-getstate-populate       :2037 (HEAD :2024)
      rewriteOp's GetStructuredStateOp case: the value was not in knownPtrs,
      visitOperand succeeded and the state is structured, so knownPtrs is
      populated. This is what materialises the `%11 = arith.addi %arg4, %c128`
      that updates the offset -- arithmetic that exists only because this arm
      ran. loop_13 is the same construct with the init loaded from memory,
      which takes the failing arm (:2042) instead.

  tts-get-structured-state-int-tensor
      TritonStructuredDialect.td:243, implemented at
      lib/Dialect/TritonStructured/IR/TritonStructuredOps.cpp:390-396: the
      operand is a RankedTensorType whose element type isIntOrIndex, so
      offsets = strides = rank. gdb on TritonStructuredOps.cpp:396 gives 5 hits
      for this file. TableGen carries no code, so the census's own line cannot
      be broken on; :396 computes the segment sizes the op is declared with.
      The op is erased here (the rewrite succeeds), so the textual witness is
      the scf.for signature above: an i32-tensor iter-arg followed by exactly
      one offset and one stride. loop_13 and loop_16 show the same op with the
      same segment sizes UNERASED, if you want to read the literal text.

VERIFIED
      tests/cases/loop_04_offset_tensor_iterarg.py
  maybestructured-generic-user 5, forop-nonptr-index-tensor 1,
  rewriteop-getstate-populate 4; TritonStructuredOps.cpp:396 = 5 direct.

RUNNABLE. o = x[0:128] + x[128:256] + x[256:384] + x[384:512].

POSITIVE. The loop carries the i32 OFFSET tensor, never a pointer, so the prepass wraps a non-pointer value in tts.get_structured_state.
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
    def loop04_offset_tensor_kernel(x_ptr, o_ptr, K: tl.constexpr,
                                    BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        offs = tl.arange(0, BLOCK)          # i32 tensor, NOT a pointer
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(0, K):
            acc += tl.load(x_ptr + offs)
            offs = offs + BLOCK             # arith.addi keeps the type
        tl.store(o_ptr + r, acc)

    return loop04_offset_tensor_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    acc = torch.zeros(BLOCK, dtype=torch.float32)
    for k in range(K):
        acc += x[k * BLOCK:(k + 1) * BLOCK]
    return {"o": acc}


SPEC = KernelSpec(
    name="loop04_offset_tensor_kernel",
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
