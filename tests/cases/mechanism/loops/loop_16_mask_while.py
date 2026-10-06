"""An scf.while, not an scf.for: the block argument's parent is the wrong op.

parseLoopIterArg rejection #4 of 4, and the only file in the whole coverage set
that emits an scf.while at all. A python `while` in a @triton.jit body lowers to
scf.while, so the mask's index tensor is a block argument of a `do` region
rather than of an scf.for, and the very first cast in parseLoopIterArg misses.

COVERS (1 id, lib/Analysis/MaskAnalysis.cpp)

  loopiter-not-forop                :609 (HEAD :609)
      The block argument's parent op is not an scf::ForOp -- a function entry
      block, an scf.if, or as here an scf.while -> failure().

PROOF, from 01-structured.mlir. The `do` region's ^bb0 block arguments are what
the mask is built from, and both the load and the store keep their raw mask
operand:

    %8:4 = scf.while (%arg4 = %0, %arg5 = %c0, %arg6 = %c1, %arg7 = %c0_i32)
             : (tensor<128xi32>, index, index, i32)
            -> (tensor<128xi32>, index, index, i32) {
      %9 = arith.cmpi slt, %arg7, %arg3 : i32
      scf.condition(%9) %arg4, %arg5, %arg6, %arg7 : ...
    } do {
    ^bb0(%arg4: tensor<128xi32>, %arg5: index, %arg6: index, %arg7: i32):
      %9 = arith.cmpi slt, %arg4, %1 : tensor<128xi32>
      %10 = tt.load %4, %9, %cst_0 : tensor<128x!tt.ptr<f32>>
      tt.store %7, %10, %9 : tensor<128x!tt.ptr<f32>>
      %11 = arith.addi %arg4, %cst : tensor<128xi32>
      %structured, %offsets, %strides = "tts.get_structured_state"(%11)
        <{resultSegmentSizes = array<i32: 1, 1, 1>}>
        : (tensor<128xi32>) -> (tensor<128xi32>, index, index)

Two things worth noticing. PtrAnalysis's prepass DID wrap the yielded index --
it inserts get_structured_state by type, and does not care that the region
belongs to a while -- but the op is never erased, so it sits there unrewritten
with the warning at py:26:
    warning: Rewriting GetStructuredStateOp failed.
And the scf.while carries the offset/stride pair (%arg5, %arg6) that nothing in
the body ever reads. Both are artefacts of a for-shaped prepass meeting a
while-shaped loop, and they are only legible with the while alone in a module.

VERIFIED
  loopiter-not-forop: 4 hits.

COMPILE-TIME PROBE: every trip overwrites the same o[0:128] slice.

POSITIVE lowering, NEGATIVE analysis: the mask index is a block argument of an scf.while, whose parent is not an scf.for.
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
    def loop16_mask_while_kernel(x_ptr, o_ptr, n, iters, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        idx = tl.arange(0, BLOCK)
        cnt = 0
        while cnt < iters:                  # scf.while, NOT scf.for
            v = tl.load(x_ptr + r, mask=idx < n, other=0.0)
            tl.store(o_ptr + r, v, mask=idx < n)
            idx += BLOCK
            cnt += 1

    return loop16_mask_while_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    o = torch.zeros(BLOCK, dtype=torch.float32)
    idx = torch.arange(BLOCK)
    for _ in range(ITERS):
        m = idx < NBOUND
        o[m] = x[:BLOCK][m]
        idx = idx + BLOCK
    return {"o": o}


SPEC = KernelSpec(
    name="loop16_mask_while_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32", "iters": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (BLOCK,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    extra={"scalar_args": [("n", "int32_t"), ("iters", "int32_t")],
           "scalar_values": {"n": NBOUND, "iters": ITERS}},
)
