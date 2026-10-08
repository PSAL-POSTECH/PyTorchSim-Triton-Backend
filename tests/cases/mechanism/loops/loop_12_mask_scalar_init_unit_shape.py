"""A splat init and a 1-element tensor: the mask parses, and is then silently ignored.

Two branches fire together here and they are inseparable, which is why one file
covers both. `tl.full((1,), pid, tl.int32)` gives an init state that has a
`scalar` and no start/end, which is parseLoopIterArg's second success shape; and
because the tensor has exactly one element, the comparison it feeds is a
tensor<1xi1>, whose every dimension has extent 1. Change the shape to drop the
second branch and you drop the first with it.

The second one is a latent bug in the linalg lowering, not a feature: parseCmp's loop
never sets cmpOpDim, so it stays -1, the computed newDim is discarded, and the
dims are copied verbatim from the lhs. The comment at MaskAnalysis.cpp:480
claims "parseCmp must fail later" for that case. It does not. The mask is
emitted as fully true.

COVERS (2 ids, lib/Analysis/MaskAnalysis.cpp)

  loopiter-scalar-init              :677 (HEAD :677)
      The init state has neither start nor end (it asserts it has a scalar), so
      start := 0 and end := scalar. Proof, from 01-structured.mlir -- the
      iter-arg %arg6 is seeded from %1, the index_cast of the program id (the
      scalar), and the extent is computed from it:
          %11:3 = scf.for %arg4 = %c0_i32 to %arg3 step %c1_i32
                      iter_args(%arg5 = %4, %arg6 = %1, %arg7 = %c0) {
            %15 = arith.addi %arg6, %9 : index
            %17 = arith.minsi %15, %16 : index
            %18 = arith.maxsi %17, %arg6 : index
            %19 = arith.subi %18, %arg6 : index

  cmp-all-unit-shape                :464 (HEAD :464)
      The cmp result is shaped but every dim has extent 1, so cmpOpDim is never
      assigned and parseCmp SUCCEEDS with dim == -1. Proof -- %19 above, the
      extent this arm just computed, is then NOT used by the load, and the mask
      is emitted as a static full extent:
          %20 = "tts.load"(%13, %cst_0)
                  <{operandSegmentSizes = array<i32: 1, 0, 1>,
                    static_mask_dims = array<i64: 1>}>
                : (tensor<1x!tt.ptr<f32>>, f32) -> tensor<1xf32>
      operandSegmentSizes 1,0,1 means ptr + NO mask operand + other, and
      static_mask_dims = [1] is the whole tensor. The `j < n` in the source has
      been discarded. loop_11's equivalent load carries a dynamic extent and a
      mask operand; that contrast is the cleanest way to see this branch.

VERIFIED
      tests/cases/loop_12_mask_scalar_init_unit_shape.py
  loopiter-scalar-init 4, cmp-all-unit-shape 4.

COMPILE-TIME PROBE. Because the mask is dropped, the compiled kernel does NOT
honour `j < n`; the reference describes what the source asks for, not what this
lowering produces. That divergence is the finding.

POSITIVE lowering, but the mask is SILENTLY DROPPED: a tensor<1xi1> comparison leaves cmpOpDim at -1 and parseCmp still returns success.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
GRID = 2
NBOUND = 96
ITERS = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop12_mask_unit_kernel(x_ptr, o_ptr, n, iters):
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        j = tl.full((1,), pid, tl.int32)    # SPLAT init: scalar-only state
        for _ in range(iters):
            v = tl.load(x_ptr + j, mask=j < n, other=0.0)   # tensor<1xi1> mask
            tl.store(o_ptr + j, v, mask=j < n)
            j += nprog

    return loop12_mask_unit_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    o = torch.zeros(N, dtype=torch.float32)
    for pid in range(GRID):
        for it in range(ITERS):
            k = pid + it * GRID
            if k < NBOUND:
                o[k] = x[k]
    return {"o": o}


SPEC = KernelSpec(
    name="loop12_mask_unit_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32", "iters": "i32"},
    constexprs={},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (N,)),
    ],
    grid=(GRID,),
    reference=reference,
    make_inputs=make_inputs,
    extra={"scalar_args": [("n", "int32_t"), ("iters", "int32_t")],
           "scalar_values": {"n": NBOUND, "iters": ITERS}},
)
