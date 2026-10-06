"""The canonical grid-stride loop: a masked load whose index is an scf.for iter-arg.

MaskState::parseLoopIterArg is the piece of MaskAnalysis that only exists
because a mask's index tensor can cross a loop boundary. It succeeds in exactly
two shapes; this file is the first (init derived from tl.arange, so the state
has start/end) and loop_12 is the second (splat init, scalar-only state). The
four rejection arms are loop_13 through loop_16.

`iters` is a RUNTIME i32, not a constexpr. If the trip count were compile-time
the frontend would unroll the loop away, the iter-arg would vanish, and
parseLoopIterArg would never be called at all.

COVERS (2 ids, lib/Analysis/MaskAnalysis.cpp -- NOT the AnalysisStructured one)

  dispatch-blockarg                 :52 (HEAD :52)
      MaskState::parse's !operand.getDefiningOp() arm for a block argument that
      is not a plain integer -- here `mi`, a tensor<128xi32> scf.for iter-arg.
      Integer block args were already taken at :36.

  loopiter-range-init               :683 (HEAD :683)
      The init state has start and end, so start becomes
      forOp.getRegionIterArg(argIndex + 1) -- the OFFSET iter-arg, +1 by a
      convention borrowed from PtrAnalysis's prepass -- and end = start + dist.
      Proof, from 01-structured.mlir, inside the loop body:
          %17:3 = scf.for %arg4 = %c0_i32 to %arg3 step %c1_i32
                      iter_args(%arg5 = %8, %arg6 = %5, %arg7 = %c1) {
            %21 = arith.subi %14, %13 : index      // dist, hoisted pre-loop
            %22 = arith.addi %arg6, %21 : index    // end   = start + dist
            %24 = arith.minsi %22, %23 : index
            %25 = arith.maxsi %24, %arg6 : index   // clamped against start
            %26 = arith.subi %25, %arg6 : index    // mask extent
            %27 = "tts.load"(%19, %26, %cst_1)
                    <{operandSegmentSizes = array<i32: 1, 1, 1>,
                      static_mask_dims = array<i64: -9223372036854775808>}>
      %arg6 IS the offset iter-arg and it is used as `start` throughout. The
      -9223372036854775808 in static_mask_dims is ShapedType::kDynamic: the
      mask extent is an SSA value recomputed every trip, which is what a
      successfully parsed loop-carried mask looks like. Compare loop_12, where
      the same slot is the constant 1.

VERIFIED
  dispatch-blockarg 4, loopiter-range-init 4.

RUNNABLE. A standard grid-stride copy of x into o for indices below `n`; see
`reference`.

POSITIVE. The canonical grid-stride loop: the mask's index tensor is an scf.for iter-arg whose init came from tl.arange.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 128
GRID = 2
NBOUND = 96
ITERS = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop11_mask_range_kernel(x_ptr, o_ptr, n, iters, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        step = nprog * BLOCK
        mi = pid * BLOCK + tl.arange(0, BLOCK)     # make_range-derived index
        for _ in range(iters):                     # RUNTIME trip count
            v = tl.load(x_ptr + mi, mask=mi < n, other=0.0)
            tl.store(o_ptr + mi, v, mask=mi < n)
            mi += step

    return loop11_mask_range_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    o = torch.zeros(N, dtype=torch.float32)
    for pid in range(GRID):
        for it in range(ITERS):
            lo = pid * BLOCK + it * GRID * BLOCK
            for j in range(BLOCK):
                if lo + j < NBOUND:
                    o[lo + j] = x[lo + j]
    return {"o": o}


SPEC = KernelSpec(
    name="loop11_mask_range_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32", "iters": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
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
