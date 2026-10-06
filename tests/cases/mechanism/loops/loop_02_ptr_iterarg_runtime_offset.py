"""loop_01, with the pre-loop offset made a RUNTIME value instead of a constant.

The only difference from loop_01_ptr_iterarg.py is `pid * BLOCK + r` in place of
`r`. That one change moves rewriteGetStructuredStateOp from its int-attr arm to
its already-a-Value arm, and nothing else about the lowering changes. Reading
the two files side by side is the point: they are deliberately the same shape.

COVERS (1 id, lib/AnalysisStructured/PtrAnalysis.cpp)

  getstate-offsets-strides-value    :1744 (HEAD :1735)
      rank > 0 and the OpFoldResult for this position is already a Value, so it
      is cast<Value>'d and pushed straight into the replacement list. No
      arith.constant is emitted for it.

      Proof, from 01-structured.mlir -- the offset iter-arg is %4, an
      index_cast of the program id, where loop_01 had a fresh `%c0`:
          %2 = arith.muli %0, %c128_i32 : i32
          %4 = arith.index_cast %2 : i32 to index
          %8 = tts.make_tptr %arg0 to sizes: [128], strides: [1],
                             offsets: [%4], ...
          %10:4 = scf.for ... iter_args(%arg3 = %8, %arg4 = %4, %arg5 = %c1,
                                        %arg6 = %cst) -> (...)
      %arg4 is seeded from %4 (a Value, this branch) while %arg5 is still seeded
      from %c1 (a constant, loop_01's branch) -- both arms run in this one loop,
      which is why the const arm is claimed by loop_01 and only the value arm
      here.

WHY IT IS A SEPARATE FILE rather than a second loop inside loop_01: the two
arms are distinguished by which iter-arg position is a constant, and that is
read off the scf.for signature. Two loops in one module means two signatures to
attribute by eye, and the whole reason for the split is not having to.

VERIFIED
      tests/cases/loop_02_ptr_iterarg_runtime_offset.py
  getstate-offsets-strides-value: 2 hits.

RUNNABLE. Each program sums its own 4 BLOCK-sized slices starting at pid*BLOCK
and writes them to o[pid*BLOCK : (pid+1)*BLOCK].

POSITIVE. Same carried pointer as loop_01 but with a RUNTIME pre-loop offset, so the state's offset is a Value, not an attribute.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
K = 4
BLOCK = 128
GRID = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop02_runtime_offset_kernel(x_ptr, o_ptr, K: tl.constexpr,
                                     BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, BLOCK)
        p = x_ptr + (pid * BLOCK + r)         # offset is an SSA value, not a const
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(0, K):
            acc += tl.load(p)
            p = p + BLOCK
        tl.store(o_ptr + pid * BLOCK + r, acc)

    return loop02_runtime_offset_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    o = torch.zeros(GRID * BLOCK, dtype=torch.float32)
    for pid in range(GRID):
        acc = torch.zeros(BLOCK, dtype=torch.float32)
        for k in range(K):
            base = pid * BLOCK + k * BLOCK
            acc += x[base:base + BLOCK]
        o[pid * BLOCK:(pid + 1) * BLOCK] = acc
    return {"o": o}


SPEC = KernelSpec(
    name="loop02_runtime_offset_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "K": "constexpr", "BLOCK": "constexpr"},
    constexprs={"K": K, "BLOCK": BLOCK},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (GRID * BLOCK,)),
    ],
    grid=(GRID,),
    reference=reference,
    make_inputs=make_inputs,
)
