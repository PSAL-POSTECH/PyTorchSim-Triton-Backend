"""A SCALAR !tt.ptr carried through scf.for -- rank-0 state, no make_tptr.

Every other pointer in the loop census is a tensor of pointers. This one is a
bare `!tt.ptr<f32>` advanced by 1 per trip, which gives PtrState rank 0: it has
a `scalar` offset and no per-dimension offsets or strides at all. Three separate
places in PtrAnalysis have a rank-0 special case and this is the only file that
reaches any of them.

COVERS (4 ids; PtrAnalysis.cpp = lib/AnalysisStructured/PtrAnalysis.cpp)

  forop-scalar-ptr-rank0            :1680 (HEAD :1671)
      isPointerType(arg) is true but state->getRank() == 0, so knownPtrs[arg] is
      set and NO op is inserted into the body. The scalar pointer iter-arg keeps
      its original SSA value. Proof -- the body of the loop in
      01-structured.mlir contains no tts.make_tptr at all, and the load is still
      addressed by the raw block argument:
          %7 = tt.load %arg3 : !tt.ptr<f32>
      Compare loop_01, where the first statement in the body is a make_tptr.

  reconcile-rank0-scalar            :1579 (HEAD :1570)
      Only newState.scalar is replaced, so exactly ONE extra iter-arg is
      threaded. Witness is the scf.for signature:
          %5:3 = scf.for ... iter_args(%arg3 = %4, %arg4 = %3, %arg5 = %cst)
                     -> (!tt.ptr<f32>, index, f32)  : i32 {
      One `index` between the pointer and the f32 accumulator. loop_01's
      rank-1 equivalent has TWO (offset and stride).

  getstate-rank0-scalar             :1728 (HEAD :1719)
      rewriteGetStructuredStateOp's rank == 0 && scalar != null arm:
      replacements = {remappedPtr, state.scalar}. %arg4 above is seeded from
      `%3 = arith.index_cast %1 : i32 to index`, the analyzed scalar offset.

  tts-get-structured-state-scalar-ptr
      TritonStructuredDialect.td:244, implemented at
      lib/Dialect/TritonStructured/IR/TritonStructuredOps.cpp:412-415:
      a scalar pointee takes the else branch and sets offsetSegmentSize = 1
      while leaving strideSegmentSize = 0 -- one offset result, ZERO stride
      results. This file is the ONLY one in the whole loop set that reaches it;
      gdb on TritonStructuredOps.cpp:415 gives 5 hits here and 0 for all
      sixteen others. TableGen carries no code, so the census's own line cannot
      be broken on; :415 is the arm that computes the segments the op is
      declared with. The op is erased on success, so the surviving textual
      witness is the `-> (!tt.ptr<f32>, index, f32)` signature above: 1 offset,
      no strides.

REMARKS ARE EXPECTED AND ARE NOT FAILURES.  --triton-to-structured prints
    PtrAnalysis: pointer is not replace with tts.make_tptr so skip
    PtrAnalysis: Failed to rewrite ... (loop_03...py:22 and :24)
for the scalar load and store. That is the scalar-pointer path declining to
build a tensor pointer -- which is exactly the branch this file is here for --
not an error. The module lowers all the way to memref.

VERIFIED
  forop-scalar-ptr-rank0 1, reconcile-rank0-scalar 2, getstate-rank0-scalar 4;
  plus TritonStructuredOps.cpp:415 = 5 under a direct breakpoint.

RUNNABLE. o[pid] = x[pid] + x[pid+1] + x[pid+2] + x[pid+3].

POSITIVE. A SCALAR pointer as the loop's iter-arg: rank-0 state, one extra iter-arg, and no tts.make_tptr anywhere.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
K = 4
GRID = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop03_scalar_ptr_kernel(x_ptr, o_ptr, K: tl.constexpr):
        pid = tl.program_id(0)
        q = x_ptr + pid              # SCALAR !tt.ptr, rank-0 PtrState
        s = 0.0
        for k in range(0, K):
            s += tl.load(q)
            q = q + 1
        tl.store(o_ptr + pid, s)

    return loop03_scalar_ptr_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    o = torch.zeros(GRID, dtype=torch.float32)
    for pid in range(GRID):
        o[pid] = x[pid:pid + K].sum()
    return {"o": o}


SPEC = KernelSpec(
    name="loop03_scalar_ptr_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "K": "constexpr"},
    constexprs={"K": K},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (GRID,)),
    ],
    grid=(GRID,),
    reference=reference,
    make_inputs=make_inputs,
)
