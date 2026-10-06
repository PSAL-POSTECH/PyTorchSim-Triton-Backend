"""`maskA & maskB` -- the two ways parseAnd merges two rank-1 mask states, and
why one of them emits an arith.select instead of an arith.minsi.

Split out of the old tests/cases/ptr_structured.py section C. Both loads
here are accepted; the interesting part is that they produce structurally
DIFFERENT clamps. Two real masks intersect to a min. A scalar condition ANDed
with a mask cannot intersect with anything -- it is all or nothing -- so the
dim becomes `select(cond, dim, 0)`.

COVERS (8)   source root /workspace/triton_shared/, worktree lines
             (MaskAnalysis.cpp is lib/Analysis/MaskAnalysis.cpp)

  dispatch-andi          MaskAnalysis.cpp:40
      parse() reaching the AndIOp arm with a TENSOR result. Both loads.
  dispatch-int-scalar    MaskAnalysis.cpp:36
      the IntegerType arm, checked BEFORE AddI/AndI/CmpI, so `n > 5` -- a
      SCALAR arith.cmpi producing i1 -- is treated as an opaque scalar and
      never goes near parseCmp's slt/ult/sge whitelist. This is why `sgt` is
      accepted here but rejected in mask_1d_parse_reject.py.
  intscalar-i1           MaskAnalysis.cpp:313
      that i1 is kept as the raw SSA value with NO index_cast -- the only form
      selectOFRs will take as a select condition.
  splat-i1-masks-init    MaskAnalysis.cpp:771
      tt.splat of the i1 to tensor shape: sets scalar AND dims AND fills masks
      with `rank` nullptrs. That rank-sized all-null masks vector is exactly
      what lets this state survive parseAnd's masks.size() == rank check.
  and-minstatescalar     MaskAnalysis.cpp:447
      !lhsState.isMask() || !rhsState.isMask() -> minStateScalar. The splat
      state is not a pure mask (it carries a scalar), so load 1 goes here.
  minstatescalar-select  MaskAnalysis.cpp:257
      exactly one side has a scalar -> dims[i] = select(cond, otherDim, 0).
  and-both-structured-dim  MaskAnalysis.cpp:391
      the per-dim merge loop with !lhsV && !rhsV: both sides are structured on
      dim 0, so masks.push_back(nullptr) and no unstructured mask is created.
      Load 2. (The merge loop runs for BOTH loads -- andi on i1 tensors is
      always isBoolOp -- which is why this id belongs next to the other one.)
  and-minstates          MaskAnalysis.cpp:450
      both sides are pure masks -> dims[i] = minsi(lhs.dims[i], rhs.dims[i]).
      Load 2.

WHAT PROVES IT FIRED (01-structured.mlir)

  load 1 -- a select, and the i1 %4 goes in raw, with no index_cast on it:
      %4  = arith.cmpi sgt, %arg2, %c5_i32 : i32     <- SCALAR cmp, i1
      %7  = tt.splat %4 : i1 -> tensor<32xi1>
      %12 = arith.select %4, %11, %c0_0 : index
      %13 = "tts.load"(%2, %12, %cst_1) <{operandSegmentSizes = array<i32: 1,1,1>,
            static_mask_dims = array<i64: -9223372036854775808>}>
  load 2 -- one minsi/maxsi chain per mask, then a min OF THE TWO:
      %19 = arith.maxsi %18, %c0_3 : index      (from r < n)
      %22 = arith.maxsi %21, %c0_5 : index      (from r < m)
      %23 = arith.minsi %19, %22 : index        <- and-minstates
      %24 = "tts.load"(%2, %23, %cst_6) ...
  Neither load leaves a tt.load or a tts.make_gather_scatter_tptr behind: both
  masks were fully absorbed into a single index operand.

torch equivalent: see reference(). Runtime scalars n=20, m=12.

parseAnd's two merge arms: select for a scalar condition, minsi for two real masks. Runtime i32 scalars n=20, m=12. Numerically runnable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256
BLOCK = 32
NVAL = 20
MVAL = 12


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_1d_and_combine_kernel(x_ptr, o_ptr, n, m, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        mp = x_ptr + r
        # 1 scalar i1 AND a mask -> select(cond, dim, 0)
        v = tl.load(mp, mask=(n > 5) & (r < n), other=0.0)
        # 2 two masks -> minsi(dim, dim)
        v += tl.load(mp, mask=(r < n) & (r < m), other=0.0)
        tl.store(o_ptr + r, v)

    return mask_1d_and_combine_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    r = torch.arange(BLOCK)
    base = x[0:BLOCK]
    z = torch.zeros(BLOCK, dtype=torch.float32)
    v = torch.where(r < NVAL, base, z)          # n > 5 is true
    v = v + torch.where(r < min(NVAL, MVAL), base, z)
    return {"o": v}


SPEC = KernelSpec(
    name="mask_1d_and_combine_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32", "m": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_values": {"n": NVAL, "m": MVAL}},
)
