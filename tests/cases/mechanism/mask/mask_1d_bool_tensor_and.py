"""An i1 TENSOR as one side of `&` -- the bool-recovery path, and the three
size checks that decide whether it becomes an unstructured mask or a failure.

Split out of the old tests/cases/ptr_structured.py section C. parseAnd is
the only place where a FAILED parse is tolerated: if the andi's result is a
tensor of i1, the failure is ignored and whatever `masks` the failed side
managed to record is used instead of a dim size. That is the entry point for
every unstructured (gather) mask in the linalg lowering. It is also guarded by two
size checks that are easy to trip, and this file trips both.

NEGATIVE KERNEL: all three loads end up refused (the recovery gets further on
load 3 than on loads 1 and 2, but minStateScalar then rejects it), so all three
fall back to tts.gather. The point is WHICH check stopped each one.

COVERS (4)   source root /workspace/triton_shared/, worktree lines
             (MaskAnalysis.cpp is lib/Analysis/MaskAnalysis.cpp)

  and-masks-rank-mismatch-fail   MaskAnalysis.cpp:376
      isBoolOp, but lhsState.masks.size() != rank. `mu` is an arith.select, so
      parse() falls off the end of the TypeSwitch and returns failure having
      recorded NOTHING -- masks.size() == 0 against rank 1. Note the check
      looks only at the LHS, which is why load 1 puts `mu` on the left.
      Load 1: `mu & (r < n)`.
  and-masks-size-mismatch-fail   MaskAnalysis.cpp:380
      lhs is fine (masks.size() == rank == 1) but rhs recorded a different
      number. Load 2 is load 1 with the operands SWAPPED -- same two values,
      different branch, which is the whole reason both ids exist.
      Load 2: `(r < n) & mu`.
  and-bool-operand-fail-tolerated  MaskAnalysis.cpp:365
      the tolerated failure itself: parse(lhs) failed and isBoolOp is true, so
      the `return failure()` is SKIPPED. Load 3's `mc` is `b8 != 0`, a cmpi
      with a non-whitelisted predicate, and parseCmp records masks[0] = the cmp
      before rejecting it. That leaves a state with masks.size() == 1 == rank,
      so both size checks pass and the per-dim merge actually runs.
  minstatescalar-neither-scalar-fail  MaskAnalysis.cpp:237
      where load 3 dies anyway: after the merge, `!lhsState.isMask()` is true
      (the failed cmp left dims EMPTY), so parseAnd calls minStateScalar, and
      neither side has a scalar. This is the branch that makes a recovered
      unstructured mask useless unless the OTHER side of the `&` is a scalar
      condition.

WHY THE THREE LOADS ARE IN ONE FILE
  Loads 1 and 2 are the same expression with the operands exchanged, and load 3
  differs only in which i1 tensor it uses. Splitting them would produce three
  files whose kernels are one line apart and whose docstrings would each have
  to re-explain the recovery mechanism.

WHAT PROVES IT FIRED
  All three tt.load ops survive --triton-to-structured with their i1 mask
  operand intact -- an accepted mask never leaves an i1 tensor behind:
      %13 = arith.andi %12, %8 : tensor<32xi1>
      %14 = tt.load %4, %13, %cst : tensor<32x!tt.ptr<f32>>
  and with tools/lower_steps.py --debug each prints
      remark: MaskAnalysis failed
      remark: PtrAnalysis: Failed to rewrite LoadOp
  02-unstructured.mlir shows the fallback keeping the mask as data:
      "tts.gather"(%arg0, %offs, %mask, %other) ... -> tensor<32xf32>
  If the recovery had SUCCEEDED instead, the output would contain a
  tts.make_gather_scatter_tptr with a `gather_scatter_mask:` clause; there is
  none, which is what distinguishes "tolerated then rejected" from "recovered".
  (mask_gs_from_structured.py in the gather family is the file where the
  recovery does succeed.)

torch equivalent: three masked gathers, see reference(). Runtime scalar n=20.

NEGATIVE: parseAnd's bool-recovery path and the two size checks plus the minStateScalar check that stop it. Runtime i32 scalar n=20. Numerically runnable through the gather path.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256
BLOCK = 32
NVAL = 20


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_1d_bool_tensor_and_kernel(x_ptr, b_ptr, o_ptr, n,
                                       BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        mp = x_ptr + r

        b8 = tl.load(b_ptr + r)
        mc = b8 != 0                       # cmpi ne: parseCmp records, then fails
        mu = tl.where(r < n, mc, b8 > 1)   # arith.select: parse records nothing

        v = tl.load(mp, mask=mu & (r < n), other=0.0)     # 1 lhs masks empty
        v += tl.load(mp, mask=(r < n) & mu, other=0.0)    # 2 rhs masks empty
        v += tl.load(mp, mask=mc & (r < n), other=0.0)    # 3 tolerated, then
        tl.store(o_ptr + r, v)                            #   minStateScalar

    return mask_1d_bool_tensor_and_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "b": torch.ones(BLOCK, dtype=torch.int8)}


def reference(inputs):
    import torch
    x, b8 = inputs["x"], inputs["b"]
    r = torch.arange(BLOCK)
    base = x[0:BLOCK]
    z = torch.zeros(BLOCK, dtype=torch.float32)
    mc = b8 != 0
    mu = torch.where(r < NVAL, mc, b8 > 1)
    v = torch.where(mu & (r < NVAL), base, z)
    v = v + torch.where((r < NVAL) & mu, base, z)
    v = v + torch.where(mc & (r < NVAL), base, z)
    return {"o": v}


SPEC = KernelSpec(
    name="mask_1d_bool_tensor_and_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "b_ptr": "*i8", "o_ptr": "*fp32", "n": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("b", IN, "int8", (BLOCK,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_values": {"n": NVAL}},
)
