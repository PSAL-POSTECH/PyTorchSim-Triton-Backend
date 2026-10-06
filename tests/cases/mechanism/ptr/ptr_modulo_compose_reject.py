"""A modulo COMBINED with something else -- the five ways that composition is
refused, none of which is about the modulo's shape.

Split out of the old tests/cases/ptr_structured.py section B. NEGATIVE
KERNEL: every load here is supposed to be rejected. PtrState carries at most
one modulus per dimension and nothing that says WHERE inside the wrapped range
the access starts, so as soon as a second modulus, a runtime per-lane modulus,
or a non-zero offset is added on top, the state can no longer be written down.
All five fall back to tts.gather and the pipeline still succeeds.

Companion files: ptr_modulo_wrap_shapes.py (the accepted shapes),
ptr_modulo_rank_reject.py (rejections caused by the modulo's RANK).

COVERS (5)   source root /workspace/triton_shared/, worktree lines

  rem-rhs-not-scalar-fail    lib/AnalysisStructured/PtrAnalysis.cpp:821
      visitOperandRem: !rhsState.scalar. Load 1 divides by a tensor loaded from
      memory, so every lane could wrap at a different modulus.
      remark: "only support cases when rhs of remainder contains scalar".
  add-1d-modulo-fail         PtrAnalysis.cpp:729
      visitOperandAdd: either side is rank 1 AND has a modulo. Load 2 is
      `(r % M) + 5`; a constant shift of a wrapped range is not a wrapped
      range, so the pattern is refused outright.
      remark: "do not support this pattern: a + arange(0, K) % M".
  rem-multi-modulo-fail      PtrAnalysis.cpp:851
      visitOperandRem: the state ALREADY has a modulo and the rank is not 1.
      Load 3 is `(rm % M) % KK` at rank 2. (At rank 1 with gather/scatter
      enabled this same condition takes the rebuild-as-gather path instead --
      that case lives in the ptr_gather_* family, which is why the rank here
      has to be 2.)
      remark: "do not support multiple modulo within an expression".
  addstate-both-modulo-fail  PtrAnalysis.cpp:424
      addState: lhsState.hasModulo() && rhsState.hasModulo(). Load 4 wraps the
      row index and the column index. Note the failure happens AFTER the
      per-dim offset/stride loop has already run, so the arith ops it created
      are left behind in the IR.
      remark: "do not support adding two pointer states that both have modulo".
  addstate-modulo-fail       PtrAnalysis.cpp:490
      the modulo loop's else arm: lhs has a modulo in dim i, rhs's offset for
      that dim is NOT zero, and it is not the dim-0-splat special case (rhs
      here is a broadcast expand_dims state, not a splat, so it has no
      `scalar`). Load 5 adds a second row term `(3 + rm)[:, None] * BN` on top
      of the wrapped rows.
      remark: "do not support adding to operand with modulo".

WHY ALL FIVE SIT IN ONE FILE
  They differ only in the mask-free address expression -- same base, same
  block shape, same fallback -- and each is one line. Reading them side by side
  is the point: 2 and 4 look almost identical to accepted code in
  ptr_modulo_wrap_shapes.py, and the difference is exactly one term.

WHAT PROVES IT FIRED
  Not one tts.make_tptr with a non-zero `shape:` entry exists in the module --
  every arith.remsi survives and becomes a gather offset instead:
      %14 = arith.remsi %13, %cst_2 : tensor<32xi32>
      %15 = tt.addptr %3, %14 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  and in 02-unstructured.mlir
      %x = "tts.gather"(%arg0, %14) ... -> tensor<32xf32>
  tools/lower_steps.py --debug prints the five remarks quoted above, in source
  order, one per load.

torch equivalent: five gathers, see reference().

NEGATIVE: five ways of composing a modulo that PtrAnalysis must refuse. Every remsi surviving as a gather offset is the observable. Numerically runnable through the fallback gather path.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 8
KK = 3
BLOCK = 32
BM = 16
BN = 32
N = 1024
NO = BLOCK + BM * BN


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_modulo_compose_reject_kernel(x_ptr, mod_ptr, o_ptr,
                                         M: tl.constexpr, KK: tl.constexpr,
                                         BLOCK: tl.constexpr,
                                         BM: tl.constexpr, BN: tl.constexpr):
        r = tl.arange(0, BLOCK)
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)

        mv = tl.load(mod_ptr + r)
        a = tl.load(x_ptr + (r % mv))          # 1 modulus is a tensor
        b = tl.load(x_ptr + ((r % M) + 5))     # 2 offset added to a rank-1 wrap

        # 3 two moduli in one expression, at rank 2
        c = tl.load(x_ptr + ((rm[:, None] % M) % KK) * BN + rn[None, :])
        # 4 both operands of the add carry a modulo
        d = tl.load(x_ptr + (rm[:, None] % M) * BN + (rn[None, :] % M))
        # 5 non-zero offset added to the dim that already wraps
        e = tl.load(x_ptr + (rm[:, None] % M) * BN
                    + (3 + rm)[:, None] * BN + rn[None, :])

        tl.store(o_ptr + r, a + b)
        tl.store(o_ptr + BLOCK + rm[:, None] * BN + rn[None, :], c + d + e)

    return ptr_modulo_compose_reject_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "mod": torch.full((BLOCK,), 3, dtype=torch.int32)}


def reference(inputs):
    import torch
    x, mod = inputs["x"], inputs["mod"].long()
    r = torch.arange(BLOCK)
    a = x[r % mod]
    b = x[(r % M) + 5]
    rm = torch.arange(BM).reshape(BM, 1)
    rn = torch.arange(BN).reshape(1, BN)
    c = x[(((rm % M) % KK) * BN + rn).reshape(-1).long()]
    d = x[((rm % M) * BN + (rn % M)).reshape(-1).long()]
    e = x[((rm % M) * BN + (3 + rm) * BN + rn).reshape(-1).long()]
    return {"o": torch.cat([a + b, c + d + e])}


SPEC = KernelSpec(
    name="ptr_modulo_compose_reject_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "mod_ptr": "*i32", "o_ptr": "*fp32",
               "M": "constexpr", "KK": "constexpr", "BLOCK": "constexpr",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"M": M, "KK": KK, "BLOCK": BLOCK, "BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (N,)), Arg("mod", IN, "int32", (BLOCK,)),
          Arg("o", OUT, "float32", (NO,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
