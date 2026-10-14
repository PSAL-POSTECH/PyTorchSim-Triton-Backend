"""A WRAP on the row index of a matmul's TRANSPOSED A operand: wrong numbers.

`rm % M` where rm is `pid_m * BM + arange(0, BM)` and the grid covers M exactly
is the IDENTITY -- every index it can produce is already in range. Inductor's
`mm`/`bmm` templates write it anyway, guarding the general case, and upstream
turns it into a wrapping tile pointer. When the A operand is also transposed,
the answer comes back wrong: every element of it.

THE THREE CONTROLS ARE THE POINT. One variable at a time, same file otherwise:

    A transposed   wrap on A's rm   -> WRONG   this file
    A transposed   wrap on B's rn   -> right
    A transposed   no wrap          -> right
    A row-major    wrap on A's rm   -> right

So it is not the wrap, and not the transpose, but the two meeting on the same
operand. `stride_am = 1, stride_ak = M` is what `x.transpose(-1, -2)` hands the
template, and it is the layout the flip/turn machinery rewrites; a wrap on the
same axis is a second claim about that axis and the two disagree.

WHERE IT CAME FROM. tests/models/test_transformer.py's attention. PyTorchSim's
per-kernel functional verify names the buffer:

    [FuncVerify] first divergence at buffer 'buf4' (op aten.bmm.default)
                 max abs diff 1.87996, 4095/4096 elements over tol

and the model's two bmms differ in exactly this -- `key @ query^T` has the
transpose on B and is correct, `value^T @ p_attn` has it on A and is not.
Reduced from that kernel's own source by subtraction: removing the two `%`
lines, and nothing else, makes it pass.

IT MAY BE ONE DEFECT WITH THE DeepSeek ONE. `develop-e2e-deepseek` recorded a
gather whose address "an arange and two moduli get wrong" (5840509) and left it
blocked. That one is an unstructured gather and this one is a structured wrap
into a matmul, so they are only the same if the wrap arithmetic is; nobody has
established that, and this file says so rather than assuming it.

THE CAUSE, AND WHY IT IS ONLY THE TRANSPOSED SPELLING. The linalg lowering's
visitOperandSplat parks a splatted scalar -- the k-loop advance -- in
`offsets[0]`, always. The WRAP goes on the CONTIGUOUS dim, measured on the two
kernels side by side:

    transposed A    strides [1, 32]    shape [32, 0]      wrap on dim 0
    row-major A     strides [64, 1]    shape [0, 2048]    wrap on dim 1

so the two collide exactly when dim 0 is the contiguous one. Then the advance
(`BLOCK_K * stride_ak = 512`) sits under a modulo of 32 and 512 mod 32 is 0:
every k block re-reads the first. Measured three ways -- the output equals "A
stuck at its first K block" to fp32 rounding (3.8e-06) against a real answer
41.9 away, staging the first A tile alone is EXACT, and K = BK passes.

Both of upstream's own lowerings break from the same fold (createStackedCastOps
sizes its first chunk `modRow * strideRow - targetOffset`, negative once the
advance is inside targetOffset), so the repair belongs there (rule 11).

FIXED UPSTREAM, and the three repairs that failed first are the reason the
fourth is where it is:

  1. MOVE IT AT `addState` to an unwrapped dim. Exact in principle -- offsets
     are absolute element offsets the lowerings SUM -- but it moved the WRONG
     offsets too: the splat added BEFORE the modulo (`(pid * BM + arange) % M`)
     reaches the same code, belongs inside the wrap, and moving it broke the
     row-major case that worked.
  2. FOLD IT INTO THE BASE POINTER. Exact and dim-independent, and lost anyway:
     the loop carries offsets and strides as iter_args, the base is fixed.
  3. PARK IT in a `pendingAdvance` field and place it at
     createTTSMakeTensorPtrOp where `shape` is final. Wrong in both cases -- the
     state the loop boundary rebuilds is not the one placed into.

WHAT THE FOURTH DOES. Still `addState`, still the same move, with one gate: the
lhs offset for the wrapped dim must be a loop-carried BLOCK ARGUMENT. That is
the only point where both facts are true at once -- the tptr already exists so
the wrap placement is normalised, and anything added now happens after the `%`.
Before the loop the same shape means the opposite thing, which is what defeated
the first attempt.

    lib/AnalysisStructured/PtrAnalysis.cpp, PtrState::addState

WHAT WOULD BREAK THIS: any of the four control rows changing answer. Keep them.

Inductor's matmul template, reduced: an identity wrap on the row index of a transposed A. Three controls next to it are right; this one is wrong in every element.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 32
N = 64
K = 64
BM, BN, BK = 32, 32, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def wrap_transposed_a(A, B, O, M: tl.constexpr, N: tl.constexpr,
                          K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                          BK: tl.constexpr):
        GROUP_M: tl.constexpr = 8
        # A IS COLUMN-MAJOR IN ITS OWN TILE -- what `x.transpose(-1, -2)` gives.
        stride_am, stride_ak = 1, M
        stride_bk, stride_bn = N, 1
        pid = tl.program_id(0)
        grid_m = (M + BM - 1) // BM
        grid_n = (N + BN - 1) // BN
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        # THE WRAP. The grid covers M exactly, so this is the identity on every
        # index it can produce -- and it changes the answer.
        ram = rm % M
        rk = tl.arange(0, BK)
        a = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        b = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(K, 0, -BK):
            acc += tl.dot(tl.load(a), tl.load(b), allow_tf32=False)
            a += BK * stride_ak
            b += BK * stride_bk
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)
        tl.store(O + (tl.broadcast_to(idx_n + N * idx_m, [BM, BN])), acc, mask)

    return wrap_transposed_a


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(K * M, generator=g, dtype=torch.float32),
            "b": torch.randn(K * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    # `a` holds A^T: [K, M] in memory, read as [M, K] with stride (1, M).
    at = inputs["a"].reshape(K, M)
    b = inputs["b"].reshape(K, N)
    return {"o": (at.t() @ b).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="wrap_transposed_a", kernel=kernel,
    signature={"A": "*fp32", "B": "*fp32", "O": "*fp32", "M": "constexpr",
               "N": "constexpr", "K": "constexpr", "BM": "constexpr",
               "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "float32", (K * M,)),
          Arg("b", IN, "float32", (K * N,)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=((M // BM) * (N // BN),),
    reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect="blocked",
    expect_why="matmul K=16 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
    # WAS `expect="blocked"`. Fixed upstream; the four control rows above are
    # what keeps it honest if either half of the collision moves again.
)
