"""A wrap on a TRANSPOSED A whose batch stride sits INSIDE the row pitch.

The sibling of mixed_wrap_transposed_a, and NOT the same defect: that one was
the k-advance folded into the wrapped dim (fixed in triton_shared
`fix/wrap-advance-fold`), and this one survives that fix. What it needs is the
layout an attention bmm actually has -- a per-head slice of one [S, H*D] buffer:

    stride_am = 1        the head's D axis, contiguous
    stride_ak = LDA      the row pitch of the WHOLE buffer, not of the slice
    stride_aq = M        the batch stride, INSIDE that pitch

That is `value.transpose(-1, -2)` on a projection output, which is where
tests/models/test_transformer.py's second bmm comes from -- and with the first
defect fixed, this is the model's next divergence:

    [FuncVerify] first divergence at buffer 'buf9' (op aten.bmm.default)
                 max abs diff 0.711388, 392907/393216 elements over tol

THE CONTROLS, one variable at a time:

    head slice   wrap on A's rm   -> WRONG   this file
    head slice   wrap on B's rn   -> right
    head slice   no wrap          -> right
    tight A^T    wrap on A's rm   -> right   (mixed_wrap_transposed_a)

WHAT IS ALREADY KNOWN. In the model's own kernel the wrap lands on the WRONG
DIMENSION: the source wraps the m index, whose stride is 1 and which is dim 0,
and upstream emits

    tts.make_tptr(..., strides = (1, 768), offsets = (%arg10, 0), shape = (64))
                  static_shape = [0, dyn]        <- the 64 is on dim 1

so a wrap of 64 elements is attached to an axis that steps by 768. The advance
is on dim 0 and unwrapped, which is why the earlier fix does not reach it.

FIXED UPSTREAM, in the same branch as its sibling. The placement is decided in
`PtrState::addState`'s shape merge, whose dim-0 branch swaps the wrap to dim 1
and assumes the scalar increments dim 1 -- with `WILL LEAD TO WRONG COMPILATION`
written beside it for when that assumption is wrong. The swap and the move are
ONE decision and the discriminator is the stride of the dim the wrap would go
to:

    strides [64, 1]   target has stride 1 -- the wrap carries, swap as before
    strides [1, 64]   THIS FILE: the wrap cannot go there, so the SCALAR moves
                      and the wrap stays on the axis the program wrapped
    strides [1, 0]    a gather rebuild: touch nothing. Its offset vector is
                      re-derived from the index expression, so moving a scalar
                      out of `offsets` only adds a second copy -- measured on
                      index_select_row_with_mod_plus_offset, where it shifted
                      every row by one. Declining the swap is not open either:
                      the else is `return failure()` and the access leaves the
                      structured path.

Reduced to Q=2, M=32, N=64, K=64, LDA=64: small enough to read a scratchpad by
hand, and it fails in every element.

The attention bmm's real layout: a per-head slice of one [S, H*D] buffer, transposed, with the template's wrap on its row index.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

Q = 2
M = 32
N = 64
K = 64
LDA = Q * M
BM, BN, BK = 32, 32, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def wrap_head_slice(A, B, O, M: tl.constexpr, N: tl.constexpr,
                        K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                        BK: tl.constexpr, LDA: tl.constexpr):
        GROUP_M: tl.constexpr = 8
        # A IS COLUMN-MAJOR IN ITS OWN TILE -- what `x.transpose(-1, -2)` gives.
        # A HEAD SLICE: the batch stride sits INSIDE the row pitch, which is
        # what a [S, H*D] projection sliced per head gives the bmm template.
        stride_am, stride_ak, stride_aq = 1, LDA, M
        q = tl.program_id(1)
        stride_bk, stride_bn, stride_bq = N, 1, K * N
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
        a = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak
                 + q * stride_aq)
        b = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn
                 + q * stride_bq)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(K, 0, -BK):
            acc += tl.dot(tl.load(a), tl.load(b), allow_tf32=False)
            a += BK * stride_ak
            b += BK * stride_bk
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)
        tl.store(O + (tl.broadcast_to(idx_n + N * idx_m + M * N * q, [BM, BN])), acc, mask)

    return wrap_head_slice


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(K * LDA, generator=g, dtype=torch.float32),
            "b": torch.randn(Q * K * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    # `a` holds A^T: [K, M] in memory, read as [M, K] with stride (1, M).
    import torch
    mem = inputs["a"].reshape(K, LDA)
    b = inputs["b"].reshape(Q, K, N)
    outs = [mem[:, q * M:(q + 1) * M].t() @ b[q] for q in range(Q)]
    return {"o": torch.stack(outs).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="wrap_head_slice", kernel=kernel,
    signature={"A": "*fp32", "B": "*fp32", "O": "*fp32", "M": "constexpr",
               "N": "constexpr", "K": "constexpr", "BM": "constexpr",
               "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK, "LDA": LDA},
    args=[Arg("a", IN, "float32", (K * LDA,)),
          Arg("b", IN, "float32", (Q * K * N,)),
          Arg("o", OUT, "float32", (Q * M * N,))],
    grid=((M // BM) * (N // BN), Q),
    reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect="blocked",
    expect_why="matmul K=16 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
    # WAS `expect="blocked"`. Fixed upstream; the controls above are what keeps
    # the swap and the move one decision if either side is touched again.
)
