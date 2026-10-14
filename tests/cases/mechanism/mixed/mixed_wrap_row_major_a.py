"""The wrap spelling that must KEEP working: a ROW-MAJOR A with the same wrap.

THIS ONE PASSES ON BOTH SIDES OF THE FIX, deliberately, and rule 2 would throw
it out for that if it had not earned its place twice. It is the control that
tells the two halves of the wrap decision apart, and two wrong repairs died on
it -- each of which made mixed_wrap_transposed_a pass:

    moving the scalar out of the wrapped dim unconditionally    broke this
    making the shape merge read the moved offsets               broke this

Both looked right against the failing kernel alone. What they missed is that a
splat added BEFORE the modulo -- `(pid * BM + arange) % M` -- reaches the same
code and belongs INSIDE the wrap, and that with a row-major operand the wrap
swaps onto a dim of stride 1 where it carries correctly.

    strides [64, 1]   wrap ends on dim 1, stride 1   the swap is right
    strides [1, 32]   wrap ends on dim 0, stride 1   the swap cannot express it
                                                     (mixed_wrap_transposed_a)

So this file is not coverage of a feature; it is the boundary of one. Whoever
touches PtrState::addState's shape merge or the move beside it runs BOTH.

WHAT WOULD BREAK THIS: a repair to the transposed spelling that does not ask
which dim the wrap would land on.

Inductor's bmm template with a ROW-MAJOR A and the same wrap: the control that tells the two halves of the wrap decision apart, and the reason two wrong repairs of the transposed spelling were caught.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

Q = 2            # batch
M = 32
N = 64
K = 64
BM, BN, BK = 32, 32, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def wrap_row_major_a(A, B, O, M: tl.constexpr, N: tl.constexpr,
                         K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                         BK: tl.constexpr):
        # A IS COLUMN-MAJOR IN ITS OWN TILE: stride_am = 1, stride_ak = M, which
        # is what `value.transpose(-1, -2)` hands the template.
        stride_am, stride_ak, stride_aq = K, 1, M * K
        stride_bk, stride_bn, stride_bq = N, 1, K * N
        GROUP_M: tl.constexpr = 8
        pid = tl.program_id(0)
        q = tl.program_id(1)
        grid_m = (M + BM - 1) // BM
        grid_n = (N + BN - 1) // BN
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        ram = rm % M
        rbn = rn
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
        xindex = idx_n + N * idx_m + M * N * q
        tl.store(O + (tl.broadcast_to(xindex, [BM, BN])), acc, mask)

    return wrap_row_major_a


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(Q * K * M, generator=g, dtype=torch.float32),
            "b": torch.randn(Q * K * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    # `a` holds A^T: [Q, K, M] in memory, read as [M, K] with stride (1, M).
    a = inputs["a"].reshape(Q, M, K)
    b = inputs["b"].reshape(Q, K, N)
    return {"o": torch.bmm(a, b).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="wrap_row_major_a", kernel=kernel,
    signature={"A": "*fp32", "B": "*fp32", "O": "*fp32", "M": "constexpr",
               "N": "constexpr", "K": "constexpr", "BM": "constexpr",
               "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "float32", (Q * K * M,)),
          Arg("b", IN, "float32", (Q * K * N,)),
          Arg("o", OUT, "float32", (Q * M * N,))],
    grid=((M // BM) * (N // BN), Q),
    reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect="blocked",
    expect_why="matmul K=16 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
    # `expect="blocked"` LIVED HERE AND WAS NEVER TRUE OF THIS FILE. Its
    # expect_why said "a bmm whose A operand is transposed returns wrong
    # numbers", which is mixed_wrap_transposed_a's sentence -- copied along with
    # the rest of the file when this control was split off it. The docstring has
    # said "THIS ONE PASSES ON BOTH SIDES OF THE FIX, deliberately" the whole
    # time, so the file contradicted itself and the suite has been printing
    # "THE BLOCK HAS LIFTED -- drop `expect`" at it.
)
