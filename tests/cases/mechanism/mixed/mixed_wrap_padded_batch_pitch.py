"""A bmm whose A is row-major with a PADDED batch pitch, and whose wrap wraps.

The third wrap spelling, and the first one where `rm % M` is not the identity.
`mixed_wrap_transposed_a` and `mixed_wrap_head_slice` both cover a grid that
divides M exactly, so the modulo the template writes can only ever return what
it was given; this one has M = 197 against BLOCK_M = 32, so the last of seven
row blocks reaches 223 and the wrap really does fold rows 197..223 back onto
0..26. They are masked away at the store, which is why the template is allowed
to read them at all -- and it is why the wrap has to be exact rather than merely
present.

    stride_am = 197      row-major inside the head
    stride_ak = 1
    stride_aq = 38816    the batch stride, and NOT M*K = 38809

THE SEVEN-ELEMENT HOLE IS THE POINT. `empty_strided((1, 12, 197, 197),
(465792, 38816, 197, 1))` is what Inductor hands the second attention bmm, so
the head pitch is not the head's size and `q * stride_aq` cannot be folded into
either the row or the k axis. Every earlier wrap kernel in this directory has a
batch stride that is exactly the product of the two it sits above:

    mixed_wrap_transposed_a    no batch axis at all
    mixed_wrap_row_major_a     stride_aq = M*K      packed
    mixed_wrap_head_slice      stride_aq = M        INSIDE the row pitch
    this file                  stride_aq = M*K + 7  OUTSIDE it, and not a
                                                    multiple of either

B IS A HEAD SLICE, unchanged from mixed_wrap_head_slice: `stride_bq = 64` inside
`stride_bk = 768`, which is `[S, H*D]` sliced per head. Both operands are kept
in the shape the model has so that a repair aimed at one of them cannot pass by
breaking the other.

WHERE IT CAME FROM. test_vit's
`triton_npu_fused__safe_softmax_addmm_bmm_clone_select_squeeze_transpose_
unsqueeze_view_12`, which is `attn @ v`. With the storage-extent fix in place
its input `buf18` verifies and its output does not:

    [FuncVerify] first divergence at buffer 'buf20' (op aten.bmm.default)
                 max abs diff 1.17539, 138274/151296 elements over tol

Reduced to Q=2 heads, M=37, N=64, K=37, PITCH=K*M+7. M is kept off a multiple of
BLOCK_M so the wrap still wraps, and small enough to read a scratchpad by hand.

WHAT WOULD BREAK THIS: any repair to the wrap that assumes the batch stride is
the product of the axes it spans -- folding `q * stride_aq` into the wrapped
dimension, or deriving the pitch from the shape rather than reading it.

Attention's second bmm as ViT has it: a padded head pitch on A, a head slice on B, and a wrap that actually wraps because M is not a multiple of the row block.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

Q = 2
M = 37                 # not a multiple of BM: the wrap has to wrap
N = 64
K = 37
PITCH = K * M + 7      # the hole empty_strided leaves at the end of each head
LDB = 768              # the [S, H*D] row pitch B is sliced out of
BM, BN, BK = 32, 32, 16
GRID_M = (M + BM - 1) // BM


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def wrap_padded_batch_pitch(A, B, O, M: tl.constexpr, N: tl.constexpr,
                                K: tl.constexpr, PITCH: tl.constexpr,
                                LDB: tl.constexpr, BM: tl.constexpr,
                                BN: tl.constexpr, BK: tl.constexpr):
        GROUP_M: tl.constexpr = 8
        # A: row-major within a head, with a PADDED head pitch.
        stride_am, stride_ak, stride_aq = K, 1, PITCH
        # B: a per-head slice of one [S, H*D] buffer -- the batch stride sits
        # inside the row pitch.
        stride_bk, stride_bn, stride_bq = LDB, 1, N
        q = tl.program_id(1)
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
        # THE WRAP, and here it is not the identity: rm reaches 63 against
        # M = 37, so the tail block reads rows 0..26 a second time.
        ram = rm % M
        rk = tl.arange(0, BK)
        a = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak
                 + q * stride_aq)
        b = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn
                 + q * stride_bq)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            # K is not a multiple of BK either, so the k tail is masked rather
            # than wrapped -- the two tails are different mechanisms and this
            # kernel has both.
            kmask = (rk[None, :] + k) < K
            av = tl.load(a, kmask, other=0.0)
            bv = tl.load(b, (rk[:, None] + k) < K, other=0.0)
            acc += tl.dot(av, bv, allow_tf32=False)
            a += BK * stride_ak
            b += BK * stride_bk
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)
        tl.store(O + (idx_n + N * idx_m + M * N * q), acc, mask)

    return wrap_padded_batch_pitch


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(Q * PITCH, generator=g, dtype=torch.float32),
            "b": torch.randn(K * LDB, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    a, b = inputs["a"], inputs["b"]
    outs = []
    for q in range(Q):
        head = a[q * PITCH:q * PITCH + M * K].reshape(M, K)
        # B's head q is the N columns starting at q*N of a [K, LDB] buffer.
        bq = b.reshape(K, LDB)[:, q * N:(q + 1) * N]
        outs.append(head @ bq)
    return {"o": torch.stack(outs).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="wrap_padded_batch_pitch", kernel=kernel,
    signature={"A": "*fp32", "B": "*fp32", "O": "*fp32", "M": "constexpr",
               "N": "constexpr", "K": "constexpr", "PITCH": "constexpr",
               "LDB": "constexpr", "BM": "constexpr", "BN": "constexpr",
               "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "PITCH": PITCH, "LDB": LDB,
                "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "float32", (Q * PITCH,)),
          Arg("b", IN, "float32", (K * LDB,)),
          Arg("o", OUT, "float32", (Q * M * N,))],
    grid=(GRID_M * (N // BN), Q),
    reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect="blocked",
    expect_why="matmul K=16 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
