"""A wrap on K, the axis a dot sums over: `rk % K` with K = 48 under a 64-wide block reads columns
0..15 twice, and both reads land in every product row. Nothing drops them, so the wrap stays
(values: C is A@B plus A[:, :16]@B[:16, :])."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 64, 64, 48
BLOCK_K = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mm_k_wrap(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BLOCK_K: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        rk = tl.arange(0, BLOCK_K) % K
        a = tl.load(A + rm[:, None] * K + rk[None, :])
        b = tl.load(B + rk[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, out_dtype=tl.float32)
        tl.store(C + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

    return mm_k_wrap


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g).to(torch.bfloat16),
            "b": torch.randn(K, N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    a, b = inputs["a"].float(), inputs["b"].float()
    w = BLOCK_K - K
    return {"c": a @ b + a[:, :w] @ b[:w, :]}


SPEC = KernelSpec(
    name="mm_k_wrap", kernel=kernel,
    signature={"A": "*bf16", "B": "*bf16", "C": "*fp32", "M": "constexpr", "N": "constexpr",
               "K": "constexpr", "BLOCK_K": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BLOCK_K": BLOCK_K},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)),
          Arg("c", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
)
