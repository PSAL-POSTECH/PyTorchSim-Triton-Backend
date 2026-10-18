"""Inductor's mm template with a block that does not divide M: `ram = rm % M` wraps rows M.. of the
last tile onto rows 0.., the A load is masked only on K, and only the store says `rm < M`. The rows a
wrap moves reach C's rows through the dot and are dropped by the store mask, so the wrap is dropped
and `rm < M` bounds the A load: A is a descriptor, not a gather (structure), and C is right (values)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 100, 64, 64
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mm_row_wrap(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        A = A + (ram[:, None] * K + rk[None, :])
        B = B + (rk[:, None] * N + rbn[None, :])
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(K, 0, -BLOCK_K):
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
            acc += tl.dot(a, b, out_dtype=tl.float32)
            A += BLOCK_K
            B += BLOCK_K * N
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)
        tl.store(C + (idx_m * N + idx_n), acc.to(tl.bfloat16), mask)

    return mm_row_wrap


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g).to(torch.bfloat16),
            "b": torch.randn(K, N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    return {"c": (inputs["a"].float() @ inputs["b"].float()).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="mm_row_wrap", kernel=kernel,
    signature={"A": "*bf16", "B": "*bf16", "C": "*bf16", "M": "constexpr", "N": "constexpr",
               "K": "constexpr", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr",
               "BLOCK_K": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)),
          Arg("c", OUT, "bfloat16", (M, N))],
    grid=(-(-M // BLOCK_M), N // BLOCK_N), reference=reference, make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
    # Both wraps gone: no operand moves through a per-row offset table.
    expect_ir={"08-transformed.mlir": {"offset_stride": 0}},
)
