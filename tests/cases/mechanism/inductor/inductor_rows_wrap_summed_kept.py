"""A row wrap a reduce sums over: `rm % M` with M = 40 under a 64-row block reads rows 0..23 twice,
and the column sums count both reads. The store's `rn < N` bounds another axis, so the wrap stays
(values: each column's sum over rows 0..39 plus rows 0..23)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 40, 64
BLOCK_M = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rows_wrap_summed(X, Y, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr):
        rm = tl.arange(0, BLOCK_M)
        rn = tl.arange(0, N)
        x = tl.load(X + (rm % M)[:, None] * N + rn[None, :])
        tl.store(Y + rn, tl.sum(x, axis=0), mask=rn < N)

    return rows_wrap_summed


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"]
    return {"y": x.sum(0) + x[:BLOCK_M - M].sum(0)}


SPEC = KernelSpec(
    name="rows_wrap_summed", kernel=kernel,
    signature={"X": "*fp32", "Y": "*fp32", "M": "constexpr", "N": "constexpr",
               "BLOCK_M": "constexpr"},
    constexprs={"M": M, "N": N, "BLOCK_M": BLOCK_M},
    args=[Arg("x", IN, "float32", (M, N)), Arg("y", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
)
