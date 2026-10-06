"""Row-wise sum with a masking select and a rank-preserving result.

tests/cases/smoke/reduce.py passes. So does either half of this on its own. Together they
do not: the lowering builds a `linalg.index` with `dim = 2` inside an op whose
iteration space has two loops, and the IR stops verifying.

    error: 'linalg.index' op expected dim (2) to be lower than the number of
           loops (2) of the enclosing LinalgOp
    note:  see current operation: %21 = "linalg.index"() <{dim = 2 : i64}> : () -> index

The two halves, measured separately against tests/cases/smoke/reduce.py:

    tl.where(m, x, 0) before the reduce           -> PASSES
    tl.sum(...)[:, None], storing rank 2          -> PASSES
    both                                          -> the error above

This is not a synthetic pairing. It is the shape Inductor emits for
`(a + b).sum(dim=1)`: a select applies the mask before reducing, and the result
keeps its rank so the store lines up with the parallel axis. Two PyTorchSim
tests reach it, `tests/ops/reduce/test_reduce.py` and `tests/ops/conv/test_pool.py`,
which is every reduction on that route.

Select-before-reduce plus a rank-preserving result. Either alone lowers; together they produce a linalg.index that outranks its enclosing LinalgOp.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 512          # rows total
N = 64           # columns -- the reduced axis, kept inside a lane
BM = 128         # rows per program == TORCHSIM_COMPILE_VECTORLANE_SIZE (dim 0 -> lanes)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_select_expand(x_ptr, o_ptr, N: tl.constexpr, BM: tl.constexpr,
                             BN: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.arange(0, BN)
        x = tl.load(x_ptr + rows[:, None] * N + cols[None, :])
        m = cols[None, :] < BN
        x = tl.where(m, x, 0)                    # half 1
        s = tl.sum(x, axis=1)[:, None]           # half 2
        tl.store(o_ptr + rows[:, None], s)

    return reduce_select_expand


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": inputs["x"].reshape(M, N).sum(dim=1)}


SPEC = KernelSpec(
    name="reduce_select_expand",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"N": N, "BM": BM, "BN": N},
    args=[
        Arg("x", IN, "float32", (M * N,)),
        Arg("o", OUT, "float32", (M,)),
    ],
    grid=(M // BM,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
