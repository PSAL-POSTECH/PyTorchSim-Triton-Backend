"""A WRAPPED STORE whose laps cannot be counted: the indirect MVOUT.

`o_ptr + (rows % M)` with `rows = pid * BM + arange(BM)`, so the wrap start is
runtime and _wrap_cut cannot say where the laps fall (ptr_wrap_stacked.py is
the three ways that happens). The store therefore takes the same fallback a
load does -- one indirect transfer over an address tile:

    {direction = "store", dram_stride = [0, 1], indirect, offset_stride = 1}

WHY THIS ONE IS WORTH KEEPING SEPARATELY FROM THE LOAD CASES
The write side is a different path, not a mirror: a load fills the masked-off
positions, a store must not write them, and rows 0..63 here are written TWICE
across the two programs so the last write has to win. GRID is 2 and M is 192
against 256 rows, which is what makes the duplicate happen.

IT USED TO BE A NEGATIVE KERNEL. StoreConverter had no wraparound branch at all
(store-no-wraparound-handling), so --structured-to-memref failed on this kernel
and the FAILURE was the observable. That pass is deleted (rule 21) and the
access lowers; the numeric check below is now the observable instead.

A wrapped store with a runtime start -> indirect MVOUT. Rows 0..63 are written by both programs; the last write wins.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 192
N = 128
BM = 128
GRID = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def structured_store_wrap_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                     N: tl.constexpr, BM: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        cols = tl.arange(0, N)
        v = tl.load(x_ptr + rows[:, None] * N + cols[None, :])
        # STORE through the wrapped pointer -- StoreConverter has no branch for it
        tl.store(o_ptr + (rows % M)[:, None] * N + cols[None, :], v)

    return structured_store_wrap_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BM * GRID * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    # Programs run in increasing pid order and rows 0..63 are written twice
    # (by pid 0 as rows 0..63 and again by pid 1 as rows 192..255 % 192), so
    # the last write wins. The explicit loop states that instead of relying on
    # torch's documented-nondeterministic duplicate-index assignment.
    x = inputs["x"].reshape(BM * GRID, N)
    o = torch.zeros(M, N, dtype=torch.float32)
    for r in range(BM * GRID):
        o[r % M, :] = x[r, :]
    return {"o": o.reshape(-1)}


SPEC = KernelSpec(
    name="structured_store_wrap_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "BM": "constexpr"},
    constexprs={"M": M, "N": N, "BM": BM},
    args=[
        Arg("x", IN, "float32", (BM * GRID * N,)),
        Arg("o", OUT, "float32", (M * N,)),
    ],
    grid=(GRID,),
    reference=reference,
    make_inputs=make_inputs,
)
