"""A wrapped STORE whose walk lands twice on one row cannot be pieces.

    o[(p * 4) % 8] = v          p = arange(0, 4)  ->  rows 0, 4, 0, 4

WHAT IT PINS. Inside one tile the later lane wins, and that is the answer the
kernel asked for. The pieces reproduce the walk BETWEEN transfers, and two
transfers say nothing about which lands last -- so a store whose walk revisits
a row takes one address per element instead, where the order is the tile's.

    p        0   1   2   3
    address  0   4   0   4        rows 0 and 4 written twice
    runs     [0, 2) delta 0   |   [2, 4) delta -8
             both runs cover rows 0 and 4

A LOAD IS UNHARMED and is deliberately not refused: reading one row twice is
the same value twice, which is what the tile asked for. The load below has the
same walk and stays a pair of strided pieces.

WHAT WOULD BREAK IT. Comparing the runs' DRAM DELTAS instead of the walk. The
two deltas here are 0 and -8 -- different -- while the rows they reach are the
same, because the second run's lanes start 8 further along. The question is
about the address, not about the bookkeeping that produces it.

A store whose wrapped walk revisits a row: the later lane wins and two transfers cannot say which lands last, so it addresses per element. The load with the same walk keeps its pieces.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 4
W = 8
ST = 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_store_revisits_a_row_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                             W: tl.constexpr, ST: tl.constexpr):
        p = tl.arange(0, N)
        row = (p * ST) % W
        # The load has the same walk and keeps its pieces.
        v = tl.load(x_ptr + row)
        tl.store(o_ptr + row, v + p.to(tl.float32))

    return ptr_wrap_store_revisits_a_row_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(W, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    o = torch.zeros(W, dtype=torch.float32)
    for p in range(N):
        o[(p * ST) % W] = x[(p * ST) % W] + float(p)
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_wrap_store_revisits_a_row_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "W": "constexpr", "ST": "constexpr"},
    constexprs={"N": N, "W": W, "ST": ST},
    args=[Arg("x", IN, "float32", (W,)), Arg("o", OUT, "float32", (W,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
