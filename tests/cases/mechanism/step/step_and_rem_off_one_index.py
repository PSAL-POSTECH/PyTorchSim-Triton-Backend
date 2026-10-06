"""GRID: `x % W` and `x // W` off ONE index, both reaching one address.

    row = nhw // OUT_H        picks the image
    col = nhw %  OUT_H        picks the row inside it
    addr = row*SR + col*SC

THE TWO HALVES ARE ONE INDEX SEEN TWICE, and they land in different terms of
the same address on the same tile axis. An axis has one index function to
offer, so this has to be refused or handled together -- never half-taken.

WHAT IT FOUND. The answer used to be WRONG here -- 3.9 against the reference,
on the plain `--triton-to-structured` with nothing else on. The two halves are
added by two sequential `tt.addptr` ops rather than by one `addi`, and
visitOperandAddptr was the one merge point that did not call
parkWrapOnTheGatheredAxis. visitOperandAdd and visitOperandMul both did. So a
`p % W` added to an already-gathered pointer lost its modulo instead of moving
it into the index, and the address came out one term short.

That is the ordinary spelling of a two-term address, which is why the defect
reached a workload. It was invisible until this file existed because the shape
needs BOTH halves of one index in one address, and nothing smaller has that.

One index seen as `// W` and `% W` in one address. The shape that found the missing park in visitOperandAddptr.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, W = 32, 8, 11
SR, SC = 64, 3
NX = ((M - 1) // W) * SR + (W - 1) * SC + N + 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def step_and_rem_off_one_index_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                          N: tl.constexpr, W: tl.constexpr,
                                          SR: tl.constexpr, SC: tl.constexpr):
        nhw = tl.arange(0, M)[:, None]
        c = tl.arange(0, N)[None, :]
        row = nhw // W
        col = nhw % W
        v = tl.load(x_ptr + row * SR + col * SC + c)
        tl.store(o_ptr + nhw * N + c, v)

    return step_and_rem_off_one_index_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    nhw = torch.arange(M)[:, None]
    c = torch.arange(N)[None, :]
    idx = (nhw // W) * SR + (nhw % W) * SC + c
    return {"o": inputs["x"][idx].reshape(-1)}


SPEC = KernelSpec(
    name="step_and_rem_off_one_index_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr",
               "N": "constexpr", "W": "constexpr", "SR": "constexpr",
               "SC": "constexpr"},
    constexprs={"M": M, "N": N, "W": W, "SR": SR, "SC": SC},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
