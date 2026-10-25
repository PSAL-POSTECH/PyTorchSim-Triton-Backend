"""A wrap the walk crosses twice, `(i * 8) % 24`: three runs, and each is what the TOG is told.

The descriptor walks the whole axis and keeps one run in a box, so the three runs are three DMAs.
Each used to reach the TOG as the one cut transfer they come from, and the TOG timed one walk of
the whole tile (togcheck noted it, "cut into laps and the TOG carries the whole box"). Now
fit-transfers-to-descriptor writes each run as its own transfer, a `piece_box` and the DRAM its
first element is at, in 09-custom.mlir -- the IR the TOG, togcheck and lower-dma all read.
Breaks if a cut reaches 09-custom.mlir again, or the pieces disagree with what Spike moves.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

W = 24          # the modulus
N = 8           # rows read
ST = 8          # the walk's stride, so the span is 56 -- more than two laps


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_runs_are_what_the_tog_reads_kernel(x_ptr, o_ptr, W: tl.constexpr,
                                   N: tl.constexpr, ST: tl.constexpr):
        i = tl.arange(0, N)
        row = (i * ST) % W
        v = tl.load(x_ptr + row)
        tl.store(o_ptr + i, v)

    return ptr_wrap_runs_are_what_the_tog_reads_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(W, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    i = torch.arange(N)
    return {"o": inputs["x"][(i * ST) % W]}


SPEC = KernelSpec(
    name="ptr_wrap_runs_are_what_the_tog_reads_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "W": "constexpr",
               "N": "constexpr", "ST": "constexpr"},
    constexprs={"W": W, "N": N, "ST": ST},
    args=[Arg("x", IN, "float32", (W,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"09-custom.mlir": {"cut": 0, "piece_box": 3, "pytorchsim.transfer": 4}},
    expect_stage="togcheck",
)
