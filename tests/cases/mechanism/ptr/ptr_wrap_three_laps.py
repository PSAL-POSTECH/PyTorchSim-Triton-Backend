"""A wrap the walk crosses TWICE: three strided runs, not two and not a gather.

    row = (i * 8) % 24        i = 0..7
    span = 7*8 = 56 > W = 24  so the walk laps the modulus three times
    rows  0 8 16 | 0 8 16 | 0 8            three runs, stride 8 in each

WHY THIS SHAPE. A ring read has at most one seam because its span fits the
modulus; a REPEAT does not, and the suite has 21 of them against 6 rings. The
run count is (o + (n-1)*st) // W + 1 and nothing caps it at two, so "two pieces"
was a limit of the old code, not of the arithmetic.

WHAT IT PINS: that the piece count follows the geometry. Without it the access
falls to a per-element gather -- correct, but it builds an index tile of eight
entries and moves one element per descriptor instead of three bursts.

Three laps of the modulus -> three strided transfers. The piece count is the geometry's, not a constant.
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
    def ptr_wrap_three_laps_kernel(x_ptr, o_ptr, W: tl.constexpr,
                                   N: tl.constexpr, ST: tl.constexpr):
        i = tl.arange(0, N)
        # Every lane is live, so all three runs are read for their values and a
        # dropped run shows up as a wrong number rather than as a refusal.
        row = (i * ST) % W
        v = tl.load(x_ptr + row)
        tl.store(o_ptr + i, v)

    return ptr_wrap_three_laps_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(W, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    i = torch.arange(N)
    return {"o": inputs["x"][(i * ST) % W]}


SPEC = KernelSpec(
    name="ptr_wrap_three_laps_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "W": "constexpr",
               "N": "constexpr", "ST": "constexpr"},
    constexprs={"W": W, "N": N, "ST": ST},
    args=[Arg("x", IN, "float32", (W,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE NUMBERS ARE RIGHT EITHER WAY -- a per-element gather reads the same
    # eight values -- so the claim lives in the IR: one stamped transfer and no
    # index buffer, rather than one `indirect` descriptor (rule 2).
    expect_ir={"04-transformed.mlir": {"cut": 1, "offset_stride": 0,
                                   "torchsim.transfer": 2}},
)
