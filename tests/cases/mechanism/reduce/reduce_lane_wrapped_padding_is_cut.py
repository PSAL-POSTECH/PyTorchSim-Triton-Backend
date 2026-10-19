"""A softmax of 8 rows of 384 in a 512 block: the columns ride the lanes, 4 laps a lane, and the last lap
is all mask (the form Inductor writes: a [1, B] mask, -inf and 0 selected in, the input loaded twice).
fit-to-numel cuts every buffer to the 3 laps the 384 columns need, so no walk runs the
pad. Breaks if a lane axis viewed as [lanes, laps], a broadcast copy or a reduce over a buffer written
twice keeps the whole block.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, B = 8, 384, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_lane_wrapped_padding_is_cut_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
        c = tl.arange(0, B)[None, :]
        mask = c < C
        at = tl.arange(0, R)[:, None] * C + c
        x = tl.load(x_ptr + at, mask, other=0.0)
        m = tl.max(tl.where(mask, x, float("-inf")), axis=1)[:, None]
        s = tl.sum(tl.where(mask, tl.exp(x - m), 0.0), axis=1)[:, None]
        y = tl.load(x_ptr + at, mask, other=0.0)
        tl.store(o_ptr + at, tl.exp(y - m) / s, mask)

    return reduce_lane_wrapped_padding_is_cut_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R, C, generator=g)}


def reference(inputs):
    import torch
    x = inputs["x"]
    return {"o": torch.softmax(x, 1)}


SPEC = KernelSpec(
    name="reduce_lane_wrapped_padding_is_cut_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr", "C": "constexpr", "B": "constexpr"},
    constexprs={"R": R, "C": C, "B": B},
    args=[Arg("x", IN, "float32", (R, C)), Arg("o", OUT, "float32", (R, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    # THE CUT IS THE CLAIM: no buffer keeps the 512-column block, the lane tiles hold 3 laps.
    # The 3-lap view: its expand_shape, the max's load and the sum's two loads (peel_reduction_init reads the
    # sum's first lap before its loop).
    expect_ir={"08-transformed.mlir": {"memref<512x": 0, "memref<128x1x3x8xf32": 4}},
)
