"""An i32 row sum of 8 rows of 1024, which the lanes do not fold: the row's columns still go on the
lanes, each lane sums its 8 laps, and only the 8 x 128 partials leave the lanes. Breaks if a reduce the
lanes do not fold keeps its few rows on a few lanes again.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C = 8, 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_few_rows_int_sum_on_lanes_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :])
        tl.store(o_ptr + tl.arange(0, R), tl.sum(x, axis=1))

    return reduce_few_rows_int_sum_on_lanes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(-1000, 1000, (R, C), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    return {"o": x.sum(1, dtype=torch.int32)}


SPEC = KernelSpec(
    name="reduce_few_rows_int_sum_on_lanes_kernel", kernel=kernel,
    signature={"x_ptr": "*i32", "o_ptr": "*i32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "int32", (R, C)), Arg("o", OUT, "int32", (R,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=0, atol=0,
)
