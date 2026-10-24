"""tl.max and tl.min (propagate_nan=NONE: arith.maxnumf/minnumf) of 8 rows of 128: the columns ride the lanes,
one a lane, so a NaN reaches the fold across the lanes as it is. Breaks if that fold returns the NaN (the unit's
max and min propagate it) rather than dropping it: before the fix row 0's max came back -inf.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C = 8, 128
NAN, INF = float("nan"), float("inf")


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_maxnum_nan_in_some_lanes_kernel(x_ptr, o_ptr, p_ptr, R: tl.constexpr, C: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :])
        tl.store(o_ptr + tl.arange(0, R), tl.max(x, axis=1))
        tl.store(p_ptr + tl.arange(0, R), tl.min(x, axis=1))

    return reduce_maxnum_nan_in_some_lanes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(R, C, generator=g)
    x[0, 5] = NAN                       # one lane
    x[1, 0] = NAN
    x[1, C - 1] = NAN                   # the first and the last lane
    x[2, :] = NAN
    x[2, 77] = 1.5                      # every lane but one
    x[3, ::3] = NAN                     # every third lane
    x[4, :] = -INF
    x[4, 3] = NAN                       # -inf beside a NaN
    x[5, 9] = NAN
    x[5, 100] = INF
    x[5, 101] = -INF                    # a NaN beside +inf and -inf
    x[6, :64] = -0.0
    x[6, 64:] = NAN                     # -0 beside half the lanes NaN
    return {"x": x}


def reference(inputs):
    import torch
    x = inputs["x"]
    return {"o": torch.where(x.isnan(), -INF, x).amax(1), "p": torch.where(x.isnan(), INF, x).amin(1)}


SPEC = KernelSpec(
    name="reduce_maxnum_nan_in_some_lanes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "p_ptr": "*fp32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (R, C)), Arg("o", OUT, "float32", (R,)), Arg("p", OUT, "float32", (R,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=0, atol=0,
    expect_ir={"08-transformed.mlir": {"pytorchsim.fold": 2}},
)
