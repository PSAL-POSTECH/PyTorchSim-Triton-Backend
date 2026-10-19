"""A NaN-propagating row max (Inductor's form: tl.maximum with propagate_nan=ALL) of 8 rows of 1024:
the columns ride the lanes, 8 laps a lane; each lane takes the max of its laps and the lanes fold.
Some rows hold a NaN in only some lanes (the first lane, the last, one in between, several), others a
NaN in every lane, all NaN, no NaN, -0 with one +0, -inf only and NaN beside +inf. Breaks if a lane's
max or the fold across the lanes drops a NaN, or makes one where there was none.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C = 8, 1024
NAN, INF = float("nan"), float("inf")


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _max_nan(a, b):
        return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)

    @triton.jit
    def reduce_max_nan_in_some_lanes_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :])
        tl.store(o_ptr + tl.arange(0, R), tl.reduce(x, 1, _max_nan))

    return reduce_max_nan_in_some_lanes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(R, C, generator=g)
    x[0, 0] = NAN                       # the first lane only
    x[1, 7 * 128 + 127] = NAN           # the last lane only, on the last lap
    x[2, 3 * 128 + 64] = NAN
    x[2, 5 * 128 + 65] = NAN            # two lanes
    x[3, 128:256] = NAN                 # every lane
    x[4, :] = NAN                       # all NaN
    x[5, :] = -0.0
    x[5, 400] = 0.0                     # -0 and one +0
    x[6, :] = -INF                      # -inf only
    x[7, 9] = NAN
    x[7, 200] = INF                     # NaN beside +inf, in other lanes
    return {"x": x}


def reference(inputs):
    return {"o": inputs["x"].amax(1)}


SPEC = KernelSpec(
    name="reduce_max_nan_in_some_lanes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (R, C)), Arg("o", OUT, "float32", (R,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=0, atol=0,
    # THE VALUES ARE THE CLAIM: one fold across the lanes, NaN where torch has it and only there.
    expect_ir={"08-transformed.mlir": {"torchsim.move_kind = \"fold\"": 1}},
)
