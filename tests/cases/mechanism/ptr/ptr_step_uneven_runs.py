"""`p // K` where K does NOT divide the extent: the runs are uneven, so the
axis cannot split -- the step slot carries the period instead.

    n = 32, K = 6  ->  [0,6) [6,12) [12,18) [18,24) [24,30) [30,32)
                       six runs, the last one short

A uniform split needs K | n (see ptr_div_only_axis_split.py); rounding the tile
up to [6, 6] reads 36 elements for a 32-element value, and the valid region --
`outer < 5` plus `outer == 5 and inner < 2` -- is not a box, so no mask can cut
it back. `tts.make_tptr` therefore carries `static_step`, one period per axis,
and p06 derives the runs the way it already does from a wrap.

WHY ONE NUMBER IS ENOUGH. The op says the PERIOD, not the runs. `shape` says a
wrap the same way and _wrap_cut turns it into uneven laps; _step_cut is that
function for a step, and both feed the same pieces in p16.

A STEPPED AXIS WALKS NOWHERE INSIDE A RUN, which is the one thing that differs:
its dram_stride is 0 and each run carries `j * stride` as its own delta.

An uneven step: six runs, the last short. The op carries the period; p06 derives the runs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K = 32, 6
NX = (N + K - 1) // K          # every row the quotient can name


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_step_uneven_runs_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                    K: tl.constexpr):
        p = tl.arange(0, N)
        tl.store(o_ptr + p, tl.load(x_ptr + p // K))

    return ptr_step_uneven_runs_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": inputs["x"][torch.arange(N) // K]}


SPEC = KernelSpec(
    name="ptr_step_uneven_runs_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "K": "constexpr"},
    constexprs={"N": N, "K": K},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
