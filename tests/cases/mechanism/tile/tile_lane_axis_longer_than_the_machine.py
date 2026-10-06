"""A lane axis LONGER than the machine, reduced across it and broadcast back.

    x    memref<512xf32>   banked on axis 0 -- 512 positions over 256 lanes
    s    tl.sum(x)         reduced ACROSS the lanes, to rank 0 in one bank
    out  x - s             the scalar back onto all 512 positions

WHAT THIS PINS. 512 is twice the machine's 256 lanes, so a lane holds TWO
positions of that axis side by side (the lane axis is the slowest -- see
`tile_strides`), and both crossbar operations here have to run more than once:

    the reduce      the unit combines the lanes at EACH of the two positions, so
                    it hands back two partial sums and the lane adds them
                    (`folds` and `_fold_in_lane` in lower_to_vcix)
    the broadcast   one round per lane-full, because the unit gives every lane
                    ONE copy and there are two positions to fill

Before `An axis longer than the machine is several rounds of it` both said
`cannot be lowered here (the shapes are not the unit's)` and the kernel stopped
in stage 5. The 128-wide kernels beside it never reach it: one round covers them.

WHAT WOULD BREAK THIS. Taking `s_ext > lanes` or `d_ext > lanes` back to a
refusal, or folding the unit's partial answers with the wrong operation --
`_fold_in_lane` has to do what the crossbar did, a max of partial maxima and not
a sum of them.

512 over 256 lanes: two positions per lane, so the crossbar runs twice and the lane finishes the reduction itself.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def lane_axis_longer_than_the_machine(x_ptr, o_ptr, N: tl.constexpr):
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rn)
        # ACROSS THE LANES, and there are more positions than lanes.
        s = tl.sum(x, axis=0)
        tl.store(o_ptr + rn, x - s)

    return lane_axis_longer_than_the_machine


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"]
    return {"o": (x - x.sum()).contiguous()}


SPEC = KernelSpec(
    name="lane_axis_longer_than_the_machine", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
