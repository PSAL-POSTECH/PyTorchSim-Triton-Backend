"""`tl.reshape((32,) -> (2, 16))` on a LOOP-CARRIED value -- the reassociation
that moves data, kept alive long enough for the pass that prices it to see it.

WHAT IT PINS
`fit_layout_to_lanes._splits_the_lane_axis`. A reassociation is a free
renumbering only while the lane axis is alone in its group. Element k of a
rank-1 tile lives in LANE k, so 32 elements sit one per lane; 2 rows of 16 sit
across 2 lanes with 16 elements in each one's vector. Every element changes
lane, no instruction moves a value between lanes, and the only honest answer is
a DRAM round trip. `[N] -> [1, N]` next door is free and must stay free, so the
question the pass asks is about the GROUP's non-unit extents, not the rank.

WHY THE LOOP, MEASURED BY LEAVE-ONE-OUT. Written without the `scf.for` the same
reshape is ABSORBED in absorb_layout_ops and never reaches this pass at all --
0 round trips, and the kernel passes without exercising anything. The loop is
what keeps it alive: a layout op on a loop RESULT cannot be pushed back into its
producer, because init, iter_arg, yield operand and result are ONE type and a
walk over values from outside the region reaches only two of the four. So
`_plan_def` refuses, no other plan fits, and the reshape stands.

The second reader is NOT load-bearing: dropping the rank-1 store and keeping
only the row sums still reaches the arm.

WHY IT EXISTS. ops_reshape_split_outer.py was written for this arm and stopped
reaching it once absorb_layout_ops learned to take both of its reshapes; it now
passes on 0 round trips. This file is the replacement, and the shape is the one
workloads/chunked_dpo_loss.py reaches the arm with -- `memref<32xf32>` expanded
to `memref<2x16xf32>`.

Reaches fit_layout_to_lanes._splits_the_lane_axis, which ops_reshape_split_outer.py stopped reaching. One DRAM round trip: the reassociation regroups the axis the lane rides.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, ROWS, STEPS = 32, 2, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_reshape_splits_the_lane_axis_kernel(x_ptr, s_ptr, N: tl.constexpr,
                                                ROWS: tl.constexpr,
                                                STEPS: tl.constexpr):
        offs = tl.arange(0, N)
        # CARRIED, so the reshape below reads a loop RESULT and cannot be
        # pushed back into it -- four slots of one type, two of them reachable.
        v = tl.zeros((N,), dtype=tl.float32)
        for i in range(0, STEPS):
            v = v + tl.load(x_ptr + offs) * (i + 1)
        # (32,) -> (2, 16): reassociation [[0, 1]], group 0 carries 2 AND 16.
        m = tl.reshape(v, (ROWS, N // ROWS))
        tl.store(s_ptr + tl.arange(0, ROWS), tl.sum(m, axis=1))

    return ops_reshape_splits_the_lane_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    v = inputs["x"] * sum(i + 1 for i in range(STEPS))
    return {"s": v.reshape(ROWS, N // ROWS).sum(dim=1)}


SPEC = KernelSpec(
    name="ops_reshape_splits_the_lane_axis_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "s_ptr": "*fp32", "N": "constexpr",
               "ROWS": "constexpr", "STEPS": "constexpr"},
    constexprs={"N": N, "ROWS": ROWS, "STEPS": STEPS},
    args=[Arg("x", IN, "float32", (N,)), Arg("s", OUT, "float32", (ROWS,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
