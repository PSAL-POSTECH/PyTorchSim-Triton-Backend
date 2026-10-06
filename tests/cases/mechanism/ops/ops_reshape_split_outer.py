"""`tl.reshape((16,) -> (2, 8))` -- the reassociation that MOVES DATA.

WHAT IT PINS
`fit_layout_to_lanes._splits_the_outer`. A reassociation is a free renumbering only
while it leaves the lane axis alone. This one does not: element k of a rank-1
tile lives in LANE k, so 16 elements sit one per lane across 16 lanes, and 2
rows of 8 sit across 2 lanes with 8 elements in each one's vector. Every element
changes lane. There is no instruction that moves a value between lanes, so the
only honest answer is a DRAM round trip -- and the only way to know that this
reshape needs one, when `[N] -> [1, N]` next door does not, is to read the
`reassociation` attribute and ask whether the outermost group carries more than
one real extent.

WHAT WOULD BREAK IT
The test used to be a NAME test:

    if o.name not in ("tensor.expand_shape", "tensor.collapse_shape"):
        return False

written when the pass ran above `bufferize`. Below it every reassociation is a
`memref.expand_shape` or a `memref.collapse_shape`, so that tuple named nothing
and the function answered False for every reassociation in every function --
measured 21929 evaluations across 240 kernels, all False. The split then rode
down to a lane axis chosen as though it were free, and the tile comes back
scrambled with nothing anywhere to say so. Restore the name test and this kernel
is what notices.

Deliberately the SMALLEST shape that does it: one load, one reshape out, one
reshape back, one store. `mrs`/`d`/`f` in ops_shape.py exercise the same arm but
alongside a transpose, a join, a split and a cat, so a failure there names five
things at once.

MEASURED
  with the fix     ALL STAGES PASSED, max_abs_err = 0
  without it       stage 5, `the transfer's scratchpad side has no recorded
                   lane axis` on the MVOUT of the un-round-tripped
                   `collapse_shape`. So the
                   inversion is not only a silent loss of a capability here --
                   it is a compile failure two stages later, with nothing
                   pointing back at the pass that decided it.

THE COST, COUNTED (rule 8b). Zero round trips today; two when this was written,
one per reassociation, neither a unit-axis copy and each crossing lanes for
real. Both reshapes are now absorbed in absorb_layout_ops -- 2 in that pass's
input, 0 in its output -- so `(16,) -> (2,8) -> (16,)` composes to nothing and
costs nothing, which is the ideal count this paragraph used to name as
unreached.

WHICH LEAVES THIS FILE COVERING LESS THAN IT SAYS (rule 2). The arm it was
written for is in fit_layout_to_lanes and this kernel no longer reaches it: the
reshapes never arrive there. `_splits_the_outer`, named above, is now
`_splits_the_lane_axis`. ops_reshape_splits_the_lane_axis.py holds that arm --
the same reassociation over a LOOP-CARRIED value, which is what keeps it alive
past absorb_layout_ops. What this file still pins is the numbers.

Pins _splits_the_outer reading `reassociation` instead of matching tensor.expand_shape/collapse_shape by name. Two DRAM round trips, both crossing lanes for real.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 16
ROWS = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_reshape_split_outer_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                       ROWS: tl.constexpr):
        offs = tl.arange(0, N)
        v = tl.load(x_ptr + offs)
        # (16,) -> (2, 8): reassociation [[0, 1]], group 0 carries 2 AND 8.
        m = tl.reshape(v * 2.0, (ROWS, N // ROWS))
        # and back, which is the same statement collapsing.
        r = tl.reshape(m + 1.0, (N,))
        tl.store(o_ptr + offs, r)

    return ops_reshape_split_outer_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"]
    return {"o": ((x * 2.0).reshape(ROWS, N // ROWS) + 1.0).reshape(N)}


SPEC = KernelSpec(
    name="ops_reshape_split_outer_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "N": "constexpr", "ROWS": "constexpr"},
    constexprs={"N": N, "ROWS": ROWS},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
