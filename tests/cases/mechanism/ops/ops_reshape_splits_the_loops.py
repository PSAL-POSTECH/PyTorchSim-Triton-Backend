"""A reshape whose reader has one operand on a SUBSET of the iteration dims.

WHAT IT PINS. `absorb_layout_ops` folds a reshape into a `linalg.generic` by
restating the one map the reshaped operand is read with -- and a map result
names an ITERATION axis, so an axis that became two has no such name and the
restatement refuses. That is true of ONE MAP and false of the OP: the name the
split axis lost was the name of a LOOP, and a loop belongs to no operand. It is
what every operand and every map agree on together, so the loop splits with the
axis and then every axis has a name again (`_iteration_takes`).

    ins  (d0, d1) -> (d0, d1)   over 128x16      the reshaped tile
    ins  (d0, d1) -> (d0)       over 128         the row sum, broadcast
    outs (d0, d1) -> (d0, d1)   over 128x16

    absorbed into the expand 128x16 -> 8x16x16:

    ins  (d0, d1, d2) -> (d0, d1, d2)
    ins  (d0, d1, d2) -> (d0, d1)
    outs (d0, d1, d2) -> (d0, d1, d2)

and the row sum, which is on a SUBSET of the dims, moves to 8x16 with them --
which is the whole point of the shape. Every kernel in the suite whose reshape
reaches a generic has all its operands on the same dims, and those are absorbed
by `_restate_elementwise` without anybody having to touch the loops. A reader
that broadcasts one operand is the case that cannot be, and `x / tl.sum(x, 1)`
is the most ordinary way to write one.

WHAT WOULD BREAK IT.

  * restating the reshaped operand's map alone: the other ins map still names
    two dims where the iteration space now has three, and linalg rejects it at
    verify -- "expected operand #1 rank (1) to match the result rank of
    indexing_map (2)" -- rather than silently
  * moving the loops and not the row sum: it would be indexed by a dim that is
    now two, which is the half-rewrite rule 4 forbids
  * `_reduce_input_move`'s split branch, which is what carries the move from
    the row sum BACK through the `linalg.reduce` that made it and on to the
    tile -- without it the group stops at the reduce and the reshape stands

THE REDUCE IS WHY THE SUM IS OVER THE INNER AXIS. Folding axis 1 of 128x16
leaves 128, and 128 is what the expand splits into 8x16 -- so the reduce's
INPUT takes a move too, and its surviving axes and its folded one have to be
told apart. A sum over axis 0 would leave 16, which the expand does not touch,
and the reduce would never be asked.

torch equivalent: y = x / x.sum(1, keepdim=True)

A tensor.expand_shape read by a generic whose other ins operand is on a subset of the iteration dims. Absorbing it splits the LOOPS, not just the one map, and carries the move back through the linalg.reduce that made the row sum.
"""

#: WHY THIS REFUSES, TRACED 2026-09-23. lower_to_dma stops at "the lane axis 1 of
#: @buf2_spad_128lane corresponds to no axis of this view", and the view is a
#: `memref.expand_shape` that SPLITS the axis the lanes ride: one source axis
#: becomes two, so no axis of the view carries the lanes and the walk back has
#: nothing to answer. `MaterialiseReshapes` would give such a view its own
#: allocation, but this one is WRITTEN (a fill and a reduce target it), and a
#: written result has no way home.
#:
#: SAYING IT AS A CONSTRAINT WAS TRIED AND REVERTED: "an axis a reshape splits
#: cannot ride", stated in AddBackendConstraints the way the fold and slice
#: facts are. It moves this kernel to a different refusal and costs the suite
#: 451/1 wrong answers -> 437/3. For a rank-1 source every axis is split, so the
#: constraint collapses to ONE_LANE and its neighbours then disagree -- the same
#: shape that made the ONE_LANE class before replication existed.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reshape_splits_the_loops_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                        N: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, N)
        x = tl.load(x_ptr + m[:, None] * N + n[None, :])     # [M, N]
        s = tl.sum(x, axis=1)                                # [M]
        # The divide is written on the RESHAPED tile, so the generic that does
        # it has an iteration space the reshape has to split, and one operand
        # (the row sum) on a subset of its dims.
        y = tl.reshape(x, (M // N, N, N)) / tl.reshape(s, (M // N, N))[:, :, None]
        tl.store(o_ptr + m[:, None] * N + n[None, :], tl.reshape(y, (M, N)))

    return reshape_splits_the_loops_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, N, generator=g, dtype=torch.float32)
    # No row may sum to zero: the reference divides by it and so does the
    # kernel, and a NaN on both sides is not a test of anything.
    return {"x": x + 4.0}


def reference(inputs):
    x = inputs["x"]
    return {"o": x / x.sum(dim=1, keepdim=True)}


SPEC = KernelSpec(
    name="reshape_splits_the_loops_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr",
               "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)),
          Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
