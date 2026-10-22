"""A rank-3 tile whose axes are PERMUTED every iteration and carried round.

WHAT IT IS FOR. loop_10 pins the rank-2 case: a carried value wanted on one axis
and produced on the other, which is a SWAP. A swap is its own inverse, so a pass
can be wrong about direction and still close the loop. A three-axis rotation is
not: `(1, 2, 0)` applied three times is the identity and applied once or twice is
not, so a pass that loses track of WHICH permutation it absorbed lands on the
wrong axis and the numbers say so.

THE SHAPE. Every axis is the same length, which is what lets the loop close at
all -- the carried value has to come back the shape it left:

    for t in range(T):
        acc = tl.trans(acc, 1, 2, 0) + w

`absorb_layout_ops` folds the permutation into the neighbours' maps, so what
reaches fit_layout_to_lanes is an elementwise add whose two operands are indexed
through DIFFERENT permutations of one iteration space, and a back edge saying the
result is the same axis class as the init.

WHAT IT PINS FOR THE WALK. The layout walk replaces a global search that rebuilds
the axis graph every round; a walk decides once, in order, and cannot revisit.
The three things it has to keep are the back edge, the stamp contract and buffer
sharing, and this kernel is the back edge at a rank the swap case cannot reach:
T=4 leaves a NET rotation of one, so an answer that quietly treats the
permutation as a swap comes back transposed rather than rotated.

T IS 4 AND NOT 3 ON PURPOSE. Three rotations of (1, 2, 0) are the identity, and a
kernel whose reference is the input is a kernel that passes with the loop body
deleted.

WHAT IT FOUND. It answered 3.02 out at T=2 and 0 at T=1, which is the value going
round the back edge once. The cause was not the back edge: `_reshape_to_relayout`
turned the permutation into a round trip correctly, and `_drop_flat_relayouts`
then erased that trip as "flat" because its two ends carry the same lane axis and
the same type -- both of which a permutation of a CUBE preserves while moving
every element. It said which permutation, in `torchsim.perm`, right beside the ends
being read. That check now asks, and the kernel refuses instead of lying.

WHAT IT MEASURES TODAY. No round trip: `_xlu_rebank` learned the rank-3 shape --
the transpose unit takes two axes, so a third is one move per index of the axis
it does not touch -- and the permutation now crosses the lanes on the crossbar,
EIGHT `torchsim.crosslane_move` and no relayout at all. Measured against the same
compile without that: zero moves, and the old refusal above.

It stops one step later. vectorize refuses a `linalg.transpose` that still
touches a scratchpad buffer after banking: nothing lowers it from there except
convert-linalg-to-loops, which reads a lane-banked buffer linearly. The move
carries the tile; what is left is the op that named the permutation.

WHAT WOULD BREAK IT. Anything that absorbs a permutation by remembering only
that one happened -- rank 2 has one non-identity permutation and rank 3 has five,
so "there was a transpose here" stops being enough at exactly this rank.

A rank-3 rotation carried across a loop. Rank 2 has one non-identity permutation, so a swap cannot be got wrong by direction; rank 3 has five, and T=4 leaves a net rotation of one so the wrong one shows in the numbers rather than cancelling.
"""

#: WHAT THE LOOP-CARRY PORT COST ON THE WAY IN, MEASURED 2026-09-23. With
#: `_BufferOf` resolving a loop body's block argument to the buffer the loop
#: carries, instead of colliding with the kernel's own `arg%d`, this kernel
#: stopped in `lower` for a while: a transposing trip is
#: built whose scratchpad side is `%arg10` itself, and lower_to_dma reads a stamp off the
#: ALLOCATION, not off a block argument.
#:
#: Resolving the argument to the loop's init in `_SpadValue` gives a WRONG
#: ANSWER, 1.96041: THIS LOOP DOES NOT CARRY ONE BUFFER. It starts at `%2` and
#: yields `%0`, so the argument names a different allocation on the first trip
#: than on the rest, and no single global stands for it.
#:
#: WHAT CLOSED IT, three statements, all upstream's shape:
#:   the carried group is ONE order -- init, body argument and WHAT THE BODY
#:   YIELDS, per loop and not per allocation (upstream resets the body's
#:   parameter layout to the body's RESULT layout, la.cc:861);
#:   the loop's own result is no help, standing behind no global, so the body's
#:   terminator names what it leaves;
#:   and `BuildCopies` does not repoint a BLOCK ARGUMENT onto a copy's global --
#:   the first trip would then read what the second wrote. Left unbuilt, the
#:   disagreement reaches `RewriteTransposes`, which moves the whole tile.

from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

S, T = 8, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def carried_rank3_permute_kernel(a_ptr, w_ptr, o_ptr,
                                     S: tl.constexpr, T: tl.constexpr):
        r = tl.arange(0, S)
        off = r[:, None, None] * S * S + r[None, :, None] * S + r[None, None, :]
        acc = tl.load(a_ptr + off)
        w = tl.load(w_ptr + off)
        for _ in range(T):
            acc = tl.trans(acc, 1, 2, 0) + w
        tl.store(o_ptr + off, acc)

    return carried_rank3_permute_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = 1.0 / (S ** 0.5)
    return {"a": torch.randn(S * S * S, generator=g, dtype=torch.float32) * s,
            "w": torch.randn(S * S * S, generator=g, dtype=torch.float32) * s}


def reference(inputs):
    acc = inputs["a"].reshape(S, S, S)
    w = inputs["w"].reshape(S, S, S)
    for _ in range(T):
        acc = acc.permute(1, 2, 0) + w
    return {"o": acc.reshape(-1)}


SPEC = KernelSpec(
    name="carried_rank3_permute_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "S": "constexpr", "T": "constexpr"},
    constexprs={"S": S, "T": T},
    args=[Arg("a", IN, "float32", (S * S * S,)),
          Arg("w", IN, "float32", (S * S * S,)),
          Arg("o", OUT, "float32", (S * S * S,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
    expect="open",
    expect_stop=Stop("triton-compile", Reason.TRITON_PASS_FAILED),
    expect_why="upstream remove-layout-conversions does not terminate on a loop carrying a "
               "rank-3 tile with a tt.trans in its body (28 min and counting); "
               "torchsim-anchor-layouts refuses it before handing it over",
)
