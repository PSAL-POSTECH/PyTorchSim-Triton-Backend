"""A turn group with a `memref.expand_shape` in it -- the turn declines.

WHAT THIS PINS. `fit_layout_to_lanes` step 3 offers every group to `_turn_group`,
which rewrites a whole group onto the other axis for free: it reverses the
descriptors, swaps a matmul's operands, renumbers a reduce's `dimensions` and
retypes every value in the group. It is all or nothing, so a member it cannot
rewrite makes the whole group decline -- and a reassociation view is such a
member.

MEASURED, over the suite's pre-fit_layout_to_lanes IR: `_turn_group` is offered 19 groups and
declines 19, and the reason it names is

    11  memref.expand_shape        chunked_dpo_loss, clip_contrastive_logits x2,
     2  memref.collapse_shape      flash_decode_splitk, gated_delta_rule,
                                   mel_logspec, ops_reshape_split_outer,
                                   ops_shape, qk_norm x2, top_p_sampling,
                                   reduce_captured_scalar, reduce_rank0_two_axis
     2  torchsim.transfer not plain rank-2
     1  linalg.transpose
     1  memref.subview
     1  memref.reshape
     1  linalg.reduce over a rank-3 input

so 13 of 19 are one missing arm. `_turn_group`'s own docstring records that a
reassociation arm WAS there and went with the tensor spellings when this pass
moved below bufferize -- it reversed `static_output_shape` and the groups of a
`tensor.expand_shape`. The memref form that replaced it was never given one, and
the note under it says why that is not a one-liner: a buffer-form view has
readers whose types the turn cannot retype, so turning half of it would be the
half-rewrite rule 4 is about.

WHAT WOULD BREAK THIS. Teaching `_turn_group` the reassociation arm, or removing
the decline. Either way this kernel's group becomes turnable and the note above
stops being true.

expect="pass" -- and IT DOES NOT COMPILE TODAY, which is a second thing this file
now holds down. Declining a turn is not an error: the group falls through to step
4 and pays a round trip instead of taking the free move, so what this file was
written to pin is not a crash, it is the cost -- watch the trip count in
`regress.py`, which is the number rule 8b makes the score.

WHAT STOPS IT NOW. With `--linalg-fold-unit-extent-dims` on, the `[1, N]` above
is no longer materialised as a tile: `row` stays a rank-1 buffer read
`(d0,d1)->(d1)`, fit_layout_to_lanes stamps it ONE_LANE and the tile it multiplies
stays banked on d0, so vectorize is handed an op with one operand in a
single bank and another spread over sixteen. It answered that by building the
vlane loop with a trip count of ONE and computing the whole kernel on lane 0 --
`verify: numerical mismatch 10.3251`, 31 of 32 elements, with nothing reported.
It now stops there by name; `tests/cases/mechanism/tile/
tile_one_bank_row_times_banked_tile.py` pins that refusal in the small. The fix
is a replicating MVIN minted by fit_layout_to_lanes, and this file goes green again
when that lands.

A reassociation view inside a turn group. _turn_group has no arm for memref.expand_shape, so the group declines the free move and step 4 pays a round trip instead. 13 of the suite's 19 turn groups decline for exactly this reason.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 16, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def turn_through_expand_kernel(a_ptr, r_ptr, c_ptr,
                                   M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        a = tl.load(a_ptr + rm[:, None] * N + rn[None, :])
        # A ROW THAT BECOMES A TILE THROUGH A RESHAPE. Upstream folds the
        # `[N] -> [1, N]` into a `memref.expand_shape`, and that view lands in
        # the same turn group as the tile it is combined with.
        row = tl.load(r_ptr + rn)
        wide = tl.reshape(row, (1, N))
        b = a * wide
        # Two reductions on opposite axes over the same tile: the shape step 3
        # exists for. One of them has to move, and a turn is the free way.
        s0 = tl.sum(b, axis=0)          # [N]
        s1 = tl.sum(b, axis=1)          # [M]
        tl.store(c_ptr + rn, s0)
        tl.store(c_ptr + N + rm, s1)

    return turn_through_expand_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M * N, generator=g, dtype=torch.float32),
            "r": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    a = inputs["a"].reshape(M, N)
    b = a * inputs["r"].reshape(1, N)
    return {"c": torch.cat([b.sum(dim=0), b.sum(dim=1)]).contiguous()}


SPEC = KernelSpec(
    name="turn_through_expand_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "r_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("a", IN, "float32", (M * N,)),
          Arg("r", IN, "float32", (N,)),
          Arg("c", OUT, "float32", (M + N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
