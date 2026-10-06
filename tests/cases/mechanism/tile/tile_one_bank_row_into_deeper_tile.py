"""A one-bank row replicated into a tile of HIGHER RANK than the row.

    row   memref<1xNxi32>   read `(d0,d1,d2,d3) -> (d2,d3)`
    tile  [M, 1, 1, N]      banked on d0, which the row does not name

WHAT THIS PINS. The row has to be present in every bank -- the same demand
`tile_one_bank_row_times_banked_tile.py` pins -- but here the SOURCE RANK AND
THE TILE RANK DIFFER. `_replicate_computed_row` chose `replicate_axes` by
matching shapes, and its three arms are rank-1 against `wide[keep]`, tile rank
with one non-unit axis, and all-unit. A rank-2 row against a rank-4 tile is
none of them, so the minter returned None, `_stage_one_map_row` ran out of
answers, and vectorize refused five passes later:

    NotVectorisable: one operand of this elementwise op is in a single bank
    (ONE_LANE) while another is banked across the lanes on iteration dim 0 of
    extent 128

The correspondence was in the operand's map the whole time -- `(d2,d3)` says
the source's two axes run along result axes 2 and 3 -- and `wide` is the
iteration space, so a map result IS the result axis. Reading it there rather
than reconstructing it from ranks is rule 5, and it subsumes all three arms.

WHY THE SHAPE IS THIS SHAPE. It is Qwen2-VL's, in the small. Inductor emits the
image-token count -- `(input_ids == image_token_id).sum()`, transformers' own
check that the placeholders match the embeddings -- as a reduction whose
parallel numel is 1 under an XBLOCK of 128, and the unit axes it carries make
the iteration space rank 4 while the loaded row stays rank 2. Qwen2-Audio's
audio-token count is the same shape.

WHAT WOULD BREAK THIS. Narrowing the minter back to shape matching, or ranking
the replicating relayout out of `_stage_one_map_row` without another rung
covering this. The suite has no other kernel where the two ranks differ: with
the fix reverted this file is the only one that moves.

Qwen2-VL's image-token count in the small: a rank-2 one-bank row replicated into a rank-4 tile. Refused by vectorize while _replicate_computed_row picked replicate_axes by matching shapes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, KEEP, MARK = 128, 32, 20, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def one_bank_row_into_deeper_tile_kernel(a_ptr, c_ptr, M: tl.constexpr,
                                             N: tl.constexpr, KEEP: tl.constexpr,
                                             MARK: tl.constexpr):
        # THE ROW IS RANK 2 AND THE TILE IS DEEPER. `[None, :]` keeps the load
        # two-dimensional, and the unit axes below put two more dims in the
        # iteration space -- which is the shape Inductor's degenerate-x
        # reduction arrives in.
        # EVERY LANE COMPUTES THE SAME SCALAR AND WRITES THE SAME ADDRESS,
        # which is what keeps the unit axes alive down to this pass. Inductor
        # emits exactly this when a reduction's parallel numel is 1 under a
        # 128-wide XBLOCK: the tile is [M, 1, 1, N] while the row it reads
        # stays [1, N], and it is that RANK GAP this file is about. Storing to
        # `c_ptr + rm` instead lets the unit extents fold away upstream, the
        # row arrives rank 1 against a rank-2 tile, and the minter's first arm
        # takes it -- which is a different code path and pins nothing here.
        rn = tl.arange(0, N)[None, :]
        # MASKED, AND THAT IS PART OF THE SHAPE. A plain load is staged by the
        # rung above the replicating one -- re-read the same DRAM with a zero
        # stride -- and that rung wants a single MVIN writer. A masked load is
        # written by the select instead, so it declines and the replicate is
        # the answer left, which is the state Qwen2-VL's count kernel arrives
        # in (`tl.load(..., r0_mask, other=0.0)`).
        row = tl.load(a_ptr + rn, mask=rn < KEEP, other=0)   # [1, N], one bank
        hit = (row == MARK).to(tl.int32)
        wide = tl.broadcast_to(hit, [M, N])          # every lane needs all of it
        got = tl.where(rn < KEEP, wide, 0)
        s = tl.sum(got, 1)[:, None]                  # [M, 1], the same count M times
        tl.store(c_ptr + tl.full([1, 1], 0, tl.int32).broadcast_to(M, 1), s, None)

    return one_bank_row_into_deeper_tile_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randint(0, 7, (N,), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    return {"c": (inputs["a"][:KEEP] == MARK).to(torch.int32).sum().reshape(1)}


SPEC = KernelSpec(
    name="one_bank_row_into_deeper_tile_kernel", kernel=kernel,
    signature={"a_ptr": "*i32", "c_ptr": "*i32", "M": "constexpr",
               "N": "constexpr", "KEEP": "constexpr", "MARK": "constexpr"},
    constexprs={"M": M, "N": N, "KEEP": KEEP, "MARK": MARK},
    args=[Arg("a", IN, "int32", (N,)),
          Arg("c", OUT, "int32", (1,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=0, atol=0,
    expect="pass",
)
