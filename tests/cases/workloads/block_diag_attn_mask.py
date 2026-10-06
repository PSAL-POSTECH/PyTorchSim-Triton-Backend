"""Pixtral's block-diagonal attention -- several images packed into ONE sequence,
and the block boundaries come out of a `cumsum` over per-image patch counts.

    ends   = cumsum(patch_counts)          [NI]
    seg[i] = #{ e in ends : e <= i }       which image token i belongs to
    keep   = seg[i] == seg[j]              attend only inside your own image

WHY THE IMAGES ARE IN ONE SEQUENCE AT ALL. Pixtral takes images at their native
aspect ratio and resolution, so every image in a batch has a DIFFERENT patch
count and there is no rectangle to pad them into without wasting most of it. The
vision encoder therefore concatenates all of them into a single flat run of
patches -- `patch_embeds_list` -> `torch.cat` -> one sequence -- and recovers the
separation with a mask instead of with a shape. `generate_block_attention_mask`
is exactly a cumsum of the list into start/end pairs and a zero written into
each `[start:end, start:end]` square.

THE MASK IS DATA, WHICH IS WHAT NOTHING ELSE HERE HAS. Every other mask in this
suite is a closed-form predicate on the two indices: `softmax_causal.py` is
`j <= i`, `sliding_window.py` is a band, `chunked_local_attn.py` is a chunk
equality. All three are decided by the tile's own geometry and a compiler could
in principle constant-fold them. This one is decided by a tensor that was
LOADED, so the kept region is not knowable at compile time and the mask's
producer is a scan followed by a reduction:

    a cumsum on the lane-free rank-1 count vector      -> block ends
    a compare of that against the token index, folded  -> a segment id per token
    the id compared against ITSELF broadcast both ways -> the block-diagonal

That last step is the interesting one: `seg[:, None] == seg[None, :]` broadcasts
ONE rank-1 tensor along both axes of the same tile, so the lane-axis assignment
sees the same value pinned to the lanes and pinned in-lane simultaneously. The
two-arange masks elsewhere broadcast two DIFFERENT tensors; this is one tensor
against a transposed copy of itself, and it is the shape a segment-id mask
always has (`flash_attn_varlen`, `block_diagonal_mask`, packed-sample training).

  The fold that builds `seg` is over the IMAGE axis, which is the inner axis of
  a [S, NI] tile -- NI is 4, a very short in-lane reduction next to a very long
  lane axis, the opposite aspect ratio from every reduction in reduce/.

WHAT IS PINNED AND WHY. The number of images is a constant: a data-dependent
count would be a data-dependent tile extent and there is no such thing. What
stays data is the counts themselves and therefore every boundary -- the blocks
are unequal, unaligned to any power of two, and different for every seed. That
is the half of "ragged" that a static-shape compiler can actually be asked to
handle, and pinning NI costs nothing because the block structure is unchanged.

  Not causal. Pixtral's vision encoder is bidirectional inside an image; the
  mask is symmetric and every row keeps a whole block, so no row is degenerate
  and the softmax cannot go 0/0.

torch equivalent:
    seg  = (cumsum(cnt)[None, :] <= arange(S)[:, None]).sum(1)
    p    = softmax(sc.masked_fill(seg[:, None] != seg[None, :], -inf), -1)

WHAT STOPS IT TODAY -- A UNIT ROW NOBODY ARRANGED TO REPLICATE. It used to reach
the suite as `KILLED by signal 6 (SIGABRT) -- no diagnostic was printed`. That
was two faults stacked: vectorize's `_mask_tail` built its keep-mask from a
list of PYTHON BOOLS, which DenseElementsAttr has no path for (`std::bad_cast`),
and the same pass drove `_nest` with a bare `__enter__` so the raise abandoned
its insertion points -- masking the real exception and aborting the interpreter
at shutdown. Both are fixed, and what the kernel actually needs is now readable:

    NotVectorisable: operand [1, 4] broadcasts along dim 0 into a tile
    [64, 4], and it was staged one element per lane -- every lane needs the
    whole row, which takes a replicating transfer this operand did not get

fit_layout_to_lanes is where that is decided -- `_stage_one_map_row` ranks the
four stagings a row can get, and `_clone_loaded_row_wide` is the free one:
a second transfer of the same DRAM with a zero stride on the broadcast
axis. It needs a DRAM address to re-read. Here the row is a `tensor.expand_shape` of the segment-id
vector -- COMPUTED, not loaded, so there is no descriptor to widen.
So the row is staged into ONE lane and vectorize refuses rather than read
another lane's bank. `_replicate_computed_row` is the answer that does not need
one -- a DRAM round trip with a zero stride -- and reaching it from here is the
fix; it is the same fix for
paged_attention, top_p_sampling and sort_row_tl, which fail on the same
sentence.

GREEN SINCE 2026-09-05, err 0, AND THE REFUSAL WAS RIGHT. The boundary row here
is a `memref.subview` of a 6-element allocation -- three overlapping windows at
offsets 0, 1 and 2 -- and `_replicate_computed_row` was handing that view to the
transfer emitter as the source of its round trip. An MVOUT moves an ALLOCATION,
with the allocation's sizes: a reassociation view is safe because its elements
are the same ones reshaped, and a subview is not, because it is fewer elements
at an offset. Moving the allocation would have moved the wrong window.

So the two halves of fit_layout_to_lanes disagreed about what a source may be,
and the minting half now materialises: a view the emitter cannot name is copied
into a buffer of its own first, by the same rewrite `_emit_cast_buffer` beside it
already does for a sub-word row, minus the cast. The test is the emitter's own
predicate, so they cannot drift apart again.

Pixtral: the only mask here whose kept region is LOADED rather than computed -- cumsum -> segment id -> one rank-1 tensor broadcast along both axes. `seg` is exact; a mismatch there is the scan.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S = 64          # patches, all images together
NI = 4          # images packed into that sequence -- pinned, see above
NEG = -1.0e30   # finite stand-in for -inf, as softmax_causal.py uses


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def block_diag_attn_mask_kernel(sc_ptr, cnt_ptr, p_ptr, seg_ptr,
                                    S: tl.constexpr, NI: tl.constexpr,
                                    NEG: tl.constexpr):
        i = tl.arange(0, S)
        j = tl.arange(0, S)
        e = tl.arange(0, NI)

        # BLOCK BOUNDARIES FROM DATA: a rank-1 scan over the patch counts.
        cnt = tl.load(cnt_ptr + e)
        ends = tl.cumsum(cnt, axis=0)

        # Which image each token is in: fold over the IMAGE axis, which is the
        # short in-lane axis of an [S, NI] tile.
        seg = tl.sum((ends[None, :] <= i[:, None]).to(tl.int32), axis=1)
        tl.store(seg_ptr + i, seg)

        # ONE rank-1 tensor broadcast along BOTH axes of the same tile.
        keep = seg[:, None] == seg[None, :]

        sc = tl.load(sc_ptr + i[:, None] * S + j[None, :])
        z = tl.where(keep, sc, NEG)
        m = tl.max(z, axis=1)
        ex = tl.where(keep, tl.exp(z - m[:, None]), 0.0)
        tl.store(p_ptr + i[:, None] * S + j[None, :],
                 ex / tl.sum(ex, axis=1)[:, None])

    return block_diag_attn_mask_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # A random composition of S into NI positive parts: NI-1 distinct cuts.
    cuts = torch.randperm(S - 1, generator=g)[:NI - 1].sort().values + 1
    b = torch.cat([torch.zeros(1, dtype=torch.long), cuts,
                   torch.tensor([S], dtype=torch.long)])
    cnt = (b[1:] - b[:-1]).to(torch.int32)
    return {"sc": torch.randn(S * S, generator=g, dtype=torch.float32),
            "cnt": cnt}


def reference(inputs):
    import torch
    sc = inputs["sc"].reshape(S, S)
    ends = inputs["cnt"].to(torch.int64).cumsum(0)
    i = torch.arange(S)
    seg = (ends[None, :] <= i[:, None]).to(torch.int32).sum(dim=1)
    keep = seg[:, None] == seg[None, :]
    z = torch.where(keep, sc, torch.full_like(sc, NEG))
    ex = torch.where(keep, torch.exp(z - z.max(dim=1, keepdim=True).values),
                     torch.zeros_like(z))
    p = ex / ex.sum(dim=1, keepdim=True)
    return {"p": p.reshape(-1).contiguous(), "seg": seg.to(torch.int32)}


SPEC = KernelSpec(
    name="block_diag_attn_mask_kernel",
    kernel=kernel,
    signature={"sc_ptr": "*fp32", "cnt_ptr": "*i32", "p_ptr": "*fp32",
               "seg_ptr": "*i32", "S": "constexpr", "NI": "constexpr",
               "NEG": "constexpr"},
    constexprs={"S": S, "NI": NI, "NEG": NEG},
    args=[Arg("sc", IN, "float32", (S * S,)),
          Arg("cnt", IN, "int32", (NI,)),
          Arg("p", OUT, "float32", (S * S,)),
          Arg("seg", OUT, "int32", (S,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
