"""Top-p (nucleus) sampling -- the cutoff that depends on the DISTRIBUTION.

torch equivalent, per row:
    p          = softmax(logits)
    sp, idx    = p.sort(descending=True)
    keep       = (sp.cumsum(0) - sp) < P          # exclusive cumsum
    out        = (sp * keep) / (sp * keep).sum()  # renormalised, sorted order

WHY IT IS NOT TOP-K
  Top-k keeps a fixed count. Top-p keeps however many it takes to reach mass P,
  so the number of survivors is DATA-DEPENDENT and cannot be a constexpr. Every
  current sampler ships both and most default to top-p, so a backend that can do
  top-k and not this can still not decode.

WHAT IT ASKS OF THE BACKEND, IN ORDER
  A descending sort (sort_row_tl.py), an exclusive cumsum along the same axis
  (ops_cumsum.py), a compare against a scalar, and a second reduction to
  renormalise. The chain is the point: sort feeds cumsum feeds mask feeds sum,
  all on one row, and every step has to stay in the lane the row lives in.

THE EXCLUSIVE CUMSUM IS LOAD-BEARING. Inclusive keeps the token that crosses P
and drops the one before it; exclusive keeps the crossing token, which is what
"smallest set whose mass exceeds P" means. Off by one here changes which tokens
can be sampled, quietly.

NO INDICES, AND THAT IS A FINDING. A sampler needs the permutation to map the
kept prefix back to token ids, and triton 3.6 has `tl.sort` but no `tl.argsort`
-- so the permutation has to come from a rank computation, which is an N x N
compare, which is the rank-3 broadcast sort_radix.py already showed this backend
refuses. The probabilities alone are the half that can be asked for today.

WHAT STOPS IT TODAY -- A UNIT ROW NOBODY ARRANGED TO REPLICATE. It used to reach
the suite as `KILLED by signal 6 (SIGABRT) -- no diagnostic was printed`. That
was two faults stacked: vectorize's `_mask_tail` built its keep-mask from a
list of PYTHON BOOLS, which DenseElementsAttr has no path for (`std::bad_cast`),
and the same pass drove `_nest` with a bare `__enter__` so the raise abandoned
its insertion points -- masking the real exception and aborting the interpreter
at shutdown. Both are fixed, and what the kernel actually needs is now readable:

    NotVectorisable: operand [1, ..., 2, 1] broadcasts along dim 0 into a tile
    [2, ..., 2], and it was staged one element per lane -- every lane needs the
    whole row, which takes a replicating transfer this operand did not get

fit_layout_to_lanes is where that is decided -- `_stage_one_map_row` ranks the
four stagings a row can get, and `_clone_loaded_row_wide` is the free one:
a second transfer of the same DRAM with a zero stride on the broadcast
axis. It needs a DRAM address to re-read. Here the unit rows are the two compare-exchange stride masks the bitonic
network builds per stage -- computed from `linalg.index`, never loaded.
So the row is staged into ONE lane and vectorize refuses rather than read
another lane's bank. `_replicate_computed_row` is the answer that does not need
one -- a DRAM round trip with a zero stride -- and reaching it from here is the
fix; it is the same fix for
block_diag_attn_mask, paged_attention and sort_row_tl.

GREEN SINCE 2026-09-04. `tl.sort` reshapes its tile into log2(n) axes of extent 2
so a compare-exchange at stride 2^k can NAME its partner, and the reassociation
says which axes are which -- for a [128, 16] tile that is `[[0..6], [7..10]]`,
seven spelling the rows and four the columns. absorb_layout could only take a
reshape whole: forward (readers see [128, 16]) is refused for a real reason, since
merging the column run takes away the ability to name axis 3 alone, which is what
the network is FOR. So backward won, the tile stayed rank 11 to the end,
select_lane_axis had eleven interchangeable axes of 2 to choose between, and the
DMA descriptor -- which holds four dims -- peeled the lane axis into an
`affine.for`, where an outer loop iterates the axis the DMA is meant to spread.

The measurement that unlocked it: `_iteration_takes` refuses the merge 21 times
and EVERY ONE of the 21 is run 1, the columns. Run 0, the rows, is refused zero
times. "The columns cannot merge" was silently deciding "the rows cannot either".
A reshape may now merge ONE of its runs, the tile arrives as [128, 2, 2, 2, 2],
and the DMA folds the four contiguous binary axes back into 16 -- never across
the lane axis, which is the clause that made the same fold wrong on its own.

Sort -> cumsum -> mask -> renormalise, all on one row. Depends on the sort, so it cannot pass before sort_row_tl.py does.
"""

#: WHERE THIS STOPS NOW, TRACED 2026-09-23. It used to refuse in adapt; the
#: written-reshape bridge carries it past that and it now stops at the
#: scratchpad budget -- 290560 bytes/lane against 262144.
#:
#: The reason is in the shape: `expand_shape` splits a 128-long lane axis into
#: SEVEN axes of 2, and this machine banks on ONE axis, so whichever one the
#: materialised tile rides it rides 2 lanes.
#:
#: THE TILE IS RANK 11 AND EVERY AXIS IS 2 -- `memref<2x...x2xi32>`, 2048
#: elements -- so there is NO axis to find. `tl.sort`'s compare-exchange has to
#: name a stride-2^k pair on its own, and the reassociation `[[0..6],[7..10]]`
#: says the first seven are the ROW run and the last four the column run. It
#: arrives that way from stage 3: 205 rank-11 types are already in
#: `04-torchsim-convert-accesses-to-transfer.mlir`.
#:
#: THIS WAS FIXED ONCE, on 2026-09-04 (torchsim-compile 85bde3f), and the fix was to merge
#: PER RUN: the column run genuinely cannot be merged, and absorption was
#: deciding all-or-nothing per reshape, so it silently refused the row run too.
#: That merge lived in `absorb_layout_ops`, which the graph route replaced --
#: there is no such pass in the tree now, and `sort_row_tl` (302096 B/lane) and
#: `sort_radix` (672296) came back with this one.
#:
#: UPSTREAM'S RESHAPE ARMS DO NOT SUBSTITUTE. They are a pair (la.cc:2173 and
#: 2314) and both ask `ShapeUtil::AlignLayouts` -- "is there a layout that makes
#: this reshape a bitcast". Here there is none on either side: `[[0..6],[7..10]]`
#: corresponds a GROUP of axes to one, and this machine banks on one axis. The
#: answer is the shape rewrite, not the layout.
#:
#: "RIDE THE WIDEST AXIS" IS NOT THE FIX AND IS NOT A FACT ABOUT THIS MACHINE.
#: Ordering `GetUnconstrainedLayouts` by extent -- upstream leaves that hook
#: `virtual` precisely so a backend may -- moves this kernel by 512 bytes out of
#: 315136 and breaks `sort_bitonic` into a different refusal. The lane axis is
#: the one spread across BANKS; riding a wide one leaves little DEPTH per bank
#: for the vector to walk, and what the chain actually needs is that its axes
#: AGREE, with the lane count a consequence. Measured and reverted.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 16
P = 0.9


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def top_p_sampling_kernel(logit_ptr, p_ptr, M: tl.constexpr,
                              N: tl.constexpr, P: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        off = rm[:, None] * N + rn[None, :]
        z = tl.load(logit_ptr + off)

        p = tl.exp(z - tl.max(z, axis=1)[:, None])
        p = p / tl.sum(p, axis=1)[:, None]

        sp = tl.sort(p, dim=1, descending=True)

        # EXCLUSIVE cumsum -- keep the token that crosses P, not the one before.
        keep = (tl.cumsum(sp, axis=1) - sp) < P
        kept = tl.where(keep, sp, 0.0)
        tl.store(p_ptr + off, kept / tl.sum(kept, axis=1)[:, None])

    return top_p_sampling_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"logit": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    p = inputs["logit"].softmax(dim=1)
    sp, _ = p.sort(dim=1, descending=True)
    keep = (sp.cumsum(dim=1) - sp) < P
    kept = sp * keep
    return {"p": kept / kept.sum(dim=1, keepdim=True)}


SPEC = KernelSpec(
    name="top_p_sampling_kernel", kernel=kernel,
    signature={"logit_ptr": "*fp32", "p_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "P": "constexpr"},
    constexprs={"M": M, "N": N, "P": P},
    args=[Arg("logit", IN, "float32", (M, N)),
          Arg("p", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # NO VECTOR SPILL: the sort's crossings are pushed and popped one row per
    # loop iteration, so llc's scheduler cannot hold a pop across the others --
    # 447 spills when torchsim-lower-to-vcix unrolled them.
    expect_ir={"11-kernel.s": {("vscale x 8-byte Folded Spill", "vscale x 16-byte Folded Spill", "vscale x 32-byte Folded Spill", "vscale x 64-byte Folded Spill"): 0}},
)
