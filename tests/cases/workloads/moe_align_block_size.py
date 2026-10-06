"""vLLM's `moe_align_block_size` -- the MoE layer's index kernel, and only indexes.

    cnt[e]    = #{t : expert[t] == e}                     histogram
    pad[e]    = ceil(cnt[e] / BLOCK) * BLOCK              round each expert UP
    base[e]   = pad[0] + ... + pad[e-1]                   exclusive prefix
    sorted[base[e] + rank_within_e(t)] = t                where each token lands
    sorted[everything else]            = T                the padding sentinel
    block_expert[b] = the e whose padded span covers block b, else -1
    n_post_pad      = sum(pad)

WHY A KERNEL EXISTS WHOSE OUTPUT IS ENTIRELY ADDRESSES
  A fused MoE GEMM wants to run one weight tile over a CONTIGUOUS run of rows
  that all use that weight. The router produces the opposite: a token->expert
  map in token order, with the experts interleaved arbitrarily. Sorting by
  expert fixes the interleaving but not the tiling -- an expert with 37 tokens
  ends mid-tile, so the next expert's rows would share a tile with a different
  weight matrix. Padding each expert's run up to a multiple of BLOCK buys the
  guarantee the GEMM needs: every tile of BLOCK rows belongs to exactly one
  expert, which is why `block_expert` is a per-BLOCK vector and not a per-row
  one. vLLM runs this once per MoE layer per step, ahead of the GEMM.

  The padded rows are why the sentinel is `T` and not -1: the downstream gather
  masks on `id < T`, so the out-of-range value has to be a number the compare
  can use, not a flag.

WHAT IS STRUCTURALLY NEW HERE. Every other MoE file in this directory moves
floats. `moe_router.py` and `deepseek_moe.py` produce a routing DECISION and
then immediately spend it on a gather; `fused_moe_expert_mlp.py` and
`moe_combine.py` consume a permutation that someone else built. This kernel is
that someone else, and it is INTEGER END TO END -- there is not one float in the
signature. Nothing here loads a hidden state, so a backend cannot hide index
arithmetic inside the shadow of a DMA; the address computation IS the kernel.

THE THREE PIECES ARE THREE DIFFERENT SHAPES OF THE SAME COUNT
  * the histogram is [E, T]  -- experts on the lanes, tokens in-lane, reduced
    along the token axis.
  * the exclusive prefix over experts is [E, E]: `sum_{e' < e} pad[e']` as a
    masked reduction rather than a scan, because a scan over experts would have
    to cross lanes and E is 4.
  * the per-token rank is [T, T]: `#{t' < t : expert[t'] == expert[t]}`. That
    is a strictly-lower-triangular equality count -- a segmented rank, which is
    the one primitive a histogram cannot give you and the reason this is not
    just a bincount.

  `dest` is then built a FOURTH way, [T, E], summing every earlier expert's
  padded size directly per token, so the base vector is never gathered through
  `expert[t]`. Both forms are here on purpose: `base` proves the prefix, `dest`
  shows that the same quantity can be reached without an indirection.

WHAT WAS PINNED. T (tokens in the batch) and NS (the padded output buffer) are
constants. In vLLM both are runtime: NS is allocated as `T + E*(BLOCK-1)`, the
worst case, which is exactly the bound used here -- 32 + 4*7 = 60 <= 64. So the
static extent is the ALLOCATION, which is static in vLLM too; what stays data is
every number that matters -- the histogram, the padding, where each token lands,
and `n_post_pad`, which is the value the host reads back to size the GEMM's grid.
The question the kernel asks is unchanged: none of the offsets can be folded.

THE ASSIGNMENT IS DELIBERATELY IMBALANCED, and expert 3 gets nothing. An empty
expert must produce pad == 0, own zero blocks, and not shift anybody -- the case
a uniform-count test never reaches.

Integer end to end, so every output is EXACT and any mismatch is a lowering fault rather than arithmetic. Three orientations of one count: [E,T] histogram, [E,E] prefix, [T,T] segmented rank. Carried `expect="open"` past the point it stopped being open -- the committed baseline at efcf0f2 already said THE GAP HAS CLOSED, so the declaration was stale when the bar was recorded, and which change closed it is not pinned. All three outputs now match exactly.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, E, BLOCK = 32, 4, 8          # tokens (top-1), experts, GEMM row-tile
NS = 64                         # padded slots; bound is T + E*(BLOCK-1) == 60
NB = NS // BLOCK                # blocks the padded buffer can hold


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def moe_align_block_size_kernel(eid_ptr, sid_ptr, bid_ptr, npad_ptr,
                                    T: tl.constexpr, E: tl.constexpr,
                                    BLOCK: tl.constexpr, NS: tl.constexpr,
                                    NB: tl.constexpr):
        rt = tl.arange(0, T)
        re = tl.arange(0, E)
        rs = tl.arange(0, NS)
        rb = tl.arange(0, NB)

        e = tl.load(eid_ptr + rt)                       # [T] expert per token

        # HISTOGRAM, [E, T]: experts on the lanes, tokens in-lane, reduced along
        # the token axis. An empty expert falls out as cnt == 0.
        cnt = tl.sum((e[None, :] == re[:, None]).to(tl.int32), axis=1)   # [E]
        pad = ((cnt + (BLOCK - 1)) // BLOCK) * BLOCK                     # [E]

        # EXCLUSIVE PREFIX over experts, [E, E]. A masked reduction and not a
        # scan: the scan direction would be the lane axis.
        base = tl.sum(tl.where(re[None, :] < re[:, None], pad[None, :], 0),
                      axis=1)                                            # [E]

        # SEGMENTED RANK, [T, T]: how many earlier tokens chose the same expert.
        # Strictly lower triangular AND equal -- the part a histogram cannot do.
        same = (e[:, None] == e[None, :]) & (rt[None, :] < rt[:, None])
        rank = tl.sum(same.to(tl.int32), axis=1)                         # [T]

        # DESTINATION, [T, E]: every earlier expert's PADDED size, summed per
        # token. Same number as base[e[t]], reached without an indirection.
        dest = rank + tl.sum(
            tl.where(re[None, :] < e[:, None], pad[None, :], 0), axis=1)

        # Padding first, then the real tokens land on top of it.
        tl.store(sid_ptr + rs, tl.full((NS,), T, tl.int32))
        tl.store(sid_ptr + dest, rt)

        # BLOCK -> EXPERT, [NB, E]. At most one expert covers a block, so the
        # +1/-1 encoding gives -1 for the blocks past n_post_pad.
        lo = base // BLOCK
        hi = (base + pad) // BLOCK
        inb = (rb[:, None] >= lo[None, :]) & (rb[:, None] < hi[None, :])
        tl.store(bid_ptr + rb,
                 tl.sum(tl.where(inb, re[None, :] + 1, 0), axis=1) - 1)

        # The number the host reads back to size the GEMM grid.
        tl.store(npad_ptr + tl.arange(0, 1), tl.sum(pad[None, :], axis=1))

    return moe_align_block_size_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    e = torch.randint(0, E, (T,), generator=g, dtype=torch.int32)
    # Expert 3 gets nothing: pad == 0, owns no block, shifts nobody.
    e = torch.where(e == E - 1, torch.zeros_like(e), e)
    return {"eid": e}


def reference(inputs):
    import torch
    e = inputs["eid"].long()
    cnt = torch.bincount(e, minlength=E)
    pad = ((cnt + BLOCK - 1) // BLOCK) * BLOCK
    base = torch.cat([torch.zeros(1, dtype=torch.long), pad.cumsum(0)[:-1]])

    seen = torch.zeros(E, dtype=torch.long)
    dest = torch.zeros(T, dtype=torch.long)
    for t in range(T):
        dest[t] = base[e[t]] + seen[e[t]]
        seen[e[t]] += 1

    sid = torch.full((NS,), T, dtype=torch.int32)
    sid[dest] = torch.arange(T, dtype=torch.int32)

    bid = torch.full((NB,), -1, dtype=torch.int32)
    for x in range(E):
        for b in range(int(base[x]) // BLOCK, int(base[x] + pad[x]) // BLOCK):
            bid[b] = x

    return {"sid": sid, "bid": bid,
            "npad": torch.tensor([int(pad.sum())], dtype=torch.int32)}


SPEC = KernelSpec(
    name="moe_align_block_size_kernel", kernel=kernel,
    signature={"eid_ptr": "*i32", "sid_ptr": "*i32", "bid_ptr": "*i32",
               "npad_ptr": "*i32", "T": "constexpr", "E": "constexpr",
               "BLOCK": "constexpr", "NS": "constexpr", "NB": "constexpr"},
    constexprs={"T": T, "E": E, "BLOCK": BLOCK, "NS": NS, "NB": NB},
    args=[Arg("eid", IN, "int32", (T,)),
          Arg("sid", OUT, "int32", (NS,)),
          Arg("bid", OUT, "int32", (NB,)),
          Arg("npad", OUT, "int32", (1,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
