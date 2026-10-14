"""FlashAttention varlen -- one packed batch, B causal triangles, cu_seqlens data.

    q, k, v   are [NT, D]: every sequence's tokens concatenated, no padding
    cu[b]     is where sequence b starts;  cu[B] == NT
    out[i]    = softmax_j( q[i].k[j] / sqrt(D) ) v[j]
                over j in [cu[seq(i)], i]      -- causal, and INSIDE i's sequence

WHY PACKING RATHER THAN PADDING. A prefill batch of sequences 7, 6, 7 and 12
tokens long padded to a rectangle is 48 rows of which 32 are real: a third of the
attention is computed and thrown away, and the quadratic term makes the waste
worse the more the lengths differ. `flash_attn_varlen_func` takes the
concatenation instead and a `cu_seqlens` prefix sum saying where the boundaries
are. Every serving stack prefills this way, so an attention kernel that needs a
rectangular batch either pads or does not run.

THE BOUNDARY IS THE KERNEL. There are two different reasons a key can be
forbidden and they are indistinguishable in the packed layout: `j > i` is the
ordinary causal rule, and `seq(j) != seq(i)` is the packing artefact -- key j is
a perfectly good past token, of somebody else's sequence. Both are one mask:

    seq(i) = #{b : i >= cu[b+1]}          a [NT, B] compare, reduced along B
    allow  = (seq(i) == seq(j)) & (j <= i)

and `seq` is computed, not gathered: counting how many sequence ENDS lie at or
below a row turns the prefix array into a segment id with one reduction and no
indirection. Empty sequences fall out correctly (they contribute an end that
every later row already passed), which a search-for-my-segment formulation does
not give for free.

WHAT IS STRUCTURALLY NEW HERE
  * `softmax_causal.py` and `attention.py` have ONE triangle, and its mask is
    `j <= i` -- affine, foldable, decidable at compile time. This mask is a
    compare against a value LOADED FROM MEMORY, so the block-triangular structure
    of the score matrix exists but the compiler cannot see it.
  * `sliding_window.py` and `chunked_local_attn.py` also narrow the mask, but
    both narrow it by a CONSTANT, so their band is still affine in (i, j).
  * `paged_attention.py` puts data in the ADDRESS; here the addresses are dense
    and affine and the data is in the PREDICATE. Those are different demands: one
    wants a gather, this one wants a mask the vectoriser cannot prove regular.

WHAT WAS PINNED, AND WHY THIS ONE IS HONESTLY EXPRESSIBLE. NT, the total token
count, is a constant 32. That is not a concession -- a packed batch is built by
filling a fixed token budget, so NT is the one extent varlen attention DOES fix,
and it is the number the launch grid is computed from. B, the sequence count, is
likewise the array's declared length. Everything that actually varies -- where
each sequence starts, how long it is, hence the entire shape of the mask -- is in
`cu` and is read. The lengths here are 7, 6, 7, 12: none equal, none a multiple
of anything, so a boundary that got rounded to a tile would be visible.

Checked against B INDEPENDENT square attentions, so cross-sequence leakage is a large error and not a drift -- the reference never sees the packed layout at all.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NT, B, D = 32, 4, 16            # packed tokens, sequences, head dim
NEG = -1.0e30


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def varlen_packed_attn_kernel(q_ptr, k_ptr, v_ptr, cu_ptr, o_ptr,
                                  NT: tl.constexpr, B: tl.constexpr,
                                  D: tl.constexpr, NEG: tl.constexpr):
        rt = tl.arange(0, NT)
        rd = tl.arange(0, D)
        rb = tl.arange(0, B)

        # SEGMENT ID from the prefix sum: how many sequence ENDS are at or below
        # this row. [NT, B], reduced along the sequence axis. No gather.
        hi = tl.load(cu_ptr + 1 + rb)                            # [B]
        seg = tl.sum((rt[:, None] >= hi[None, :]).to(tl.int32), axis=1)   # [NT]

        off = rt[:, None] * D + rd[None, :]
        q = tl.load(q_ptr + off)
        k = tl.load(k_ptr + off)
        v = tl.load(v_ptr + off)

        s = tl.dot(q, tl.trans(k)) * (1.0 / (D ** 0.5))          # [NT, NT]

        # ONE mask, TWO reasons: causal within the sequence, and not somebody
        # else's sequence at all. Only the second half is data-dependent.
        allow = (seg[:, None] == seg[None, :]) & (rt[None, :] <= rt[:, None])
        s = tl.where(allow, s, NEG)

        s = s - tl.max(s, axis=1)[:, None]
        p = tl.exp(s)
        p = p / tl.sum(p, axis=1)[:, None]
        tl.store(o_ptr + off, tl.dot(p, v))

    return varlen_packed_attn_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Lengths 7, 6, 7, 12 -- unequal, and none a tile multiple.
    cu = torch.tensor([0, 7, 13, 20, NT], dtype=torch.int32)
    r = lambda: torch.randn(NT * D, generator=g, dtype=torch.float32)
    return {"q": r(), "k": r(), "v": r(), "cu": cu}


def reference(inputs):
    import torch
    q = inputs["q"].reshape(NT, D)
    k = inputs["k"].reshape(NT, D)
    v = inputs["v"].reshape(NT, D)
    cu = inputs["cu"].tolist()
    out = torch.zeros(NT, D, dtype=torch.float32)
    for b in range(B):
        lo, hi = cu[b], cu[b + 1]
        if hi <= lo:
            continue
        n = hi - lo
        s = (q[lo:hi] @ k[lo:hi].T) / (D ** 0.5)
        causal = torch.arange(n)[None, :] <= torch.arange(n)[:, None]
        s = s.masked_fill(~causal, float("-inf"))
        out[lo:hi] = s.softmax(dim=1) @ v[lo:hi]
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="varlen_packed_attn_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32",
               "cu_ptr": "*i32", "o_ptr": "*fp32", "NT": "constexpr",
               "B": "constexpr", "D": "constexpr", "NEG": "constexpr"},
    constexprs={"NT": NT, "B": B, "D": D, "NEG": NEG},
    args=[Arg("q", IN, "float32", (NT * D,)),
          Arg("k", IN, "float32", (NT * D,)),
          Arg("v", IN, "float32", (NT * D,)),
          Arg("cu", IN, "int32", (B + 1,)),
          Arg("o", OUT, "float32", (NT * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect="blocked",
    expect_why="matmul K=16 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
