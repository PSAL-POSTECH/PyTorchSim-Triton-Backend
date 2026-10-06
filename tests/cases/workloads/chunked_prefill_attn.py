"""vLLM V1 unified attention -- decode rows and prefill rows in the SAME launch.

Four requests share one launch. Two are decoding (one query row each, a long
cached prefix behind them); two are mid-prefill (a chunk of query rows, with
however much of themselves is already in the cache):

    r0  q_len 1    kv_len 16    ctx 15     decode
    r1  q_len 1    kv_len 16    ctx 15     decode
    r2  q_len 14   kv_len 16    ctx 2      prefill chunk, partly cached
    r3  q_len 16   kv_len 16    ctx 0      prefill chunk, nothing cached

For query row i, local index li inside its request r:

    ctx[r]  = kv_len[r] - q_len[r]                 keys already in the cache
    allow j iff  0 <= j - kv_start[r] <= ctx[r] + li

WHY THE TWO PHASES STOPPED BEING TWO KERNELS. A server that runs prefill and
decode in separate batches has to choose which to starve: batching decodes waits
for a prefill to finish and adds latency to every token; batching prefills leaves
the array idle on one-row queries. Continuous batching mixes them, and once
mixed there is no reason to launch twice -- so vLLM V1 dropped the split entirely
and its `unified_attention` takes one query buffer where a decode is just a
prefill chunk of length one. Chunked prefill is the same idea applied to a single
long request: send it 512 tokens at a time so a decode never waits behind it.

WHAT THAT DOES TO THE MASK, AND IT IS THE WHOLE KERNEL. The causal boundary is
no longer `j <= i`. A query row is not at position `li` in its sequence, it is at
position `ctx[r] + li`, because `ctx[r]` tokens of that sequence were attended in
an earlier launch and live in the cache. So the diagonal is SHIFTED, by a
different amount per request, and the amount is data. Two consequences:

  * `ctx == 0` gives an ordinary triangle (r3, a fresh prefill).
  * `q_len == 1` gives a single full row (r0, r1, decode) -- the mask degenerates
    rather than being special-cased. That is the unification: one predicate, and
    the phase is a property of the numbers in it.

  Get the shift wrong and a prefill row attends to its own future or forgets its
  own prefix; the decode rows keep working either way, which is what makes this
  worth a kernel rather than a flag.

WHAT IS STRUCTURALLY NEW AGAINST `varlen_packed_attn.py`, the nearest file. That
kernel packs a ragged batch too, but every row there obeys the SAME rule (causal
inside my sequence) and its q and kv share one index space, so `j <= i` compares
two positions in one buffer. Here q and kv are separately packed with separate
prefix sums, the causal comparison happens in each request's LOCAL kv
coordinates, and the offset between the two coordinate systems is `ctx[r]` --
per request, from memory. That offset is why the query and key axes cannot be
the same axis, and why this file exists next to that one.

  It is also the first mask here built from a GATHER: `seg` comes out of the
  usual [NQ, B] prefix compare, and then `q_cu[seg]`, `k_cu[seg]` are read with
  it -- one index per lane, addressing four small arrays, to place each row in
  its own coordinate frame before a single compare decides everything.

WHAT WAS PINNED. NQ (32 packed query rows) and NK (64 packed kv rows). Both are
fixed budgets in vLLM as well -- the scheduler fills a token budget and the KV is
a preallocated pool -- and they are what the grid is computed from. The split
into requests, the per-request q_len, the per-request kv_len and therefore every
causal boundary in the launch are in `q_cu` and `k_cu` and are read. The mix
itself is data: nothing in the kernel knows which rows are decodes.

MEASURED 2026-09-03, AND FIXED 2026-09-04. When select_lane_axis gained the turn
it was missing, this kernel's eight round trips went and the answer went with
them -- 510 of 512 elements, err 1.69952. The turn was right; what did not
travel with it was the MASK.

WHERE IT WENT, because the first three guesses were all wrong. Not the weight
push: SPIKE_DEBUG showed the array holding the softmax rows on 32 lanes and the
first matmul reproducing k @ q.t() to 2.69e-06 across all 64 keys. Not the S
buffer, not its six reinterpret_casts, and NOT "K > N breaks the push" -- the
default sweep carries (128,64,128), (16,16,32), (16,16,64) and (16,16,128) and
every one passes, so that was disproved rather than left standing.

What the array actually held was the softmax of an UNMASKED row: every query
attended to all 64 keys instead of its own request's window.

    row  0   got keys 0..63 (64)   want 0..15 (16)     decode, kv_len 16
    row 16   got keys 0..63 (64)   want 48..48 ( 1)    first row of a fresh chunk

`_turn_group` transposes a generic by substituting d0 <-> d1 in its
`indexing_maps` and reversing each map's results. The body names axes too:

    %20 = linalg.index 1        kloc = rk - k0,  rk is the KEY axis

Before the turn the space is [query, key] and dim 1 is the key. After it the
space is [key, query] and dim 1 is the QUERY, and the op still said 1 -- so the
compare subtracted the per-query k0 from the query index and `allow` came out
true everywhere. That is rule 18 on an axis-naming attribute rather than on a
stride: a dim substitution that stops at `indexing_maps` leaves the body reading
the axis it used to mean. The turn now renumbers `linalg.index d` to `1 - d`,
which is its own inverse like every other half of a turn (rule 17).

This kernel is the one that reaches memory with it: the gathered k0/lim make the
mask data rather than a triangle, so a mask that silently degenerates to all-true
still compiles, still round-trips nothing, and is only visible in the values.

The reference runs the four requests separately with their own ctx shift, so a diagonal placed by q position instead of kv position is correct on r3, correct on the decodes, and wrong on r2 alone.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NQ, NK, R, D = 32, 64, 4, 16    # packed query rows, packed kv rows, requests, dim
NEG = -1.0e30


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def chunked_prefill_attn_kernel(q_ptr, k_ptr, v_ptr, qcu_ptr, kcu_ptr,
                                    o_ptr, NQ: tl.constexpr, NK: tl.constexpr,
                                    R: tl.constexpr, D: tl.constexpr,
                                    NEG: tl.constexpr):
        rq = tl.arange(0, NQ)
        rk = tl.arange(0, NK)
        rd = tl.arange(0, D)
        rr = tl.arange(0, R)

        # Which request each query row belongs to: [NQ, R] compare against the
        # query prefix sum, reduced along the request axis.
        qhi = tl.load(qcu_ptr + 1 + rr)                                  # [R]
        rid = tl.sum((rq[:, None] >= qhi[None, :]).to(tl.int32), axis=1)  # [NQ]

        # Both coordinate frames, gathered with that id -- one index per lane.
        q0 = tl.load(qcu_ptr + rid)
        q1 = tl.load(qcu_ptr + rid + 1)
        k0 = tl.load(kcu_ptr + rid)
        k1 = tl.load(kcu_ptr + rid + 1)

        li = rq - q0                          # position of this row in its chunk
        # THE SHIFTED DIAGONAL: ctx = kv_len - q_len is what an earlier launch
        # already put in the cache. ctx == 0 is a plain triangle, q_len == 1 is
        # a decode; neither is special-cased.
        lim = (k1 - k0) - (q1 - q0) + li      # last allowed LOCAL key index

        q = tl.load(q_ptr + rq[:, None] * D + rd[None, :])
        k = tl.load(k_ptr + rk[:, None] * D + rd[None, :])
        v = tl.load(v_ptr + rk[:, None] * D + rd[None, :])

        s = tl.dot(q, tl.trans(k)) * (1.0 / (D ** 0.5))                  # [NQ,NK]

        kloc = rk[None, :] - k0[:, None]      # key index in THIS request's frame
        allow = (kloc >= 0) & (kloc <= lim[:, None])
        s = tl.where(allow, s, NEG)

        s = s - tl.max(s, axis=1)[:, None]
        p = tl.exp(s)
        p = p / tl.sum(p, axis=1)[:, None]
        tl.store(o_ptr + rq[:, None] * D + rd[None, :], tl.dot(p, v))

    return chunked_prefill_attn_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # r0, r1 decode (1 query row, 15 cached); r2 prefill chunk with 2 cached;
    # r3 fresh prefill chunk with nothing cached.
    qcu = torch.tensor([0, 1, 2, 16, NQ], dtype=torch.int32)
    kcu = torch.tensor([0, 16, 32, 48, NK], dtype=torch.int32)
    return {"q": torch.randn(NQ * D, generator=g, dtype=torch.float32),
            "k": torch.randn(NK * D, generator=g, dtype=torch.float32),
            "v": torch.randn(NK * D, generator=g, dtype=torch.float32),
            "qcu": qcu, "kcu": kcu}


def reference(inputs):
    import torch
    q = inputs["q"].reshape(NQ, D)
    k = inputs["k"].reshape(NK, D)
    v = inputs["v"].reshape(NK, D)
    qcu = inputs["qcu"].tolist()
    kcu = inputs["kcu"].tolist()
    out = torch.zeros(NQ, D, dtype=torch.float32)
    for r in range(R):
        qa, qb = qcu[r], qcu[r + 1]
        ka, kb = kcu[r], kcu[r + 1]
        nq, nk = qb - qa, kb - ka
        ctx = nk - nq
        s = (q[qa:qb] @ k[ka:kb].T) / (D ** 0.5)
        allow = (torch.arange(nk)[None, :]
                 <= (ctx + torch.arange(nq))[:, None])
        s = s.masked_fill(~allow, float("-inf"))
        out[qa:qb] = s.softmax(dim=1) @ v[ka:kb]
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="chunked_prefill_attn_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32",
               "qcu_ptr": "*i32", "kcu_ptr": "*i32", "o_ptr": "*fp32",
               "NQ": "constexpr", "NK": "constexpr", "R": "constexpr",
               "D": "constexpr", "NEG": "constexpr"},
    constexprs={"NQ": NQ, "NK": NK, "R": R, "D": D, "NEG": NEG},
    args=[Arg("q", IN, "float32", (NQ * D,)),
          Arg("k", IN, "float32", (NK * D,)),
          Arg("v", IN, "float32", (NK * D,)),
          Arg("qcu", IN, "int32", (R + 1,)),
          Arg("kcu", IN, "int32", (R + 1,)),
          Arg("o", OUT, "float32", (NQ * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
