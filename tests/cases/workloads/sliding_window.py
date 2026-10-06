"""Mistral's sliding window and its rolling KV buffer -- a BAND, not a half-plane.

    attend to j  iff  i - W < j <= i          a band of width W
    cache slot   k mod W                      the buffer never grows

TWO THINGS, AND THEY ARE THE SAME IDEA TWICE. If a token can only see W tokens
back, then a key older than W is unreachable forever -- so the cache does not
need to keep it, and Mistral's is a ring of exactly W slots. The mask and the
buffer are the same bound written in two places.

THE MASK IS THE PART THIS BACKEND HAS NOT SEEN. Causal masking is a HALF-PLANE:
`j <= i`, one comparison, and `softmax_causal` already covers it. A window is a
BAND -- two comparisons, and the kept region moves with the row rather than
growing with it. It is the difference between a triangle and a diagonal stripe,
and the stripe is what makes attention linear in sequence length instead of
quadratic.

  Layer L sees W tokens; layer 2L sees 2W through what layer L already mixed in.
  Mistral-7B's W = 4096 over 32 layers reaches 131k tokens of receptive field
  with every layer costing a 4096-wide band. The information travels the way it
  does in a CNN's stack of 3x3s, not through any single long-range edge.

THE ROLLING BUFFER IS A MODULO WRITE. Position k goes to slot `k mod W`, so the
cache is written in place and the oldest entry is overwritten by the newest.
Reading it back in order means unrolling the ring:

    slot of j = j mod W          for the W most recent j

`kv_cache_write` already has a gather whose destination row comes from DRAM;
this one's index is COMPUTED (`k & (W-1)` since W is a power of two), which is
the cheaper half of the same shape -- no index load, just arithmetic the lane
does itself.

WHAT IT ASKS OF THIS MACHINE. The band is two comparisons on `linalg.index`
values, which stay inside a lane; the modulo write is a scatter whose row index
is per-row, which is the cheap gather form. Neither crosses lanes, so a sliding
window costs this backend nothing that causal masking did not already.

EXCEPT FOR ONE THING, AND IT IS THE BAND. `i[:, None] - j[None, :]` reads ONE
`memref<64xi32>` twice in one generic, `(d0,d1)->(d0)` and `(d0,d1)->(d1)`, so
one buffer axis is two iteration dims, and a rule-17 contradiction: whichever
dim rides the lanes, the other read wants another lane's bank. The answer is one
replicating cross-lane move, and nothing but that move can give it.

    before   FAIL  adapt: NotVectorisable: operands of one elementwise op are
                   banked on different iteration dims [0, 1]
    after    PASS  p 5.96046e-08, c 0, 1 DRAM round trip

WHAT WAS IN THE WAY was not the cost but a deferral: `_stage_one_map_row`
skipped any row a `linalg.broadcast` also reads, on the grounds that
`_widen_broadcast_chain` would retype it in place -- a rewrite that cannot run
below bufferize (no results to replace) and fires 0 times over the suite. This
row does feed a broadcast, whose own splat is free and stays exactly where it
is. Put that deferral back and this kernel is red again.

torch equivalent:
    d = i[:, None] - j[None, :]
    p = softmax(scores.masked_fill((d < 0) | (d >= W), -inf), dim=-1)
    cache[k % W] = v[k]

Mistral: a BAND mask (two comparisons, moving with the row) rather than causal's half-plane, and a rolling KV buffer written modulo W.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, W, D = 64, 16, 32           # sequence, window, head dim


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def sliding_window_kernel(sc_ptr, v_ptr, p_ptr, c_ptr,
                              S: tl.constexpr, W: tl.constexpr,
                              D: tl.constexpr):
        i = tl.arange(0, S)
        j = tl.arange(0, S)
        d = tl.arange(0, D)

        # THE BAND. Causal is `dist >= 0`; a window adds the far edge, and it is
        # the second comparison that makes the kept region move with the row.
        dist = i[:, None] - j[None, :]
        keep = (dist >= 0) & (dist < W)

        sc = tl.load(sc_ptr + i[:, None] * S + j[None, :])
        z = tl.where(keep, sc, float("-inf"))
        m = tl.max(z, axis=1)
        ex = tl.where(keep, tl.exp(z - m[:, None]), 0.0)
        p = ex / tl.sum(ex, axis=1)[:, None]
        tl.store(p_ptr + i[:, None] * S + j[None, :], p)

        # THE RING. Slot `k mod W`, and W is a power of two so the modulo is a
        # mask -- arithmetic each lane does for itself, no index to load.
        k = tl.arange(0, S)
        v = tl.load(v_ptr + k[:, None] * D + d[None, :])
        slot = k & (W - 1)
        tl.store(c_ptr + slot[:, None] * D + d[None, :], v)

    return sliding_window_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"sc": torch.randn(S * S, generator=g, dtype=torch.float32),
            "v": torch.randn(S * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    sc = inputs["sc"].reshape(S, S)
    v = inputs["v"].reshape(S, D)
    i = torch.arange(S)[:, None]
    j = torch.arange(S)[None, :]
    d = i - j
    keep = (d >= 0) & (d < W)
    z = sc.masked_fill(~keep, float("-inf"))
    p = torch.softmax(z, dim=-1)
    # The ring after the whole sequence: slot s holds the LAST k with k % W == s.
    c = torch.zeros(W, D)
    for k in range(S):
        c[k % W] = v[k]
    return {"p": p.reshape(-1).contiguous(), "c": c.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="sliding_window_kernel",
    kernel=kernel,
    signature={"sc_ptr": "*fp32", "v_ptr": "*fp32", "p_ptr": "*fp32",
               "c_ptr": "*fp32", "S": "constexpr", "W": "constexpr",
               "D": "constexpr"},
    constexprs={"S": S, "W": W, "D": D},
    args=[Arg("sc", IN, "float32", (S * S,)),
          Arg("v", IN, "float32", (S * D,)),
          Arg("p", OUT, "float32", (S * S,)),
          Arg("c", OUT, "float32", (W * D,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
