"""LLaMA's grouped-query attention -- one KV head serving a group of Q heads.

    head h reads kv head h // G,   G = n_heads / n_kv_heads

WHY GQA EXISTS, and it is a memory argument rather than a quality one. During
decoding the KV cache is read in full for every single token generated, so its
size sets the bandwidth floor of the whole model -- for a 7B model at 4k context
that is gigabytes moved per token. Multi-Query Attention (Shazeer 2019) took the
extreme position of ONE kv head for all of them, which cuts the cache by n_heads
but costs quality and destabilises training. Ainslie et al. (2023) split the
difference: G query heads share a kv head, G typically 4 or 8. LLaMA-2 70B,
Mistral, and essentially every model since ship GQA, because the interpolation
turns out to be nearly free -- quality tracks MHA while the cache tracks MQA.

WHAT IT IS MECHANICALLY. `repeat_kv` -- a broadcast along the head axis, where
each kv head is repeated G times to line back up with the query heads. Nothing
is recomputed and nothing is gathered from far away; it is the cheapest possible
operation on a general-purpose machine, which is exactly why it is worth asking
about on this one.

WHY THE HEAD AXIS IS THE INTERESTING ONE HERE. Every broadcast this suite has
met so far runs along a spatial or feature axis. This one runs along HEADS, and
heads are the axis a decoder naturally puts on the lanes -- they are the
independent work. So:

    heads on the lanes    head h and head h+1 are in different banks, and a kv
                          head has to reach G of them. There is no lane-to-lane
                          move, so that is a replicating DRAM round trip.
    heads in-lane         the repeat is a splat inside one bank, free.

That is the same trade `convnext_block` resolved by writing its tile [HW, C]
instead of [C, HW]. Here the tile is written heads-major ON PURPOSE -- H is the
lane count -- because the expensive orientation is the one worth a witness.

AND THE INDEX IS COMPUTED, NOT LOADED. `h // G` is integer division of the
iteration variable, so there is no index buffer in DRAM at all -- unlike
`embedding` (token ids) or `kv_cache_write` (page table). Whether the backend
sees an affine access or synthesises a gather is the second question this file
asks, and the two have very different costs.

THE SHAPE IS THE SCORE-AND-GATE OF ONE DECODE STEP: a per-head dot of q against
its kv head, squashed, then applied to that same kv head's value. Attention
proper is `attention.py`'s job; what is new here is only the repeat.

torch equivalent: kk = k.repeat_interleave(G, 0); s = (q*kk).sum(-1);
                  out = sigmoid(s)[:, None] * v.repeat_interleave(G, 0)

LLaMA: repeat_kv -- a broadcast along the HEAD axis, with the heads deliberately on the lanes, from a computed rather than loaded index.

WHICH CARRIER, BY COST. Each kv tile reaches its [128, 64] layout as one XLU move (202 cycles by the
target's vcix keys) rather than as its load in 32 DMA pieces (31 * 27 more): lower-copies prices both
and expect_ir holds the two moves. Breaks if a copy is folded into its load whatever that costs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

H, G, D = 128, 4, 64             # query heads (== lane count), group size, head dim
HKV = H // G                     # kv heads


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gqa_repeat_kv_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                             H: tl.constexpr, G: tl.constexpr,
                             D: tl.constexpr, HKV: tl.constexpr):
        h = tl.arange(0, H)
        d = tl.arange(0, D)
        # THE REPEAT. `h // G` maps G consecutive query heads onto one kv head;
        # the index is computed from the iteration variable, not read from DRAM.
        hkv = h // G

        q = tl.load(q_ptr + h[:, None] * D + d[None, :])
        k = tl.load(k_ptr + hkv[:, None] * D + d[None, :])
        v = tl.load(v_ptr + hkv[:, None] * D + d[None, :])

        # A per-head score: folds D, which is the in-lane axis, so it is the
        # cheap direction of reduction and the repeat is the only cost here.
        s = tl.sum(q * k, axis=1)
        tl.store(o_ptr + h[:, None] * D + d[None, :],
                 tl.sigmoid(s)[:, None] * v)

    return gqa_repeat_kv_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"q": torch.randn(H * D, generator=g, dtype=torch.float32) * 0.2,
            "k": torch.randn(HKV * D, generator=g, dtype=torch.float32) * 0.2,
            "v": torch.randn(HKV * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    q = inputs["q"].reshape(H, D)
    k = inputs["k"].reshape(HKV, D).repeat_interleave(G, dim=0)
    v = inputs["v"].reshape(HKV, D).repeat_interleave(G, dim=0)
    s = (q * k).sum(dim=1)
    return {"o": (torch.sigmoid(s)[:, None] * v).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="gqa_repeat_kv_kernel",
    kernel=kernel,
    signature={"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32",
               "o_ptr": "*fp32", "H": "constexpr", "G": "constexpr",
               "D": "constexpr", "HKV": "constexpr"},
    constexprs={"H": H, "G": G, "D": D, "HKV": HKV},
    args=[Arg("q", IN, "float32", (H * D,)),
          Arg("k", IN, "float32", (HKV * D,)),
          Arg("v", IN, "float32", (HKV * D,)),
          Arg("o", OUT, "float32", (H * D,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"08-transformed.mlir": {"pytorchsim.crosslane_move": 2},
               "09-custom.mlir": {"piece_box": 0}},
)
