"""DeepSeek-V3.2's sparse attention -- an indexer picks the keys, then a gather.

    score  = cheap_indexer(q, k)            [S, S], one number per (query, key)
    top-k  = the k best keys FOR EACH QUERY  -- a different set per row
    out    = softmax over just those k, against the gathered v

ATTENTION THAT DOES NOT LOOK AT EVERYTHING. Flash attention made the quadratic
cost fit in memory; it did not make it stop being quadratic. The next move is to
decide, cheaply, which keys are worth reading at all -- V3.2 runs a lightweight
indexer over the whole context, keeps the top k per query, and pays the full
attention price only on those.

  The indexer has to be much cheaper than what it saves, so it is a small
  projection rather than the real q-k product. Being approximate is fine: it
  chooses what to look at, and the real scores still decide the weights.

THIS IS `moe_router`'s SHAPE AT A DIFFERENT SCALE, and that is why it is here.
There the reduction folded to a POSITION and the position became an address --
over 8 experts. Here it is the same statement over thousands of keys, and the
selection is PER ROW rather than per token, so the gather's index varies along
both axes of the tile.

  Which is the expensive gather form: one index per tile ELEMENT, the shape
  `t5_rel_bias` has and every other gather here does not.

WHAT THE SUITE'S TOP-K CAN AND CANNOT DO. `moe_router` and `sampling` both take
the top k by repeated max, which costs k passes over the axis. That is right for
k = 2 of 8; for k = 64 of 4096 it is 64 passes, and the honest answer at that
size is a partial sort this machine has no primitive for. So this kernel keeps k
small and the axis short -- it tests the SHAPE (a per-row selection feeding a
per-element gather), not the scale.

torch equivalent:
    idx = indexer.topk(K, dim=-1).indices
    s = scores.gather(-1, idx)
    p = s.softmax(-1)
    out = (p.unsqueeze(-1) * v[idx]).sum(-2)

DeepSeek-V3.2: a per-ROW top-k choosing which keys to attend to -- moe_router's shape with the selection varying along both axes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, K = 32, 4               # keys, and how many survive per query


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def sparse_attention_kernel(ix_ptr, sc_ptr, o_ptr,
                                S: tl.constexpr, K: tl.constexpr):
        i = tl.arange(0, S)
        j = tl.arange(0, S)
        k = tl.arange(0, K)

        # THE INDEXER'S SCORES, given. What matters here is what is done with
        # them, not how they were made.
        ix = tl.load(ix_ptr + i[:, None] * S + j[None, :])

        # TOP-K BY REPEATED MAX, which is the only form available: k passes,
        # each erasing the current winner. The THRESHOLD is what comes out.
        q = ix
        thr = tl.max(q, axis=1)
        for _ in tl.static_range(K - 1):
            q = tl.where(q >= thr[:, None], -1e30, q)
            thr = tl.max(q, axis=1)

        # The survivors, as a mask rather than as indices. A row keeps exactly
        # the entries at or above its own threshold, and the softmax runs over
        # the full row with the rest driven to zero -- which is the same
        # function as gathering k of them and normalising those.
        keep = ix >= thr[:, None]

        sc = tl.load(sc_ptr + i[:, None] * S + j[None, :])
        b = tl.where(keep, sc, float("-inf"))
        m = tl.max(b, axis=1)
        ex = tl.where(keep, tl.exp(b - m[:, None]), 0.0)
        tl.store(o_ptr + i[:, None] * S + j[None, :],
                 ex / tl.sum(ex, axis=1)[:, None])

    return sparse_attention_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"ix": torch.randn(S * S, generator=g, dtype=torch.float32),
            "sc": torch.randn(S * S, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    ix = inputs["ix"].reshape(S, S)
    sc = inputs["sc"].reshape(S, S)
    thr = ix.topk(K, dim=1).values[:, -1:]
    keep = ix >= thr
    b = sc.masked_fill(~keep, float("-inf"))
    return {"o": torch.softmax(b, dim=-1).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="sparse_attention_kernel",
    kernel=kernel,
    signature={"ix_ptr": "*fp32", "sc_ptr": "*fp32", "o_ptr": "*fp32",
               "S": "constexpr", "K": "constexpr"},
    constexprs={"S": S, "K": K},
    args=[Arg("ix", IN, "float32", (S * S,)),
          Arg("sc", IN, "float32", (S * S,)),
          Arg("o", OUT, "float32", (S * S,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
