"""vLLM `copy_blocks` -- duplicate KV pages in place, from a runtime pair list.

    for p in range(P):
        k[dst[p]] = k[src[p]]        whole pages, [S, D] each
        v[dst[p]] = v[src[p]]

No arithmetic. The cache is both the source and the destination, and both ends
of every transfer are addresses read from memory.

WHY A SERVING STACK COPIES KV PAGES AT ALL. Two sequences that share a prompt
share its KV, so vLLM refcounts the pages instead of duplicating them -- a
prefix cache hit, or the n>1 / beam-search fan-out, gives several sequences the
same physical block. The moment one of them appends a token, the shared page has
to become private: allocate a new block, COPY the old one into it, then write.
That is copy-on-write, `cache_ops.copy_blocks` is the copy, and it runs between
the scheduler's decision and the model's forward pass on any step where a fork
diverged. The pair list is built on the host and handed over as a tensor,
because which blocks diverged this step is not known until the step.

WHAT IS STRUCTURALLY NEW, AGAINST THE TWO NEAREST FILES
  * `paged_attention.py` established the rank-3 page gather: `table[b]*S*D + s*D
    + d`, a [B, S, D] tile whose outer base is data. It only READS. This kernel
    has the same gather AND the mirror-image scatter, and they hit the SAME
    buffer -- so `k_ptr` appears once as a load base and once as a store base
    within one expression's distance, both with a data-dependent outer index.
    A backend that models the cache as read-only, or that needs a load's buffer
    to be distinct from a store's to prove independence, has nothing to fall
    back on here.
  * `kv_cache_write.py` scatters into the cache, but from a separate dense source
    tensor and one ROW at a time. Here the unit is a whole page, so the moved
    tile is rank 3 with only its outermost axis irregular -- which is the shape
    that decides whether this becomes P clean DMA descriptors or S*D scalar
    copies.
  * `moe_permute.py` also gathers and scatters one index vector, but through two
    different buffers and with a data payload in between. This one has no
    payload: the value loaded is the value stored.

WHY THE PAIRS ARE DISJOINT, AND WHY THAT IS NOT A SIMPLIFICATION. `src` and
`dst` are drawn from one `randperm`, so no page is both a source and a
destination and no destination repeats. That is the real precondition, not a
convenience: copy-on-write always allocates a FREE block as the destination, so a
destination is by construction a page nobody is reading. Because of it the P
copies commute and the kernel's answer does not depend on the order they are
issued in -- which is what lets the whole thing be one vectorised transfer
instead of a sequenced loop. State it, and an aliasing analysis that cannot see
it has somewhere to be told.

WHAT WAS PINNED. PAGES (the pool), S (slots per page) and D are all compile-time
in vLLM too -- the cache is allocated once with that geometry. P, the number of
pairs, is the one genuinely per-step quantity and is pinned at 4 here because it
is a tile extent; the real kernel launches a grid of P and reads the same list.
Nothing about WHICH page moves where is static: both index vectors are loaded,
and shuffled so an identity mapping cannot pass.

A pure copy, so the whole cache must come back bit-exact: the P written pages equal their sources and the untouched pages are unchanged. Any nonzero error is a lowering fault.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT, INOUT

PAGES, S, D, P = 8, 4, 16, 4    # pool pages, slots/page, head dim, copy pairs


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kv_copy_blocks_kernel(k_ptr, v_ptr, src_ptr, dst_ptr,
                              S: tl.constexpr, D: tl.constexpr,
                              P: tl.constexpr):
        rp = tl.arange(0, P)
        rs = tl.arange(0, S)
        rd = tl.arange(0, D)

        s = tl.load(src_ptr + rp)                  # [P] physical page ids, both
        d = tl.load(dst_ptr + rp)                  #     ends read from memory

        # [P, S, D] twice: same tile shape, same buffer, opposite direction.
        # The outer axis is data; the inner two are contiguous.
        sl = rs[None, :, None] * D + rd[None, None, :]
        soff = s[:, None, None] * (S * D) + sl
        doff = d[:, None, None] * (S * D) + sl

        tl.store(k_ptr + doff, tl.load(k_ptr + soff))
        tl.store(v_ptr + doff, tl.load(v_ptr + soff))

    return kv_copy_blocks_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # One permutation split in two: sources and destinations are disjoint and
    # each destination is distinct, which is the copy-on-write precondition.
    perm = torch.randperm(PAGES, generator=g).to(torch.int32)
    return {"k": torch.randn(PAGES * S * D, generator=g, dtype=torch.float32),
            "v": torch.randn(PAGES * S * D, generator=g, dtype=torch.float32),
            "src": perm[:P].contiguous(),
            "dst": perm[P:].contiguous()}


def reference(inputs):
    import torch
    src = inputs["src"].long()
    dst = inputs["dst"].long()
    out = {}
    for nm in ("k", "v"):
        c = inputs[nm].reshape(PAGES, S, D).clone()
        c[dst] = c[src]
        out[nm] = c.reshape(-1).contiguous()
    return out


SPEC = KernelSpec(
    name="kv_copy_blocks_kernel", kernel=kernel,
    signature={"k_ptr": "*fp32", "v_ptr": "*fp32", "src_ptr": "*i32",
               "dst_ptr": "*i32", "S": "constexpr", "D": "constexpr",
               "P": "constexpr"},
    constexprs={"S": S, "D": D, "P": P},
    args=[Arg("k", INOUT, "float32", (PAGES * S * D,)),
          Arg("v", INOUT, "float32", (PAGES * S * D,)),
          Arg("src", IN, "int32", (P,)),
          Arg("dst", IN, "int32", (P,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
