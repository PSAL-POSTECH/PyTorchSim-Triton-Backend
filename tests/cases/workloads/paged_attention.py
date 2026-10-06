"""Paged attention -- the KV a block table says, not the KV that is contiguous.

torch equivalent, for one query and a page table of B blocks of S slots:
    k = k_cache[table].reshape(B*S, D)      # gather whole PAGES
    out = softmax(q @ k.T / sqrt(D)) @ v

WHY vLLM MADE THE CACHE NON-CONTIGUOUS
  A sequence's KV grows a token at a time and every sequence grows a different
  amount, so a contiguous cache per sequence either wastes the tail or copies on
  every extend. Paging fixes the allocation and moves the indirection into a
  table: block i of THIS sequence lives at physical page table[i]. Every serving
  stack in use does this, so an attention kernel that needs contiguous KV cannot
  serve.

WHAT MAKES IT HARDER THAN dl/kv_cache_write.py
  That kernel writes one slot, so its indirection is a scalar. This one READS a
  whole page per table entry -- the base of a [S, D] tile is loaded from memory,
  and there are B of them. The address arithmetic is
  `table[b] * S * D + s * D + d`, which is a gather whose innermost two axes are
  contiguous and whose outer axis is not. That is the shape the DMA has to
  express: B separate transfers, or one transfer with a per-block base.

  It is also why the table is int32 and loaded first: the tile's ADDRESS depends
  on data, so nothing about this access can be folded at compile time.

THE PAGES ARE DELIBERATELY OUT OF ORDER in make_inputs. A table that happens to
be the identity would let a backend that ignored the indirection pass.

WHAT STOPS IT TODAY -- A UNIT ROW NOBODY ARRANGED TO REPLICATE. It used to reach
the suite as `KILLED by signal 6 (SIGABRT) -- no diagnostic was printed`. That
was two faults stacked: vectorize's `_mask_tail` built its keep-mask from a
list of PYTHON BOOLS, which DenseElementsAttr has no path for (`std::bad_cast`),
and the same pass drove `_nest` with a bare `__enter__` so the raise abandoned
its insertion points -- masking the real exception and aborting the interpreter
at shutdown. Both are fixed, and what the kernel actually needs is now readable:

    NotVectorisable: operand [1, 1, 16] broadcasts along dim 0 into a tile
    [4, 4, 16], and it was staged one element per lane -- every lane needs the
    whole row, which takes a replicating transfer this operand did not get

fit_layout_to_lanes is where that is decided -- `_stage_one_map_row` ranks the four
stagings a row can get, and `_clone_loaded_row_wide` is the free one: a second
transfer of the same DRAM with a zero stride on the broadcast axis. It needs a
DRAM address to re-read, and here the row is the per-head scale, expanded to
`1x1x16` and multiplied into the `4x4x16` score tile -- computed, not loaded, so
there is no address. `_replicate_computed_row` is the answer that does not need
one: a DRAM round trip with a zero stride.

THAT REFUSAL NO LONGER FIRES, AND THIS KERNEL VERIFIES (1.19209e-07). It was
carried as `expect="open"` past the point where it stopped being open -- the
committed baseline at efcf0f2 already reports "THE GAP HAS CLOSED", so the
declaration was stale when the bar was recorded. Which change closed it is not
pinned here, and the sweep says it was not this kernel alone: the three files
this section used to name as needing the same fix -- block_diag_attn_mask,
top_p_sampling, sort_row_tl -- no longer show the replication refusal either.
They stop at later and DIFFERENT walls (a vlane split on axis 0 peeled into the
outer loop nest for two of them, relayout wanting an allocation for the third),
which is what the replication half being done looks like from the outside.

WHAT IT PINS NOW is the whole of the above as an ordinary kernel: a rank-3
gather whose outer base is data, a deliberately shuffled page table, and a
computed row broadcast across lanes. Any of those regressing shows up here as a
number rather than as a refusal.

A rank-3 gather whose OUTER base comes from memory. The hard part is not the attention, it is that the tile's address is data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

PAGES, S, D, B = 8, 4, 16, 4          # cache pages, slots/page, head dim, blocks


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def paged_attention_kernel(q_ptr, k_ptr, v_ptr, tab_ptr, o_ptr,
                               S: tl.constexpr, D: tl.constexpr,
                               B: tl.constexpr):
        rd = tl.arange(0, D)
        rs = tl.arange(0, S)
        rb = tl.arange(0, B)

        q = tl.load(q_ptr + rd)                       # [D]
        page = tl.load(tab_ptr + rb)                  # [B]  physical page ids

        # [B, S, D]: the page base is DATA, the slot and dim are affine.
        off = (page[:, None, None] * (S * D)
               + rs[None, :, None] * D + rd[None, None, :])
        k = tl.load(k_ptr + off)
        v = tl.load(v_ptr + off)

        # Scores over every slot of every block, then one softmax across both.
        sc = tl.sum(k * q[None, None, :], axis=2) * (1.0 / (D ** 0.5))   # [B,S]
        sc = sc - tl.max(tl.max(sc, axis=1), axis=0)
        w = tl.exp(sc)
        w = w / tl.sum(tl.sum(w, axis=1), axis=0)
        tl.store(o_ptr + rd, tl.sum(tl.sum(w[:, :, None] * v, axis=1), axis=0))

    return paged_attention_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Shuffled, so a backend that ignores the table cannot come out right.
    table = torch.randperm(PAGES, generator=g)[:B].to(torch.int32)
    return {"q": torch.randn(D, generator=g, dtype=torch.float32),
            "k": torch.randn(PAGES * S, D, generator=g, dtype=torch.float32),
            "v": torch.randn(PAGES * S, D, generator=g, dtype=torch.float32),
            "tab": table}


def reference(inputs):
    idx = inputs["tab"].long()
    k = inputs["k"].view(PAGES, S, D)[idx].reshape(B * S, D)
    v = inputs["v"].view(PAGES, S, D)[idx].reshape(B * S, D)
    sc = (k @ inputs["q"]) / (D ** 0.5)
    return {"o": (sc.softmax(dim=0) @ v)}


SPEC = KernelSpec(
    name="paged_attention_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32",
               "tab_ptr": "*i32", "o_ptr": "*fp32",
               "S": "constexpr", "D": "constexpr", "B": "constexpr"},
    constexprs={"S": S, "D": D, "B": B},
    args=[Arg("q", IN, "float32", (D,)),
          Arg("k", IN, "float32", (PAGES * S, D)),
          Arg("v", IN, "float32", (PAGES * S, D)),
          Arg("tab", IN, "int32", (B,)),
          Arg("o", OUT, "float32", (D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
