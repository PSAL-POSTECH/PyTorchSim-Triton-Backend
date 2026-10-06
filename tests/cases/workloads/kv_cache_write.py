"""A paged KV-cache write -- the gather/scatter shape LLM serving is built on.

    cache[block_table[b] * BLOCK + slot, :] = k[b, :]

vLLM and every serving stack after it keeps the KV cache in fixed-size BLOCKS
and a per-sequence BLOCK TABLE mapping logical position to physical block. So a
decode step's write is not a contiguous store: the row index comes out of a
table in memory, and two sequences' consecutive tokens land nowhere near each
other.

WHY IT IS THE SHAPE THAT MATTERS HERE
  The destination row is an INDEX READ FROM DRAM, which is a scatter whose index
  is a value and not an affine function of the loop. That is the one addressing
  mode a DMA descriptor cannot fold into a stride, and it is why the descriptor
  has an indirect mode at all.

  It is also where the gather rule shows its teeth: the scattered axis is dim 0
  and the lane axis is dim 0, so ONE index per row serves the whole row and the
  lane already holds the only index its slots need. Line those two up and the
  index buffer is BM entries; miss and it is BM*BD, which is what an
  unstructured scatter costs.

  The read arm gathers back through the same table, so a wrong index shows up as
  a mismatch rather than as a silent overwrite of memory nobody checks.

torch equivalent: cache.index_copy_ along dim 0, then index_select back.

Paged KV cache: the destination row is an index read from DRAM, which is the one addressing mode no stride can fold. Scattered axis and lane axis line up, so it is one index per row.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT, INOUT

NROW, BD = 32, 16                  # cache rows, head dim
BM = 8                             # tokens written this step


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kv_cache_write_kernel(k_ptr, slot_ptr, cache_ptr, o_ptr,
                              BD: tl.constexpr, BM: tl.constexpr):
        m = tl.arange(0, BM)
        d = tl.arange(0, BD)
        slot = tl.load(slot_ptr + m)                    # physical row per token
        k = tl.load(k_ptr + m[:, None] * BD + d[None, :])
        tl.store(cache_ptr + slot[:, None] * BD + d[None, :], k)
        back = tl.load(cache_ptr + slot[:, None] * BD + d[None, :])
        tl.store(o_ptr + m[:, None] * BD + d[None, :], back * 2.0)

    return kv_cache_write_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # A permutation slice: no two tokens share a physical row, so the write
    # order cannot change the answer.
    slot = torch.randperm(NROW, generator=g)[:BM].to(torch.int32)
    return {"k": torch.randn(BM * BD, generator=g, dtype=torch.float32),
            "slot": slot,
            "cache": torch.zeros(NROW * BD, dtype=torch.float32)}


def reference(inputs):
    import torch
    k = inputs["k"].reshape(BM, BD)
    slot = inputs["slot"].long()
    cache = inputs["cache"].reshape(NROW, BD).clone()
    cache[slot] = k
    return {"cache": cache.reshape(-1), "o": (cache[slot] * 2.0).reshape(-1)}


SPEC = KernelSpec(
    name="kv_cache_write_kernel", kernel=kernel,
    signature={"k_ptr": "*fp32", "slot_ptr": "*i32", "cache_ptr": "*fp32",
               "o_ptr": "*fp32", "BD": "constexpr", "BM": "constexpr"},
    constexprs={"BD": BD, "BM": BM},
    args=[Arg("k", IN, "float32", (BM * BD,)), Arg("slot", IN, "int32", (BM,)),
          Arg("cache", INOUT, "float32", (NROW * BD,)),
          Arg("o", OUT, "float32", (BM * BD,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
