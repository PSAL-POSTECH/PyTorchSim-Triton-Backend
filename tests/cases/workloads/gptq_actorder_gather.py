"""GPTQ with `desc_act=True` -- the activations are PERMUTED before the matmul.

    xp = x[:, perm]                       # runtime gather along the contracted axis
    w  = ((packed >> (4*(k%2))) & 0xF) - 8
    w  = w * scale[k // G]                # groups are contiguous ONLY after perm
    o  = xp @ w

WHY A PERMUTATION IS PART OF A QUANTISED LINEAR AT ALL
  GPTQ quantises a weight column by column, and each step feeds the residual
  error forward into the columns not yet done. Doing the columns in order of
  DECREASING activation magnitude (`desc_act`, "act order") spends the accurate
  early steps on the rows the model actually uses and dumps the accumulated
  error on the rows that barely fire. It is worth one to two points of
  perplexity on a 4-bit 7B and it is on by default in most GPTQ checkpoints.

  The cost is that the group boundaries are now in permuted order. `g_idx` is
  stored per input row and says which group that row belongs to. The naive
  kernel reads `g_idx[k]` and gathers a scale per k; exllama's kernel instead
  sorts once so that g_idx becomes non-decreasing, PERMUTES THE ACTIVATIONS to
  match, and then the group lookup collapses back to `k // G`. This file is the
  second form, because it is the one that ships, and because the interesting
  operation is then isolated: one gather, on the activations, in front of an
  otherwise ordinary grouped matmul.

WHAT IT ASKS THAT quant_int4_matmul.py DOES NOT
  quant_int4_matmul.py is exactly this kernel with `perm` removed. Everything
  after the gather -- index-dependent nibble shift, coarse-axis scale broadcast,
  int -> float -> dot -- is identical on purpose, so the delta is one thing:

    THE INNER INDEX OF A 2-D LOAD COMES FROM DRAM. `x_ptr + m*K + perm[k]` has a
    data-dependent offset on the axis that is the vector axis of the tile. Every
    other index in that kernel is affine in `tl.arange`, which is what lets a
    load become a strided `memref.subview`; this one cannot be, so it has to
    stay a real gather -- a vector of addresses, not a base and a stride. That
    is the whole question, and no amount of pointer analysis removes it, because
    `perm` is a runtime tensor.

  Note WHICH operand is gathered. `kv_cache_write` gathers rows, whose index is
  per-row and therefore uniform across the vector; here the index varies WITHIN
  the vector, which is the expensive form.

STORAGE. `perm` is int32 -- it is a genuine permutation of [0, K), produced once
at load time and reused for every token, so it is an input and not something the
kernel derives. `wq` is int32-containered nibbles for the reason
quant_int4_matmul.py gives: this ABI has no i8, and the values are what matter.
The permuted activation `xp` is stored as well as `o` so a gather fault can be
told from a matmul fault -- `xp` is a pure data movement and must be bit-exact.

desc_act: a DRAM-sourced index on the INNER axis of a 2-D load, in front of the grouped matmul quant_int4_matmul.py already covers. `xp` is pure data movement and must be bit-exact.

LAYOUT PIN. `expect_ir` holds `torchsim.copy` in 04-passes/04-torchsim-materialize-layouts.mlir at 0; what that pins is the gather index keeping the data tile's layout (the linalg lowering: the index rows and the 1-D offset are built in the offset's encoding); without it 2.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N, G = 16, 32, 16, 8            # tokens, hidden, out channels, group size


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gptq_actorder_gather_kernel(x_ptr, perm_ptr, wq_ptr, s_ptr,
                                    xp_ptr, o_ptr, M: tl.constexpr,
                                    K: tl.constexpr, N: tl.constexpr,
                                    G: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)

        # THE ACT-ORDER GATHER. `p` is a runtime permutation of [0, K), so the
        # inner offset of this 2-D load is a loaded tensor and not an affine
        # function of the iteration index. The row offset stays affine.
        p = tl.load(perm_ptr + rk)                                   # [K] int32
        xp = tl.load(x_ptr + rm[:, None] * K + p[None, :])           # [M, K]
        tl.store(xp_ptr + rm[:, None] * K + rk[None, :], xp)

        # From here down this is quant_int4_matmul.py verbatim -- and it only
        # works because the permutation already made the groups contiguous.
        byte = tl.load(wq_ptr + (rk[:, None] // 2) * N + rn[None, :])
        w = ((byte >> ((rk[:, None] % 2) * 4)) & 0xF) - 8             # [K, N]
        s = tl.load(s_ptr + (rk[:, None] // G) * N + rn[None, :])
        wf = w.to(tl.float32) * s

        tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(xp, wf))

    return gptq_actorder_gather_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, K, generator=g, dtype=torch.float32),
            # A genuine permutation: the argsort of the per-row activation norm
            # is what GPTQ stores, and any permutation exercises the same load.
            "perm": torch.randperm(K, generator=g).to(torch.int32),
            "wq": torch.randint(0, 256, (K // 2, N), generator=g,
                                dtype=torch.int32),
            "s": torch.rand(K // G, N, generator=g, dtype=torch.float32) * 0.1}


def reference(inputs):
    import torch
    xp = inputs["x"][:, inputs["perm"].to(torch.int64)]
    byte = inputs["wq"]
    lo = (byte & 0xF) - 8
    hi = ((byte >> 4) & 0xF) - 8
    w = torch.stack([lo, hi], dim=1).reshape(K, N).to(torch.float32)
    w = w * inputs["s"].repeat_interleave(G, dim=0)
    return {"xp": xp.contiguous(), "o": xp @ w}


SPEC = KernelSpec(
    name="gptq_actorder_gather_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "perm_ptr": "*i32", "wq_ptr": "*i32",
               "s_ptr": "*fp32", "xp_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "K": "constexpr", "N": "constexpr",
               "G": "constexpr"},
    constexprs={"M": M, "K": K, "N": N, "G": G},
    args=[Arg("x", IN, "float32", (M, K)),
          Arg("perm", IN, "int32", (K,)),
          Arg("wq", IN, "int32", (K // 2, N)),
          Arg("s", IN, "float32", (K // G, N)),
          Arg("xp", OUT, "float32", (M, K)),
          Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"04-passes/04-torchsim-materialize-layouts.mlir": {"torchsim.copy": 0}},
)
