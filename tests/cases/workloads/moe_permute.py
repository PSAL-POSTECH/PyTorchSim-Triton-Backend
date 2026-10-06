"""MoE permute/unpermute -- the row movement `moe_align_block_size` only planned.

    y[r]      = x[src[r]]        if src[r] < T   else 0     gather  (permute)
    z[src[r]] = u[r]             where src[r] < T           scatter (unpermute)

`src` is exactly the `sorted_token_ids` buffer the align kernel emits: a slot per
expert-sorted row, padded up to the GEMM's row tile, with `T` -- the token count
-- as the sentinel in the slots no token landed on.

WHY THE COPY IS A SEPARATE KERNEL FROM THE GEMM. `fused_moe_expert_mlp.py`
assumes its rows already sit contiguously under one expert's weights; that
assumption is what this kernel pays for. The alternative is a GEMM that gathers
its A operand row by row through an index vector, which turns a dense tile load
into T independent descriptors and stalls the array. Every MoE runtime that
matters materialises the permuted activations instead: vLLM's
`moe_permute`/`moe_unpermute`, TensorRT-LLM's `expandInputRowsKernel`. The hidden
state gets copied twice per MoE layer, and it is still the cheaper trade.

WHY BOTH HALVES ARE IN ONE FILE. They are inverse addressings of the SAME index
vector, and that is the thing to test: `src` appears once as the SOURCE row of a
gather and once as the DESTINATION row of a scatter, in the same kernel, over
the same lanes. A backend that only handles the index on one side of the copy
passes half of this file.

THE NEAREST NEIGHBOURS, AND WHAT IS DIFFERENT
  * `moe_combine.py` scatters too, but its destination index has DUPLICATES
    (top-2 puts two rows on one token) so its store is a read-modify-write. Here
    every live slot holds a distinct token, the store is a plain write, and what
    is new instead is the PADDING: `src` is not a permutation, it is a partial
    injection, so both halves carry a mask.
  * `kv_cache_write.py` scatters a dense block into a cache through a slot
    vector -- one direction, no sentinel, no gather.
  * `ptr_scatter_roundtrip.py` is a total permutation with nothing masked off.

THE MASK IS LOAD-BEARING IN TWO DIFFERENT WAYS, which is the point of the
sentinel being `T` rather than -1. On the gather it selects `other=0`, so a
padded slot reads a defined zero rather than whatever is past the end of `x`. On
the scatter it suppresses the store, because a padded slot has no token to
return to. Both sides also CLAMP the row (`where(ok, s, 0)`) before it enters
the address: a masked-off lane still computes its offset, and `T * D` is one row
past the end of a T-row buffer.

WHAT WAS PINNED. R, the number of padded slots, is a constant 64 for T = 32
tokens -- the same buffer the align kernel sizes at `T + E*(BLOCK-1)`, so the
extent is static there too. How many of those slots are live, and which slot
each token goes to, is entirely in `src`. The zeroing of `z` before the scatter
is not ceremony either: with padding present the scatter cannot be assumed to
cover every output row, so the buffer has to start defined.

A pure copy, so both outputs are bit-exact and any drift is a lowering fault. One index vector used as a gather source and a scatter destination in the same launch, masked on both sides.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, R, D = 32, 64, 16            # tokens, padded expert-sorted slots, model width


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def moe_permute_kernel(x_ptr, u_ptr, src_ptr, y_ptr, z_ptr,
                           T: tl.constexpr, R: tl.constexpr,
                           D: tl.constexpr):
        rt = tl.arange(0, T)
        rr = tl.arange(0, R)
        rd = tl.arange(0, D)

        s = tl.load(src_ptr + rr)               # [R] token per slot, T == pad
        ok = s < T
        # Clamp BEFORE the address: a masked lane still forms its offset, and
        # the sentinel T is one row past the end of x.
        row = tl.where(ok, s, 0)
        off = row[:, None] * D + rd[None, :]

        # PERMUTE: src indexes the SOURCE row. Padded slots read a defined zero.
        tl.store(y_ptr + rr[:, None] * D + rd[None, :],
                 tl.load(x_ptr + off, mask=ok[:, None], other=0.0))

        # UNPERMUTE: the same src now indexes the DESTINATION row. Zero first --
        # with padding present the scatter does not cover every output row.
        tl.store(z_ptr + rt[:, None] * D + rd[None, :],
                 tl.zeros((T, D), dtype=tl.float32))
        u = tl.load(u_ptr + rr[:, None] * D + rd[None, :])
        tl.store(z_ptr + off, u, mask=ok[:, None])

    return moe_permute_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # T tokens dropped into R slots: the live slots are scattered, so a backend
    # that ignored the index cannot come out right, and the rest are sentinel.
    slots = torch.randperm(R, generator=g)[:T]
    src = torch.full((R,), T, dtype=torch.int32)
    src[slots] = torch.randperm(T, generator=g).to(torch.int32)
    return {"x": torch.randn(T * D, generator=g, dtype=torch.float32),
            "u": torch.randn(R * D, generator=g, dtype=torch.float32),
            "src": src}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(T, D)
    u = inputs["u"].reshape(R, D)
    s = inputs["src"].long()
    ok = s < T

    y = torch.zeros(R, D, dtype=torch.float32)
    y[ok] = x[s[ok]]

    z = torch.zeros(T, D, dtype=torch.float32)
    z[s[ok]] = u[ok]

    return {"y": y.reshape(-1).contiguous(), "z": z.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="moe_permute_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "u_ptr": "*fp32", "src_ptr": "*i32",
               "y_ptr": "*fp32", "z_ptr": "*fp32", "T": "constexpr",
               "R": "constexpr", "D": "constexpr"},
    constexprs={"T": T, "R": R, "D": D},
    args=[Arg("x", IN, "float32", (T * D,)),
          Arg("u", IN, "float32", (R * D,)),
          Arg("src", IN, "int32", (R,)),
          Arg("y", OUT, "float32", (R * D,)),
          Arg("z", OUT, "float32", (T * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
