"""The MoE combine -- weighted scatter-add of top-2 expert outputs back to tokens.

    for every slot r:   out[dst[r]] += g[r] * y[r]

y is the expert-major buffer: after dispatch, the tokens have been permuted so
each expert's rows are contiguous and one GEMM per expert can run over a slice of
it. `dst[r]` says which token row slot r came from and `g[r]` is that token's
routing weight for that expert. With top-2 every token appears TWICE, so the
destination index has duplicates -- and that is the whole kernel.

WHERE IT SITS. `moe_router.py` and `deepseek_moe.py` choose the experts;
`fused_moe_expert_mlp.py` runs one. This is the last step, `finalize_moe_routing`
in TensorRT-LLM and `moe_sum` after `unpermute` in vLLM, and it is the only part
of a MoE layer where TWO DIFFERENT SOURCE ROWS ADDRESS THE SAME DESTINATION.
Nothing else in this suite does that: `ptr_scatter_roundtrip.py` scatters through
a permutation, so every address is written exactly once and the store needs no
accumulator.

WHY THERE IS NO ATOMIC HERE, WHICH IS ALSO WHY THERE IS NONE IN PRODUCTION. An
atomic float add is both slow and nondeterministic -- the same batch gives
different logits on a rerun, which makes an eval unreproducible. So the runtimes
buy their way out on the host: the permutation is built so that the slots are
grouped by RANK -- all the tokens' first-choice experts, then all their
second-choice -- and within a rank group every token appears exactly once. The
destination index of a single tile is then a PERMUTATION, and the scatter-add
degenerates into K passes of gather, add, scatter with no collision inside any
one of them. The duplicates survive only ACROSS passes, where the sequential
loop already orders them.

  That is the shape here: T slots per pass, `dst` a permutation of the token
  rows within the pass, K = 2 passes accumulating into the same output.

WHAT IT ASKS OF THE BACKEND
  1. A READ-MODIFY-WRITE THROUGH A SCATTERED ADDRESS. `o_ptr + dst*D + d` is both
     the load's address and the store's, in the same expression, in a loop. A
     `tts.gather` and a `tts.scatter` on the OUTPUT buffer, aliasing each other by
     construction -- and correct only because the two are ordered.
  2. THE INDEX IS ONE PER ROW, and rows are what banks hold, so it is the cheap
     one-index-per-lane form rather than one per element (embedding.py).
  3. The source side is structured and the destination side is not, in the same
     store: `y` is read at contiguous rows and written at scattered ones. The two
     halves of one copy want different addressing.

The zeroing store before the loop is not ceremony: pass 0 accumulates like every
other pass, so the buffer has to start at zero rather than at whatever the
allocator left. A real kernel gets it for free from the first pass writing
instead of adding; written out here so the accumulate is visible K times, not
K-1.

The only kernel here whose store address repeats. A gather and a scatter on the OUTPUT buffer in one expression; correct only if the two passes stay ordered, so a reordering shows up as a wrong number.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, D, K = 128, 16, 2            # tokens (== lane count), model width, top-k
R = K * T                       # expert-major slots, one per (token, expert)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def moe_combine_kernel(y_ptr, dst_ptr, g_ptr, o_ptr,
                           T: tl.constexpr, D: tl.constexpr, K: tl.constexpr):
        rt = tl.arange(0, T)
        rd = tl.arange(0, D)

        # Start at zero: every pass below accumulates, including the first.
        tl.store(o_ptr + rt[:, None] * D + rd[None, :],
                 tl.zeros((T, D), dtype=tl.float32))

        for k in tl.static_range(K):
            row = k * T + rt                       # source rows: contiguous
            d = tl.load(dst_ptr + row)             # destination token: scattered
            g = tl.load(g_ptr + row)               # its routing weight
            v = tl.load(y_ptr + row[:, None] * D + rd[None, :])

            # One address expression, read AND written. The duplicates live
            # across passes, so the loop order is what makes this an add.
            off = d[:, None] * D + rd[None, :]
            tl.store(o_ptr + off, tl.load(o_ptr + off) + g[:, None] * v)

    return moe_combine_kernel


def make_inputs(torch, seed=0):
    gen = torch.Generator().manual_seed(seed)
    # Slots are grouped by RANK: pass k holds every token exactly once, in the
    # order that expert-major permutation happens to put them.
    perms = [torch.randperm(T, generator=gen) for _ in range(K)]
    dst = torch.cat(perms).to(torch.int32)

    # Top-2 weights that sum to 1 per token, indexed back through the perms.
    w0 = torch.rand(T, generator=gen, dtype=torch.float32) * 0.6 + 0.2
    per_tok = torch.stack([w0, 1.0 - w0])
    g = torch.cat([per_tok[k][perms[k]] for k in range(K)])

    return {"y": torch.randn(R * D, generator=gen, dtype=torch.float32),
            "dst": dst, "g": g.contiguous()}


def reference(inputs):
    import torch
    y = inputs["y"].reshape(R, D)
    dst = inputs["dst"].long()
    out = torch.zeros(T, D, dtype=torch.float32)
    out.index_add_(0, dst, inputs["g"][:, None] * y)
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="moe_combine_kernel", kernel=kernel,
    signature={"y_ptr": "*fp32", "dst_ptr": "*i32", "g_ptr": "*fp32",
               "o_ptr": "*fp32", "T": "constexpr", "D": "constexpr",
               "K": "constexpr"},
    constexprs={"T": T, "D": D, "K": K},
    args=[Arg("y", IN, "float32", (R * D,)),
          Arg("dst", IN, "int32", (R,)),
          Arg("g", IN, "float32", (R,)),
          Arg("o", OUT, "float32", (T * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
