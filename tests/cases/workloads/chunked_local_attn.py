"""Llama 4's iRoPE chunked attention -- fixed non-overlapping BLOCKS along the
sequence, and the full-attention layer it alternates with, from one score tile.

    chunk(x) = (x - pad) // chunk_size
    keep_chunked = (chunk(j) == chunk(i)) and (j <= i)      3 layers in 4
    keep_full    =                           (j <= i)       the 4th

A BLOCK, NOT A BAND. `sliding_window.py` keeps `i - W < j <= i`: the kept region
SLIDES, every row sees a different set, and the covered span grows one token per
row forever. Chunked attention cuts the sequence at fixed multiples of
`chunk_size` and every token inside a cut sees only that cut, so rows 0 and
chunk_size-1 of the same block attend to overlapping key sets and rows on
opposite sides of a boundary attend to DISJOINT ones. Nothing crosses a boundary
at all, ever, in that layer -- there is no leak-through the way a band has, and
that is the property the design wants.

WHY LLAMA 4 WANTS IT: THE `i` IN iRoPE IS "INTERLEAVED". Three layers in four
are chunked AND rotary; the fourth has NO positional encoding whatsoever and
attends over the whole sequence. The claim is a division of labour -- local
layers get position from RoPE and are cheap because they are chunked, the global
layers get to look arbitrarily far and, because they carry no RoPE at all, have
nothing that was fitted to the training length and therefore nothing to
extrapolate wrongly. That is what lets Llama 4 Scout advertise a 10M context
while the rotary layers only ever see an 8192-token chunk. The global layers
additionally scale their logits by an inverse-temperature that grows with
position, which is arithmetic on the scores and not a masking question, so it is
not in this kernel.

  Both masks are emitted here, from the SAME score tile, because the
  alternation is the mechanism: a kernel that showed only the chunked mask
  would be describing a windowed model, and this one is a windowed model
  interleaved with a global one. Two reductions over two different maskings of
  one loaded tile also puts them in the same lane-axis component, which is the
  cheap way to check they agree about layout.

THE `- pad` AND THE DIVISION THAT HAS TO FLOOR. `left_padding` is per-sequence
in a left-padded batch, so the chunk grid is anchored to the first REAL token
rather than to offset zero and the boundaries are not aligned to the tile, to a
power of two, or to anything else. That makes `i - pad` negative for the pad
positions, and floor and truncate disagree there -- python and torch `//` floor,
`arith.divsi` truncates toward zero, so a straight port silently merges the pad
block into chunk 0. The kernel biases by one whole chunk before dividing and
subtracts it back:

    (i - pad + CHUNK) // CHUNK - 1

The dividend is now positive for every `i >= 0` given `pad < CHUNK`, so the two
conventions coincide and the value is the true floor. The bias would cancel out
of the equality on its own; it is written with the `- 1` so the tensor really is
the chunk index and can be read against the reference.

WHAT IT ASKS OF THIS MACHINE. Two integer divisions by a non-power-of-two-free
constant on `linalg.index` values, one broadcast along the lanes and one
in-lane, then an equality between them -- the same opposite-axis compare
`softmax_causal.py` set up, except the operands went through a `divsi` first, so
the affine-index machinery has to survive a division rather than only a compare.

torch equivalent:
    c = torch.div(arange(S) - PAD, CHUNK, rounding_mode="floor")
    keep = (c[:, None] == c[None, :]) & (i[:, None] >= j[None, :])

Llama4 iRoPE: fixed non-overlapping chunks (an equality between two floor-divided index tensors) and the full-causal layer it alternates with, both from one score tile.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S = 64          # sequence
CHUNK = 16      # attention_chunk_size (8192 in the shipped model)
PAD = 3         # left_padding: the chunk grid is anchored off zero on purpose
NEG = -1.0e30


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def chunked_local_attn_kernel(sc_ptr, pc_ptr, pf_ptr, S: tl.constexpr,
                                  CHUNK: tl.constexpr, PAD: tl.constexpr,
                                  NEG: tl.constexpr):
        i = tl.arange(0, S)
        j = tl.arange(0, S)
        sc = tl.load(sc_ptr + i[:, None] * S + j[None, :])

        # FLOOR division out of a truncating one: bias by a whole chunk so the
        # dividend cannot go negative, then take the chunk back off.
        qc = (i - PAD + CHUNK) // CHUNK - 1
        kc = (j - PAD + CHUNK) // CHUNK - 1

        causal = i[:, None] >= j[None, :]
        # The block: an equality between two divided index tensors broadcast
        # along opposite axes. Not a band -- nothing crosses a boundary.
        chunked = causal & (qc[:, None] == kc[None, :])

        # THE CHUNKED LAYER (3 in 4, rotary).
        zc = tl.where(chunked, sc, NEG)
        mc = tl.max(zc, axis=1)
        ec = tl.where(chunked, tl.exp(zc - mc[:, None]), 0.0)
        tl.store(pc_ptr + i[:, None] * S + j[None, :],
                 ec / tl.sum(ec, axis=1)[:, None])

        # THE LAYER IT ALTERNATES WITH (every 4th, NoPE, full causal).
        zf = tl.where(causal, sc, NEG)
        mf = tl.max(zf, axis=1)
        ef = tl.where(causal, tl.exp(zf - mf[:, None]), 0.0)
        tl.store(pf_ptr + i[:, None] * S + j[None, :],
                 ef / tl.sum(ef, axis=1)[:, None])

    return chunked_local_attn_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"sc": torch.randn(S * S, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    sc = inputs["sc"].reshape(S, S)
    idx = torch.arange(S)
    c = torch.div(idx - PAD, CHUNK, rounding_mode="floor")
    causal = idx[:, None] >= idx[None, :]
    chunked = causal & (c[:, None] == c[None, :])

    def soft(keep):
        z = torch.where(keep, sc, torch.full_like(sc, NEG))
        e = torch.where(keep, torch.exp(z - z.max(dim=1, keepdim=True).values),
                        torch.zeros_like(z))
        return (e / e.sum(dim=1, keepdim=True)).reshape(-1).contiguous()

    return {"pc": soft(chunked), "pf": soft(causal)}


SPEC = KernelSpec(
    name="chunked_local_attn_kernel",
    kernel=kernel,
    signature={"sc_ptr": "*fp32", "pc_ptr": "*fp32", "pf_ptr": "*fp32",
               "S": "constexpr", "CHUNK": "constexpr", "PAD": "constexpr",
               "NEG": "constexpr"},
    constexprs={"S": S, "CHUNK": CHUNK, "PAD": PAD, "NEG": NEG},
    args=[Arg("sc", IN, "float32", (S * S,)),
          Arg("pc", OUT, "float32", (S * S,)),
          Arg("pf", OUT, "float32", (S * S,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
