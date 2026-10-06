"""Transformer-XL relative attention, as NeMo's Conformer runs it -- two bias
vectors, two score matrices, and `rel_shift`.

nemo/collections/asr/parts/submodules/multi_head_attention.py,
`RelPositionMultiHeadAttention`:

    matrix_ac = (q + pos_bias_u) @ k.T                 [S, S]   content
    matrix_bd = (q + pos_bias_v) @ pos_emb.T           [S, P]   position
    matrix_bd = rel_shift(matrix_bd)[:, :S]            aligned to (i - j)
    p         = softmax((matrix_ac + matrix_bd) / sqrt(d_k))

THE DECOMPOSITION IS THE IDEA. Expanding `(q + pos)(k + pos)` in the original
absolute-position attention gives four terms; Dai et al. (2019) replace the two
that involve an absolute key position with a relative embedding and two LEARNED
BIAS VECTORS, `u` and `v`, which stand in for "the query's own position" -- a
constant, because if position is relative then the query is always at zero. So
there are two score matrices instead of one, they are added before the softmax,
and each is a separate `tl.dot` with the SAME left operand up to a rank-1 bias.
Two matmuls sharing an input and differing in their right operand's meaning is
the shape; `attention.py` chains two matmuls, this one races two.

`pos_emb` HAS 2S-1 ROWS, NOT S. Relative distance runs from `+(S-1)` to
`-(S-1)`, so the position matmul is [S, D] @ [D, 2S-1] and produces a WIDER
matrix than the content one -- every query scored against every possible
distance, most of which it cannot use. 2S-1 is 63 here and not a power of two;
the tile is 64 wide and the last row of `pos_emb` is padding that is never read.

REL_SHIFT IS THE PART WITH NO EQUIVALENT IN THIS SUITE. `matrix_bd[i, r]` is
query `i` against distance index `r`, and what the softmax needs is query `i`
against KEY `j`, i.e. distance `i - j`. The mapping `r = (S-1-i) + j` is a
different shift per row, and Transformer-XL does not write it as a gather. It
writes it as:

    pad one zero column on the left      [S, P] -> [S, P+1]
    reinterpret                          [S, P+1] -> [P+1, S]
    drop the first row                   [P+1, S] -> [P, S]
    reinterpret                          [P, S] -> [S, P]

and the whole effect comes from the fact that the WRITE stride is P+1 and the
READ stride is P. Flattened, it is exactly `out[m] = padded[S + m]`. THE ZERO
COLUMN IS NEVER READ -- verified, the kept slice touches distance indices 0..62
only -- so the pad carries no data at all. Its entire function is to make the
two strides differ by one, which is what turns a reinterpretation into a
per-row shift.

  WHY IT GOES THROUGH MEMORY HERE. A `[S, P+1]` tile cannot exist: 33 and 65 are
  not powers of two, and no arange, reshape or zeros will produce one. So the
  pad is a store with row stride P+1 into a buffer, and the reinterpret-and-slice
  is a load at `S + i * P + j` out of the same buffer. That IS the operation --
  a view is not data movement, and here the only place a non-power-of-two stride
  can live is an address. The padded matrix is an output so the two halves can
  be inspected apart; `bds` is what the softmax actually adds.

  A compiler is free to fold the whole thing into `bd[i, S-1-i+j]` and skip the
  buffer. Nothing here does it, and the fold is only valid because the pad is
  provably unread, which depends on P >= 2S-1.

ALiBi AND T5 ARE NOT THIS. `alibi.py` computes its bias from `i - j` directly
and `t5_rel_bias.py` looks it up in a bucketed table; both produce a term that is
ADDED to the scores and neither involves q. Here the position term is a matmul
against the query -- it is content-dependent, it costs a second GEMM per layer,
and the alignment is a shift rather than an index.

Single head, no mask: NeMo's Conformer encoder attends both ways and masks only
padding. The head axis would be a static loop, as in `alibi.py`.

Two score matmuls into one softmax, and an alignment that is a stride mismatch rather than a gather: written at P+1, read at P. `bdp` is exact by construction, so a mismatch there is the store and a mismatch in `bds` alone is the reinterpretation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, D = 32, 32        # queries/keys, head dim
P = 2 * S            # relative positions: NeMo uses 2S-1, the last row is pad
SCALE = 1.0 / (D ** 0.5)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rel_shift_attn_kernel(q_ptr, k_ptr, v_ptr, pe_ptr, u_ptr, w_ptr,
                              bdp_ptr, bds_ptr, o_ptr,
                              S: tl.constexpr, D: tl.constexpr,
                              P: tl.constexpr, SCALE: tl.constexpr):
        i = tl.arange(0, S)[:, None]
        d = tl.arange(0, D)[None, :]
        dt = tl.arange(0, D)[:, None]
        j = tl.arange(0, S)[None, :]
        r = tl.arange(0, P)[None, :]

        q = tl.load(q_ptr + i * D + d)
        # Both right operands are read transposed by their own addressing, the
        # shape tl.dot wants, as in attention.py.
        kt = tl.load(k_ptr + j * D + dt)                     # [D, S]
        pt = tl.load(pe_ptr + r * D + dt)                    # [D, P]

        # The two learned biases: the query's own position, which is a constant
        # once position is relative.
        u = tl.load(u_ptr + tl.arange(0, D))
        w = tl.load(w_ptr + tl.arange(0, D))

        ac = tl.dot(q + u[None, :], kt)                      # [S, S] content
        bd = tl.dot(q + w[None, :], pt)                      # [S, P] position

        # REL_SHIFT, as a stride mismatch. Write with row stride P+1 (the pad
        # column first, then the data), read back with row stride P from offset
        # S -- which is the flat form of pad/view/slice/view.
        tl.store(bdp_ptr + tl.arange(0, S) * (P + 1),
                 tl.zeros((S,), dtype=tl.float32))
        tl.store(bdp_ptr + i * (P + 1) + 1 + r, bd)
        bds = tl.load(bdp_ptr + S + i * P + j)               # [S, S]
        tl.store(bds_ptr + i * S + j, bds)

        sc = (ac + bds) * SCALE
        m = tl.max(sc, axis=1)
        ex = tl.exp(sc - m[:, None])
        p = ex / tl.sum(ex, axis=1)[:, None]

        v = tl.load(v_ptr + i * D + d)                       # [S, D]
        tl.store(o_ptr + i * D + d, tl.dot(p, v))

    return rel_shift_attn_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)
    return {"q": r(S * D) * 0.5, "k": r(S * D) * 0.5, "v": r(S * D) * 0.5,
            "pe": r(P * D) * 0.5, "u": r(D) * 0.1, "w": r(D) * 0.1}


def reference(inputs):
    import torch
    q = inputs["q"].reshape(S, D)
    k = inputs["k"].reshape(S, D)
    v = inputs["v"].reshape(S, D)
    pe = inputs["pe"].reshape(P, D)
    ac = (q + inputs["u"]) @ k.T                             # [S, S]
    bd = (q + inputs["w"]) @ pe.T                            # [S, P]

    # rel_shift, exactly as NeMo writes it.
    bdp = torch.nn.functional.pad(bd, (1, 0))                # [S, P+1]
    bds = bdp.reshape(P + 1, S)[1:].reshape(S, P)[:, :S]

    p = torch.softmax((ac + bds) * SCALE, dim=-1)
    return {"bdp": bdp.reshape(-1).contiguous(),
            "bds": bds.reshape(-1).contiguous(),
            "o": (p @ v).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="rel_shift_attn_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32",
               "pe_ptr": "*fp32", "u_ptr": "*fp32", "w_ptr": "*fp32",
               "bdp_ptr": "*fp32", "bds_ptr": "*fp32", "o_ptr": "*fp32",
               "S": "constexpr", "D": "constexpr", "P": "constexpr",
               "SCALE": "constexpr"},
    constexprs={"S": S, "D": D, "P": P, "SCALE": SCALE},
    args=[Arg("q", IN, "float32", (S * D,)),
          Arg("k", IN, "float32", (S * D,)),
          Arg("v", IN, "float32", (S * D,)),
          Arg("pe", IN, "float32", (P * D,)),
          Arg("u", IN, "float32", (D,)),
          Arg("w", IN, "float32", (D,)),
          Arg("bdp", OUT, "float32", (S * (P + 1),)),
          Arg("bds", OUT, "float32", (S * S,)),
          Arg("o", OUT, "float32", (S * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
