"""Residual vector quantisation, encode side: search a codebook for the nearest
entry, subtract it, and search again with the next codebook.

    for q in levels:
        d[m, k] = ||r[m]||^2 - 2 r[m].c_q[k] + ||c_q[k]||^2
        i[q, m] = argmin_k d[m, k]
        r[m]   -= c_q[i[q, m]]

This is `EuclideanCodebook.quantize` inside `ResidualVectorQuantizer`, and it is
how EnCodec, DAC and Mimi turn a continuous encoder output into the integer
tokens a language model can predict. EnCodec's 24 kHz model stacks up to 32
codebooks of 1024 entries at 75 frames/s; Mimi uses 8 of 2048 at 12.5 frames/s,
which is 1.1 kbit/s and the reason a speech LLM can afford to model audio as
tokens at all. The levels are RESIDUAL: each one quantises what the ones before
it got wrong, so `q` codebooks of `K` entries cover `K^q` points with `qK`
vectors of storage.

WHAT IS NEW HERE, AND embedding.py IS THE WRONG COMPARISON TO STOP AT.
`embedding.py` is the DECODE side of exactly this: an index arrives from
somewhere and a row is gathered. This kernel is where the index comes from, and
the index is the result of a reduction over a distance matrix the kernel just
computed. The suite already has one reduction-to-a-position feeding a gather --
`moe_router.py`'s top-2, whose argmax picks an expert row. What is not in the
suite is that the pair is a LOOP BODY WHOSE OUTPUT IS ITS OWN NEXT INPUT:

    matmul -> reduce to an index -> gather by that index -> subtract -> matmul

`r` crosses the back edge, so level 1's addresses depend on level 0's argmin.
There is no level of this that can be hoisted, reordered or computed
speculatively; a compiler that wants to overlap the gather with the next matmul
has to break a genuine dependence. `moe_router.py` runs the pattern once and
`selective_scan.py` carries state across a loop without any addressing in it;
this carries both.

THE DISTANCE IS A MATMUL PLUS TWO NORMS, which is how every implementation
writes it -- expanding `||r - c||^2` turns `M*K` D-length subtractions into one
`[M, D] @ [D, K]` GEMM. The two norm terms are where the layout gets
interesting:

  * `||r||^2` folds along the input tile's in-lane axis and broadcasts back
    along it -- the ordinary direction, and the same shape as a softmax's max.
  * `||c||^2` folds along the CODEBOOK tile's in-lane axis, giving one value per
    entry, and is then broadcast along the DISTANCE tile's in-lane axis. The
    entries are the lane axis of one tile and the in-lane axis of the other, so
    this rank-1 vector changes which axis it lives on between producing and
    consuming. It is the only transposed rank-1 in this file and the place a
    lane/vector mix-up would show as plausible-but-wrong indices.

  `||r||^2` IS CONSTANT ALONG k AND CANNOT CHANGE THE ARGMIN. Dropping it saves
  a reduction per level and every fast implementation does, some by negating and
  taking a max instead. It is kept because a distance that is not a distance
  makes the residual chain unverifiable -- and because a compiler noticing that
  a term is invariant along the reduced axis of an argmin is a real
  optimisation, and it cannot notice what is not there.

THE INDEX IS THE OUTPUT THAT MATTERS. `idx` is `[Q, M]` int32 -- the tokens --
and it is exact or wrong, never close: one codebook entry away is a different
symbol. `res` is what the next level would quantise, and it is float, so the two
outputs fail in different ways. A tie in the argmin would make the two
implementations legitimately disagree; the codebook is Gaussian in 16
dimensions, where exact ties do not happen.

The gather is `c[idx[m], :]` -- one index per lane addressing a whole row, which
is the cheap DMA form `embedding.py` documents, not the per-element form
`t5_rel_bias.py` needs.

GREEN SINCE 2026-09-04, AND WHAT IT WAS OPEN ON. vectorize refused with
"operands of one elementwise op are banked on different iteration dims [0, 1]"
-- four of this kernel's all-parallel ops read one buffer the codebook GEMM pins
on one axis beside buffers that follow the op onto the other. select_lane_axis
had the test for that (`_is_one_iteration_dim`) and reported it only when a widen
could fix it, so the ladder never reached step 4 and the refusal fell out a pass
later. The test is now asked plainly, and the cut it plans moves ONLY the operand
another op pins: a copy of an unpinned one rejoins this op's own class and is a
relayout with the same layout at both ends. Costs 12 round trips.

WHY THE OTHER FOUR IN THIS FAMILY DID NOT COME WITH IT. qk_norm,
clip_contrastive_logits, block_scale_2d_matmul and lightning_attn have their
pinned operand in the `outs` slot, and a relayout mints a new value while the op
already produces that one -- `_plan_cuts` says so in as many words. They need a
cut ON THE RESULT, which is a rewrite this pass does not have yet.

The encode side of a codec: a distance GEMM, an argmin that produces a TOKEN, a row gather addressed by it, and a residual that carries the whole chain to the next level. `idx` is integral -- exact or wrong -- and `res` is float, so the two say different things.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 32       # frames to quantise
D = 16       # codebook dimension (EnCodec projects down to 8 or 16)
K = 32       # entries per codebook (EnCodec: 1024, Mimi: 2048)
Q = 3        # residual levels (EnCodec: up to 32, Mimi: 8)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rvq_nearest_code_kernel(x_ptr, cb_ptr, idx_ptr, res_ptr,
                                M: tl.constexpr, D: tl.constexpr,
                                K: tl.constexpr, Q: tl.constexpr):
        m = tl.arange(0, M)[:, None]
        d = tl.arange(0, D)[None, :]
        kk = tl.arange(0, K)[:, None]
        dt = tl.arange(0, D)[:, None]
        kt = tl.arange(0, K)[None, :]

        r = tl.load(x_ptr + m * D + d)                        # [M, D]

        for q in tl.static_range(Q):
            base = q * K * D
            cb = tl.load(cb_ptr + base + kk * D + d)          # [K, D]
            cbt = tl.load(cb_ptr + base + kt * D + dt)        # [D, K]

            # ||c||^2 is folded on the codebook's in-lane axis and consumed on
            # the distance tile's in-lane axis: the entries change axis here.
            cn = tl.sum(cb * cb, axis=1)                      # [K]
            rn = tl.sum(r * r, axis=1)                        # [M]
            dist = rn[:, None] - 2.0 * tl.dot(r, cbt) + cn[None, :]

            # THE INDEX IS COMPUTED, not supplied. This is the only thing
            # standing between a continuous vector and a token.
            i = tl.argmin(dist, axis=1)                       # [M] i32
            tl.store(idx_ptr + q * M + tl.arange(0, M), i)

            # ...and immediately becomes an address: one index per lane, whole
            # row. The subtraction is what makes the next level residual.
            sel = tl.load(cb_ptr + base + i[:, None] * D + d)
            r = r - sel

        tl.store(res_ptr + m * D + d, r)

    return rvq_nearest_code_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Later codebooks quantise smaller residuals, so their entries are smaller.
    # This is not cosmetic: with a flat scale the residual norm measured
    # 22.4 -> 21.2 -> 21.3 -> 20.7, i.e. level 1 OVERSHOOTS and undoes level 0,
    # and the chain stops being a refinement. Halving per level gives
    # 22.4 -> 21.2 -> 18.3 -> 16.4. A trained codec gets the same effect from
    # training each level on the previous one's error.
    cb = torch.cat([torch.randn(K * D, generator=g, dtype=torch.float32)
                    * (0.5 ** q) for q in range(Q)])
    return {"x": torch.randn(M * D, generator=g, dtype=torch.float32),
            "cb": cb}


def reference(inputs):
    import torch
    r = inputs["x"].reshape(M, D).clone()
    cb = inputs["cb"].reshape(Q, K, D)
    idx = torch.empty(Q, M, dtype=torch.int32)
    for q in range(Q):
        c = cb[q]
        dist = ((r * r).sum(1, keepdim=True) - 2.0 * (r @ c.T)
                + (c * c).sum(1)[None, :])
        i = dist.argmin(dim=1)
        idx[q] = i.to(torch.int32)
        r = r - c[i]
    return {"idx": idx.reshape(-1).contiguous(),
            "res": r.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="rvq_nearest_code_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "cb_ptr": "*fp32", "idx_ptr": "*i32",
               "res_ptr": "*fp32", "M": "constexpr", "D": "constexpr",
               "K": "constexpr", "Q": "constexpr"},
    constexprs={"M": M, "D": D, "K": K, "Q": Q},
    args=[Arg("x", IN, "float32", (M * D,)),
          Arg("cb", IN, "float32", (Q * K * D,)),
          Arg("idx", OUT, "int32", (Q * M,)),
          Arg("res", OUT, "float32", (M * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect="blocked",
    expect_why="matmul K=16 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
