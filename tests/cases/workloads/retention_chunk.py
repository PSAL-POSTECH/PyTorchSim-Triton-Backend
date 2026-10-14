"""RetNet's chunked retention -- the same block shape with the gate taken away.

Per head, with a FIXED decay gamma and blocks of B:

    Dm[i, j] = gamma^(i-j)  for i >= j, else 0
    nrm[i]   = 1 / sqrt(sum_j Dm[i, j])          # the row sum of the decay mask
    o        = nrm * ( ((Q K^T) * Dm) V  +  gamma^(i+1) * (Q S) )
    S       <- gamma^B S + (K * gamma^(B-1-j))^T V

WHY THIS IS THE ONE TO HAVE ALONGSIDE THE OTHERS. Mamba-2, the gated delta rule
and lightning attention are three points on the same family: state `S`, decay per
step, chunk the sequence, quadratic inside and recurrent between. What separates
them is only what multiplies the state. Mamba-2 has a decay computed from the
input (`segment_sum_decay.py` builds it). The delta rule has a non-diagonal
transition (`gated_delta_rule.py`). Retention has NOTHING -- gamma is a fixed
constant per head, `1 - 2^(-5-h)`, not learned and not input-dependent. It is the
degenerate endpoint, and having it pinned means a failure in the gated kernels
can be attributed to the gate rather than to the chunking.

WHAT IT ADDS THAT lightning_attn.py DOES NOT HAVE, and the two are close enough
that this is the whole justification for a second file:

  * THE DECAY IS DATA, NOT A LITERAL. Lightning attention has one slope and it is
    a `tl.constexpr`, so its [B, B] decay matrix is loop-invariant and hoistable
    out of everything. Retention has a DIFFERENT gamma per head, read from
    memory, so the matrix has to be rebuilt inside the head loop from a value the
    compiler cannot see. Same expression, opposite schedule: the exponentials
    move from prologue to loop body, and whether they do is checkable in the IR.

  * THERE IS A REDUCTION OVER THE DECAY MATRIX, FEEDING BACK INTO IT. `nrm` is
    the row sum of `Dm` -- so the [B, B] tile is built, folded on its inner axis,
    and the result broadcast back over the output rows before the matmul's
    product leaves the block. torchscale computes exactly this
    (`mask / mask.sum(-1, keepdim=True).sqrt()`) and it is the only numerical
    safeguard retention has: with no softmax there is no normaliser, so early
    rows of a block -- which sum few terms -- would otherwise come out
    systematically smaller than late ones. Nothing else in the linear-attention
    set reduces its own decay matrix.

  * TWO NESTED STATIC LOOPS, heads outside and chunks inside, with a [D, DV]
    state that must be RE-ZEROED at each head boundary and carried across the
    chunk boundary. Getting the two mixed up is a class of bug the single-head
    kernels cannot express.

THE NORMALISER FACTORS OUT, WHICH IS WHY THE REFERENCE CAN BE THE RECURRENCE.
`nrm[i]` scales both terms of row i equally, so the whole thing is
`nrm[i] * (q_i^T KV_i)` with `KV` the plain un-normalised recurrence. The
reference runs that recurrence step by step and applies the scale afterwards; the
kernel runs the block form. They agree only if the split, the mask, the two
boundary weights and the row sum are all right, so agreement is worth something.

gamma^(B-1-j) weighting the OLDEST element of the block most is correct and looks
inverted: by the time the block ends, element j has already decayed B-1-j times.
The per-block states are stored so that term can be checked on its own.

Chunked linear attention with no gate at all: the decay is a per-head constant READ FROM MEMORY, and its own row sum normalises the block. lightning_attn.py has the same block split with a constexpr decay.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

H, BLK, NC = 2, 16, 2           # heads, block length, block count
T = BLK * NC                    # sequence length
D, DV = 16, 16                  # key dim, value dim -- the state is D x DV
TD = T * D
TDV = T * DV
BLKM1 = BLK - 1


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def retention_chunk_kernel(q_ptr, k_ptr, v_ptr, gam_ptr, o_ptr, s_ptr,
                               H: tl.constexpr, NC: tl.constexpr,
                               BLK: tl.constexpr, D: tl.constexpr,
                               DV: tl.constexpr, TD: tl.constexpr,
                               TDV: tl.constexpr, BLKM1: tl.constexpr):
        rb = tl.arange(0, BLK)
        rd = tl.arange(0, D)
        rv = tl.arange(0, DV)
        i = rb[:, None].to(tl.float32)
        j = rb[None, :].to(tl.float32)

        for h in tl.static_range(H):
            # THE DECAY IS A LOAD. Rebuilt per head; nothing here is hoistable.
            lg = tl.log(tl.load(gam_ptr + h))
            Dm = tl.where(i >= j, tl.exp(lg * (i - j)), 0.0)        # [BLK, BLK]
            # A reduction over the decay matrix, broadcast back over its rows.
            nrm = 1.0 / tl.sqrt(tl.sum(Dm, axis=1))                 # [BLK]
            wi = tl.exp(lg * (rb.to(tl.float32) + 1.0)) * nrm       # incoming
            wo = tl.exp(lg * (BLKM1 - rb.to(tl.float32)))           # outgoing
            gB = tl.exp(lg * BLK)

            # Re-zeroed per head, carried across chunks.
            S = tl.zeros((D, DV), dtype=tl.float32)
            for ck in tl.static_range(NC):
                bq = h * TD + ck * BLK * D
                bv = h * TDV + ck * BLK * DV
                qb = tl.load(q_ptr + bq + rb[:, None] * D + rd[None, :])
                kT = tl.load(k_ptr + bq + rb[None, :] * D + rd[:, None])
                vb = tl.load(v_ptr + bv + rb[:, None] * DV + rv[None, :])

                o = tl.dot(tl.dot(qb, kT) * Dm, vb) * nrm[:, None]
                o += tl.dot(qb * wi[:, None], S)
                tl.store(o_ptr + bv + rb[:, None] * DV + rv[None, :], o)

                S = gB * S + tl.dot(kT * wo[None, :], vb)
                tl.store(s_ptr + (h * NC + ck) * (D * DV)
                         + rd[:, None] * DV + rv[None, :], S)

    return retention_chunk_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = 1.0 / (D ** 0.5)
    # gamma_h = 1 - 2^(-5-h): the fixed schedule, not a parameter.
    gam = torch.tensor([1.0 - 2.0 ** (-5.0 - h) for h in range(H)],
                       dtype=torch.float32)
    return {"q": torch.randn(H * T * D, generator=g, dtype=torch.float32) * s,
            "k": torch.randn(H * T * D, generator=g, dtype=torch.float32) * s,
            "v": torch.randn(H * T * DV, generator=g, dtype=torch.float32),
            "gam": gam}


def reference(inputs):
    import torch
    q = inputs["q"].reshape(H, T, D)
    k = inputs["k"].reshape(H, T, D)
    v = inputs["v"].reshape(H, T, DV)
    o = torch.empty(H, T, DV, dtype=torch.float32)
    st = torch.empty(H, NC, D, DV, dtype=torch.float32)
    idx = torch.arange(BLK, dtype=torch.float32)
    for h in range(H):
        gam = float(inputs["gam"][h])
        # The normaliser: row sums of the block's decay mask.
        Dm = torch.where(idx[:, None] >= idx[None, :],
                         gam ** (idx[:, None] - idx[None, :]),
                         torch.zeros((), dtype=torch.float32))
        nrm = Dm.sum(dim=1).rsqrt()
        # The PLAIN recurrence -- not the block form the kernel runs.
        KV = torch.zeros(D, DV, dtype=torch.float32)
        for t in range(T):
            KV = gam * KV + k[h, t][:, None] * v[h, t][None, :]
            o[h, t] = nrm[t % BLK] * (q[h, t] @ KV)
            if (t + 1) % BLK == 0:
                st[h, t // BLK] = KV
    return {"o": o.reshape(-1).contiguous(), "s": st.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="retention_chunk_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32",
               "gam_ptr": "*fp32", "o_ptr": "*fp32", "s_ptr": "*fp32",
               "H": "constexpr", "NC": "constexpr", "BLK": "constexpr",
               "D": "constexpr", "DV": "constexpr", "TD": "constexpr",
               "TDV": "constexpr", "BLKM1": "constexpr"},
    constexprs={"H": H, "NC": NC, "BLK": BLK, "D": D, "DV": DV,
                "TD": TD, "TDV": TDV, "BLKM1": BLKM1},
    args=[Arg("q", IN, "float32", (H * T * D,)),
          Arg("k", IN, "float32", (H * T * D,)),
          Arg("v", IN, "float32", (H * T * DV,)),
          Arg("gam", IN, "float32", (H,)),
          Arg("o", OUT, "float32", (H * T * DV,)),
          Arg("s", OUT, "float32", (H * NC * D * DV,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
    expect="blocked",
    expect_why="matmul K=16 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
