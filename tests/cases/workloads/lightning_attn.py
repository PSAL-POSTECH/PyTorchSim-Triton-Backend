"""MiniMax-M1's lightning attention -- no softmax, so the block can be split in two.

The model is a linear recurrence over an outer-product state:

    KV_t = lam * KV_{t-1} + k_t v_t^T          lam fixed per head
    o_t  = q_t^T KV_t

Written blockwise over blocks of B, with KV the state entering the block:

    intra  = ((Q K^T) * M) V        M[i, j] = lam^(i-j) for i >= j, else 0
    inter  = (Q * lam^(i+1)) KV
    o      = intra + inter
    KV    <- lam^B KV + (K * lam^(B-1-j))^T V

WHY BOTH HALVES EXIST. The recurrent form is O(T d^2) with a dependency chain T
long -- correct, and it leaves the matmul unit idle. The quadratic form is
O(T^2 d) with no dependency at all -- fast, and it is quadratic. Lightning
attention takes the intra-block part quadratically (B^2 is 256 here and lands on
one dot) and the inter-block part recurrently (T/B steps over a d x d state).
That split is only legal BECAUSE THERE IS NO SOFTMAX: a softmax's normaliser
couples every score in the row to every other, so scores cannot be summed in two
groups and added. Remove it and the two halves are just two terms.

WHAT IS DIFFERENT FROM ssd_chunk.py, THE NEAREST KERNEL. Same chunked shape,
different state and different decay:

  * THE STATE IS AN OUTER-PRODUCT MATRIX, [D, DV], not a vector. Mamba-2's B and
    C are vectors and its per-step contribution `B_t x_t` is a vector, so its
    carried state is [N]. Here the per-step contribution is `k_t v_t^T` and the
    state is a full d x d matrix -- so the inter-block term is a MATMUL against
    the carried value, `tl.dot(Q_scaled, KV)`, and the state update is another
    one, `tl.dot(K^T_scaled, V)`. ssd_chunk.py's boundary terms are broadcasts
    and row sums; here every one of the four terms in the block is a dot.

  * THE DECAY IS FIXED AND COMPILE-TIME. Mamba's whole claim is that `a_t`
    depends on the input, so its decay matrix comes out of a segment sum over
    data. Lightning attention's lam is one constant per head -- an ALiBi-style
    slope, not a parameter -- so the [B, B] decay matrix is built inside the
    kernel from `tl.arange` and a constexpr, with no loads at all. It is
    loop-invariant across every block and every sequence, which is a hoist a
    data-dependent decay can never offer, and whether the backend takes it is
    visible in whether the exponentials appear inside the block loop.

  * MASKING IS THE ONLY CAUSALITY. `M` is lower-triangular by construction, so
    the intra term is causal; the inter term is unconditionally in the past. No
    running max, no rescaling, no second pass -- everything `softmax_causal.py`
    needs for stability is absent because there is nothing to normalise.

THE KV STATES ARE STORED PER BLOCK. `o` alone mixes the intra and inter terms, so
a disagreement in it is ambiguous. The block-boundary states depend only on the
update rule, which separates the recurrence from the quadratic part.

lam < 1 makes every exponential in here a decay in (0, 1]; lam^(B-1-j) weights
the OLDEST element of the block most, which is the direction that looks wrong at
a glance and is right: by the end of the block, element j has already decayed
B-1-j times.

GREEN SINCE 2026-09-04. vectorize refused this with "operands of one
elementwise op are banked on different iteration dims": one buffer a matmul pins
on one axis sits in an all-parallel op beside buffers that follow the op onto the
other. select_lane_axis owned that decision and did not know it -- its
`_is_one_iteration_dim` reported only the disagreements a widen could already
fix, so the ladder never reached step 4 and the refusal fell out a pass later.

The pinned operand here is the op's `outs`, and a relayout mints a new value
while the op already produces that one. So the cut goes on the CONSUMER instead:
the matmul that pins it reads a copy banked its way, and the producing op is free
to follow its own operands. Same trip, no new rewrite. Costs 24 round trips -- its `outs` is pinned by twelve matmuls, so
the consumer cut is paid once per reader.

Four dots per block and a [D,DV] carried state, with the decay matrix built from a constexpr instead of loaded. The reference runs the recurrence, so agreement is evidence the block split is right.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLK, NC = 16, 4                 # block length, block count
T = BLK * NC                    # sequence length
D, DV = 16, 16                  # key dim, value dim -- the state is D x DV
LG = -0.0625                    # log lam: the head's fixed decay slope
LGB = LG * BLK                  # what a whole block of decay comes to
BLKM1 = BLK - 1


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def lightning_attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, s_ptr,
                              BLK: tl.constexpr, NC: tl.constexpr,
                              D: tl.constexpr, DV: tl.constexpr,
                              LG: tl.constexpr, LGB: tl.constexpr,
                              BLKM1: tl.constexpr):
        rb = tl.arange(0, BLK)
        rd = tl.arange(0, D)
        rv = tl.arange(0, DV)

        # THE DECAY MATRIX, built from literals: no loads, loop-invariant.
        i = rb[:, None]
        j = rb[None, :]
        M = tl.where(i >= j, tl.exp(LG * (i - j)), 0.0)         # [BLK, BLK]
        # Weight on the incoming state, and on each element of the outgoing one.
        wi = tl.exp(LG * (rb[:, None] + 1))                     # [BLK, 1]
        wo = tl.exp(LG * (BLKM1 - rb[None, :]))                 # [1, BLK]

        KV = tl.zeros((D, DV), dtype=tl.float32)
        for ck in tl.static_range(NC):
            base = ck * BLK
            qb = tl.load(q_ptr + (base + rb[:, None]) * D + rd[None, :])   # [B,D]
            # K transposed, so the time axis is in-lane for the state update.
            kT = tl.load(k_ptr + (base + rb[None, :]) * D + rd[:, None])   # [D,B]
            vb = tl.load(v_ptr + (base + rb[:, None]) * DV + rv[None, :])  # [B,DV]

            # INTRA: the quadratic half, masked and decayed.
            o = tl.dot(tl.dot(qb, kT) * M, vb)
            # INTER: the linear half, a matmul against the carried state.
            o += tl.dot(qb * wi, KV)
            tl.store(o_ptr + (base + rb[:, None]) * DV + rv[None, :], o)

            # THE RECURRENCE: decay the state a whole block, add this block's
            # contribution with each element decayed by how much of the block
            # is left after it.
            KV = tl.exp(LGB) * KV + tl.dot(kT * wo, vb)
            tl.store(s_ptr + ck * (D * DV) + rd[:, None] * DV + rv[None, :], KV)

    return lightning_attn_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = 1.0 / (D ** 0.5)
    return {"q": torch.randn(T * D, generator=g, dtype=torch.float32) * s,
            "k": torch.randn(T * D, generator=g, dtype=torch.float32) * s,
            "v": torch.randn(T * DV, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    # The RECURRENT form -- deliberately not the blockwise one the kernel runs.
    q = inputs["q"].reshape(T, D)
    k = inputs["k"].reshape(T, D)
    v = inputs["v"].reshape(T, DV)
    lam = float(torch.tensor(LG, dtype=torch.float32).exp())
    KV = torch.zeros(D, DV, dtype=torch.float32)
    o = torch.empty(T, DV, dtype=torch.float32)
    st = torch.empty(NC, D, DV, dtype=torch.float32)
    for t in range(T):
        KV = lam * KV + k[t][:, None] * v[t][None, :]
        o[t] = q[t] @ KV
        if (t + 1) % BLK == 0:
            st[t // BLK] = KV
    return {"o": o.reshape(-1).contiguous(), "s": st.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="lightning_attn_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32",
               "o_ptr": "*fp32", "s_ptr": "*fp32", "BLK": "constexpr",
               "NC": "constexpr", "D": "constexpr", "DV": "constexpr",
               "LG": "constexpr", "LGB": "constexpr", "BLKM1": "constexpr"},
    constexprs={"BLK": BLK, "NC": NC, "D": D, "DV": DV,
                "LG": LG, "LGB": LGB, "BLKM1": BLKM1},
    args=[Arg("q", IN, "float32", (T * D,)),
          Arg("k", IN, "float32", (T * D,)),
          Arg("v", IN, "float32", (T * DV,)),
          Arg("o", OUT, "float32", (T * DV,)),
          Arg("s", OUT, "float32", (NC * D * DV,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
