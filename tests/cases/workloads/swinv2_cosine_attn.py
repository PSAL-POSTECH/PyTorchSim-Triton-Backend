"""Swin Transformer V2's scaled cosine attention -- the dot product, normalised.

    attn = normalize(q, dim=-1) @ normalize(k, dim=-1).T
    attn = attn * clamp(logit_scale, max=log(1/0.01)).exp()
    out  = softmax(attn) @ v

torch equivalent: WindowAttention.forward in the official Swin-V2, minus the
log-CPB bias (t5_rel_bias.py already covers an additive per-pair bias).

WHY THE SCALE MOVED FROM A CONSTANT TO A PARAMETER. Liu et al. tried to scale
Swin from 88M to 3B and it diverged: activation magnitudes grew with depth until
a few (query, key) pairs dominated every attention map -- the paper shows the
ratio between the largest and the mean logit reaching 10^4 in the deep layers.
The cause is that `q . k / sqrt(d)` grows with ||q|| ||k||, which nothing bounds.

  COSINE ATTENTION BOUNDS IT BY CONSTRUCTION. After normalisation the score is
  in [-1, 1] no matter how large the activations get, so the amplitude problem
  is gone -- and with it the useful dynamic range, because a logit range of 2 is
  far too flat for a softmax. Hence the LEARNED TEMPERATURE that puts it back,
  one per head, clamped so it cannot exceed 100x and re-create the problem it
  was introduced to fix.

WHAT IS STRUCTURALLY NEW HERE, against attention.py:

  A REDUCTION FEEDING A MATMUL OPERAND. attention.py reduces the matmul's
  OUTPUT (the row max, the row sum). Here both INPUTS are reduced first --
  sum-of-squares along the head dim, one rsqrt per token, applied back as a
  per-row scale -- and only then does `tl.dot` see them. So the reduction's
  chosen lane axis has to agree with the systolic array's operand layout on the
  way IN, which is the reverse of the constraint attention.py describes.

  AND FOR K IT AGREES WITH THE WRONG ONE. `tl.dot` wants K as (D, N), where the
  head dim is the OUTER axis -- and the normalisation folds the head dim, which
  in that tile is not the inner axis. So K IS LOADED TWICE: once as (N, D) to
  fold, once transposed by its addressing to multiply, with the rank-1 result of
  the first load applied as `kn[None, :]` to the second. That is a fold and a
  matmul disagreeing about a layout inside one kernel, and the resolution costs
  a second read rather than a relayout.

  A SCALAR FROM MEMORY MULTIPLYING A MATMUL RESULT. `logit_scale` is per head,
  so inside the static head loop it is a single loaded element -- not a vector,
  not a constexpr. `clamp` then `exp` on it, and the whole (N, N) score tile is
  scaled by that one value. attention.py's SCALE is a constexpr folded into the
  IR; this one cannot be.

The heads are a static loop for the usual reason -- a head axis would make the
tile rank 3 -- and there are two of them so that the per-head scalar is actually
per head and not a constant in disguise.

GREEN SINCE 2026-09-04. vectorize refused it with "one operand of this
elementwise op is in a single bank (ONE_LANE) while another is banked across the
lanes" -- two whole tiles whose layouts disagree, with no broadcast row to widen,
so the answer is a cut rather than a replication. select_lane_axis could not plan
one: `dim_splits` bucketed operands by iteration dim and dropped every operand
with no axis, and ONE_LANE is exactly that. It is a dim of its own now -- a whole
tile in one bank is a layout, not the absence of one -- and the cut goes on the
operand a matmul pins, at the matmul that reads it. Costs 2 round trips.

WHAT IS STILL REFUSED BESIDE IT. When the op settles on ONE_LANE, only freeing
the pin at its READER is offered. Cutting a pinned INPUT at the op itself also
compiles, and closes retention_chunk at 26 round trips -- but on ssd_chunk the
same path answers 40.6174, turning a refusal into a wrong number, which
philosophy 12 ranks as the worse outcome. Telling those two cases apart is what
retention_chunk and ssd_chunk are still open on.

Attention whose scores are bounded by construction: both matmul OPERANDS are reduced before the dot (K twice, because the fold and the systolic layout disagree), and a per-head scalar from memory scales the result.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

HEADS = 2
N = 32                          # tokens in the window
D = 16                          # head dim
LOGIT_MAX = 4.605170185988092   # log(1 / 0.01), the paper's cap


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def swinv2_cosine_attn_kernel(q_ptr, k_ptr, v_ptr, ls_ptr, o_ptr,
                                  HEADS: tl.constexpr, N: tl.constexpr,
                                  D: tl.constexpr, LOGIT_MAX: tl.constexpr):
        n = tl.arange(0, N)
        d = tl.arange(0, D)

        for h in tl.static_range(HEADS):
            base = h * N * D

            # Q: fold the head dim (inner axis of an (N, D) tile), then scale
            # the rows back. The reduction feeds a MATMUL OPERAND, not a
            # matmul result.
            q = tl.load(q_ptr + base + n[:, None] * D + d[None, :])
            qn = q * (1.0 / tl.sqrt(tl.sum(q * q, axis=1)))[:, None]

            # K TWICE. The fold wants D inner; tl.dot wants D outer. The first
            # load is only there to produce the rank-1 norms, which are then
            # applied along the transposed tile's LANE axis.
            kf = tl.load(k_ptr + base + n[:, None] * D + d[None, :])
            kinv = 1.0 / tl.sqrt(tl.sum(kf * kf, axis=1))
            kt = tl.load(k_ptr + base + n[None, :] * D + d[:, None])
            kt = kt * kinv[None, :]

            # The learned per-head temperature: one element out of memory,
            # clamped, exponentiated, and applied to the whole score tile.
            ls = tl.exp(tl.minimum(tl.load(ls_ptr + h), LOGIT_MAX))
            s = tl.dot(qn, kt) * ls

            m = tl.max(s, axis=1)
            e = tl.exp(s - m[:, None])
            p = e / tl.sum(e, axis=1)[:, None]

            v = tl.load(v_ptr + base + n[:, None] * D + d[None, :])
            tl.store(o_ptr + base + n[:, None] * D + d[None, :], tl.dot(p, v))

    return swinv2_cosine_attn_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    nd = HEADS * N * D
    # Deliberately lopsided magnitudes per head -- that is the failure cosine
    # attention exists to survive, and raw QK^T would blow up on it.
    q = torch.randn(HEADS, N, D, generator=g, dtype=torch.float32)
    q[1] *= 8.0
    k = torch.randn(HEADS, N, D, generator=g, dtype=torch.float32)
    k[1] *= 8.0
    # One head under the cap (the paper's init, log 10) and one over it, so the
    # clamp is exercised rather than merely present.
    ls = torch.tensor([2.302585092994046, 5.5], dtype=torch.float32)
    return {"q": q.reshape(-1).contiguous(), "k": k.reshape(-1).contiguous(),
            "v": torch.randn(nd, generator=g, dtype=torch.float32),
            "ls": ls}


def reference(inputs):
    import torch
    F = torch.nn.functional
    q = inputs["q"].reshape(HEADS, N, D)
    k = inputs["k"].reshape(HEADS, N, D)
    v = inputs["v"].reshape(HEADS, N, D)
    ls = inputs["ls"].clamp(max=LOGIT_MAX).exp().reshape(HEADS, 1, 1)
    attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
    p = torch.softmax(attn * ls, dim=-1)
    return {"o": (p @ v).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="swinv2_cosine_attn_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32",
               "ls_ptr": "*fp32", "o_ptr": "*fp32", "HEADS": "constexpr",
               "N": "constexpr", "D": "constexpr", "LOGIT_MAX": "constexpr"},
    constexprs={"HEADS": HEADS, "N": N, "D": D, "LOGIT_MAX": LOGIT_MAX},
    args=[Arg("q", IN, "float32", (HEADS * N * D,)),
          Arg("k", IN, "float32", (HEADS * N * D,)),
          Arg("v", IN, "float32", (HEADS * N * D,)),
          Arg("ls", IN, "float32", (HEADS,)),
          Arg("o", OUT, "float32", (HEADS * N * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # Two chained matmuls plus two rsqrts, none of it in torch's order.
    rtol=1e-3, atol=1e-3,
)
