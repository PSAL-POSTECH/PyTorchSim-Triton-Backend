"""FlashInfer `merge_state` -- a SHARED prefix branch merged into R private ones.

Every request in the batch attends to the same system prompt and then to its own
tokens. Split the attention at that seam and each request produces two softmax
states; the merge puts them back together:

    shared    o_s[r], m_s[r], l_s[r]   over ONE k/v pair of buffers, all r
    suffix    o_u[r], m_u[r], l_u[r]   over request r's own k/v
    m  = max(m_s, m_u);  w_s = l_s e^(m_s-m);  w_u = l_u e^(m_u-m)
    out[r] = (o_s[r] w_s + o_u[r] w_u) / (w_s + w_u)

WHY SPLIT IT AT ALL -- CASCADE INFERENCE. R requests sharing a 2000-token system
prompt, attended the ordinary way, read those 2000 keys R times: the shared KV is
re-fetched once per request and DRAM bandwidth is what decode is short of. Split
instead, and the shared half is a MATMUL -- one [SP, D] tile of keys against R
queries, loaded once, resident -- while only the small private half stays
memory-bound. FlashInfer's `MultiLevelCascadeAttentionWrapper` is exactly this,
and `merge_state` is the operator that makes the split legal.

THE MERGE IS THE PART THAT CANNOT BE HANDWAVED. Softmax does not decompose by
concatenation: each half normalised by its own maximum and its own denominator,
so before they can be added both have to be rescaled to the joint maximum. That
is the same algebra `flash_decode_splitk.py` performs -- and this file exists
BECAUSE OF ONE STRUCTURAL DIFFERENCE:

    flash_decode_splitk    two branches of ONE sequence's own KV. Both states
                           are per-sequence, both are computed from tiles of the
                           same shape, and the merge is symmetric in every index.
    cascade_merge_state    one branch is SHARED ACROSS REQUESTS. Its k/v tile has
                           no request axis at all -- it enters the score matrix
                           broadcast, `k_sh[None, :, :] * q[:, None, :]` -- while
                           the other branch's k/v is indexed BY request. So the
                           merge combines a rank-2 [R, D] state with another
                           rank-2 [R, D] state whose two halves were produced by
                           reductions over differently-ranked operands.

  The shared branch's m and l are still per-request even though its KV is not:
  the queries differ, so the same keys give R different maxima. That asymmetry --
  shared data, private statistics -- is what a backend can get wrong here and
  cannot get wrong in the split-k case, where everything is private.

WHAT ELSE IS NEARBY. `paged_attention.py` has the same rank-3 [R, S, D] reduce
over a data-addressed tile; it has no merge and no second branch.
`gqa_repeat_kv.py` broadcasts KV across query heads, but by REPLICATING it into
the same shape, not by leaving one operand a rank lower through the reduction.

THE TWO BRANCHES ARE DELIBERATELY UNBALANCED IN SCALE (the shared keys are 2.5x),
so `m_s` and `m_u` are far apart and dropping either `e^(m-m)` factor is a wrong
answer rather than a rounding. A shared prefix really is the higher-scoring half
in practice -- it is what the model was told to pay attention to.

WHAT WAS PINNED. R (requests in the batch), SP (shared prefix length) and SU
(private suffix length) are constants. In FlashInfer all three are runtime, and
SU in particular differs per request; here every request has the same suffix
length, which is the one thing this file does not ask. It does not need to:
per-request ragged suffixes are `varlen_packed_attn.py`'s question, and mixing
them in would make a merge failure and a boundary failure indistinguishable.
What stays fully general is the thing being tested -- the states are real,
computed from real KV, and the merge is checked against a softmax over the
CONCATENATION of the two branches rather than against itself.

WHERE IT STOPS TODAY, and it is the shared branch that does it. `k_sh[None, :,
:] * q[:, None, :]` puts a [SP, D] tile with NO request axis into an op whose
other operand is banked on the request axis, and once the unit extents come off
upstream the shared tile is not materialised per request any more: it is one
buffer in one bank, stamped ONE_LANE, multiplied into a [R, SP, D] tile banked on
d0. vectorize used to take ONE_LANE from that one slot and build a one-trip
vlane loop -- `verify: numerical mismatch nan`, with nothing reported -- and now
refuses by name. `tests/cases/mechanism/tile/tile_one_bank_row_times_banked_tile.py`
pins the refusal in the small. The answer is the replication the file's own
opening argument describes (shared data, private statistics: the shared KV wants
to be in every bank), and minting that copy is fit_layout_to_lanes's.

One branch broadcast across requests, one indexed by request, merged with per-request maxima. Compared against a softmax over the concatenation, so a dropped rescale is a large error, not a drift.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, SP, SU, D = 8, 16, 16, 16    # requests, shared prefix keys, suffix keys, dim


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def cascade_merge_state_kernel(q_ptr, ks_ptr, vs_ptr, ku_ptr, vu_ptr,
                                   o_ptr, R: tl.constexpr, SP: tl.constexpr,
                                   SU: tl.constexpr, D: tl.constexpr):
        rr = tl.arange(0, R)
        rp = tl.arange(0, SP)
        ru = tl.arange(0, SU)
        rd = tl.arange(0, D)
        scale = 1.0 / (D ** 0.5)

        q = tl.load(q_ptr + rr[:, None] * D + rd[None, :])            # [R, D]

        # SHARED BRANCH. One [SP, D] tile of keys for the whole batch: it enters
        # broadcast over the request axis, with no request index in its address.
        ks = tl.load(ks_ptr + rp[:, None] * D + rd[None, :])          # [SP, D]
        vs = tl.load(vs_ptr + rp[:, None] * D + rd[None, :])
        ss = tl.sum(ks[None, :, :] * q[:, None, :], axis=2) * scale   # [R, SP]
        ms = tl.max(ss, axis=1)                                       # [R]
        es = tl.exp(ss - ms[:, None])
        ls = tl.sum(es, axis=1)                                       # [R]
        os_ = tl.sum(es[:, :, None] * vs[None, :, :], axis=1) / ls[:, None]

        # SUFFIX BRANCH. Same reduction, but the k/v tile carries the request
        # axis in its ADDRESS: [R, SU, D], outer stride SU*D.
        uoff = (rr[:, None, None] * (SU * D)
                + ru[None, :, None] * D + rd[None, None, :])
        ku = tl.load(ku_ptr + uoff)
        vu = tl.load(vu_ptr + uoff)
        su = tl.sum(ku * q[:, None, :], axis=2) * scale               # [R, SU]
        mu = tl.max(su, axis=1)
        eu = tl.exp(su - mu[:, None])
        lu = tl.sum(eu, axis=1)
        ou = tl.sum(eu[:, :, None] * vu, axis=1) / lu[:, None]

        # THE MERGE. Both states are [R, D] with per-request m and l -- the
        # shared branch's statistics are private even though its KV was not.
        m = tl.maximum(ms, mu)
        ws = ls * tl.exp(ms - m)
        wu = lu * tl.exp(mu - m)
        out = (os_ * ws[:, None] + ou * wu[:, None]) / (ws + wu)[:, None]
        tl.store(o_ptr + rr[:, None] * D + rd[None, :], out)

    return cascade_merge_state_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)
    return {"q": rn(R * D),
            # The shared prefix scores higher, so the two maxima are far apart.
            "ks": rn(SP * D) * 2.5, "vs": rn(SP * D),
            "ku": rn(R * SU * D), "vu": rn(R * SU * D)}


def reference(inputs):
    import torch
    q = inputs["q"].reshape(R, D)
    ks = inputs["ks"].reshape(SP, D)
    vs = inputs["vs"].reshape(SP, D)
    ku = inputs["ku"].reshape(R, SU, D)
    vu = inputs["vu"].reshape(R, SU, D)
    # Checked against ONE softmax over the concatenated keys, so the merge is
    # compared with what it claims to equal and not with its own algebra.
    k = torch.cat([ks[None].expand(R, SP, D), ku], dim=1)      # [R, SP+SU, D]
    v = torch.cat([vs[None].expand(R, SP, D), vu], dim=1)
    s = torch.einsum("rd,rsd->rs", q, k) / (D ** 0.5)
    o = torch.einsum("rs,rsd->rd", s.softmax(dim=1), v)
    return {"o": o.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="cascade_merge_state_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "ks_ptr": "*fp32", "vs_ptr": "*fp32",
               "ku_ptr": "*fp32", "vu_ptr": "*fp32", "o_ptr": "*fp32",
               "R": "constexpr", "SP": "constexpr", "SU": "constexpr",
               "D": "constexpr"},
    constexprs={"R": R, "SP": SP, "SU": SU, "D": D},
    args=[Arg("q", IN, "float32", (R * D,)),
          Arg("ks", IN, "float32", (SP * D,)),
          Arg("vs", IN, "float32", (SP * D,)),
          Arg("ku", IN, "float32", (R * SU * D,)),
          Arg("vu", IN, "float32", (R * SU * D,)),
          Arg("o", OUT, "float32", (R * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
