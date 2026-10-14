"""Flux / SD3's MMDiT double-stream block -- two towers with SEPARATE weights
that meet inside ONE attention and then split apart again.

    qi, ki, vi = img @ Wqkv_img          image stream, its own projection
    qt, kt, vt = txt @ Wqkv_txt          text stream, its own projection
    A          = softmax(cat(qt,qi) cat(kt,ki)^T / sqrt(D)) cat(vt,vi)
    at, ai     = A[:LT], A[LT:]          split back on the SEQUENCE axis
    img        = img + ai @ Wo_img ;  img += mlp_img(img)
    txt        = txt + at @ Wo_txt ;  txt += mlp_txt(txt)

WHY TWO SETS OF WEIGHTS. Text tokens and image patches have different
statistics -- different scales, different rank, different useful nonlinearity --
and Esser et al. (SD3, 2024) found that giving each modality its own projections
beats a shared tower at every size they trained. But the two must still SEE each
other, and cross-attention (one modality queries the other) is asymmetric: the
text never gets to look at the image. MMDiT's answer is to concatenate the
sequences for the attention only. Every token attends to every token, in one
softmax, and everything else in the block is per-stream. Flux's 19 double blocks
are built this way and its 38 single blocks (flux_single_block.py) are the
merged-stream continuation of the same tensor.

WHAT IS STRUCTURALLY NEW HERE. Every other multi-branch kernel in this suite has
branches that differ by DATA and share their weights, or share data and differ by
weight. This one has two branches that share NEITHER -- separate inputs, separate
Wqkv, separate Wo, separate MLP -- and are nonetheless not independent, because
one reduction (the softmax denominator) spans both. So the backend sees two
disjoint dataflow graphs joined at exactly one point and split again, which is a
DIAMOND WITH DIFFERENT WEIGHTS ON THE TWO SIDES:

  swiglu.py            two matmuls, ONE x tile, joined elementwise
  flux_single_block    ONE matmul, four column readers, joined by a concat
  this                 two full sub-networks, joined by ONE softmax over the
                       concatenation of their sequence axes, then re-split

AND THE SEQUENCE AXIS IS THE LANE AXIS, which is why the join costs something.
The concatenation is along the axis the banks hold, so `cat(kt, ki)` is a tile
whose rows come from two different producers with two different weight matrices.
Materialising it would be a relayout; the kernel instead does what every flash
kernel does with its KV blocks and keeps the two blocks separate, running the
softmax over their UNION:

    m = max(rowmax(q kt^T), rowmax(q ki^T))
    l = rowsum(exp(s_t - m)) + rowsum(exp(s_i - m))
    o = (p_t @ vt + p_i @ vi) / l

which is the same number as one softmax over the concatenation, and the reason
attention block-tiles at all. What is different from attention.py is that the two
blocks are not two iterations of one loop over one K tensor -- they are two
DIFFERENT tensors produced by two different GEMMs, so the running max has to be
combined across values that were never in the same buffer.

LI != LT ON PURPOSE. Flux runs 512 text tokens against 4096 image patches; making
them equal here would let a lowering fold the two attention blocks into one
square and still be right. They are 32 and 16, so every one of the four
score tiles has a different shape -- [LI,LT], [LI,LI], [LT,LT], [LT,LI] -- and the
two `p @ v` products contract over different lengths into the same accumulator.

k IS TRANSPOSED WITH tl.trans, NOT LOADED TRANSPOSED. attention.py gets its
[D, N] operand by addressing DRAM, which it can because K is an input. Here k is
a MATMUL RESULT, so the layout the matrix unit pinned on its output has to be
flipped in place before it can be an operand again -- a `tt.trans` on a value
that never reaches memory. That is the one thing this file asks for that no other
attention kernel in the suite does.

GREEN SINCE 2026-09-04, AND WHAT IT WAS OPEN ON. It carried expect="open" with
"two operands of one elementwise op are banked on different iteration dims" --
one of the four score tiles reached vectorize with its two readers wanting
opposite axes, and select_lane_axis had no turn to offer that fixed it. It does
now: `conflicts()` joined the turn's yardstick, which is what lets a turn be
taken for removing a conflict rather than only a contradiction. Two relayouts
went with it, and it is one of the two kernels that measure that change (the
other is chunked_prefill_attn, which measured the half of it that was wrong).

NOT MODELLED: the per-stream AdaLN-Zero modulation and its two gates
(adaln_zero.py), the 2-D RoPE applied to the joint q and k (rope_2d.py), and
QK-RMSNorm (qk_norm.py). GELU is tanh-approximate, matching the block's
`nn.GELU(approximate="tanh")`, and written out as `2 sigmoid(2u) - 1` so a
saturating exp gives -1 rather than a NaN.

Two sub-networks with no shared weight, joined by ONE softmax over the concatenation of their sequence axes and split again. Four score tiles of four different shapes, a running max combined across two tensors that were never in one buffer, and a tl.trans on a matmul RESULT rather than a transposed DRAM read. Green since the turn's yardstick counts conflicts; it was open on two operands of one elementwise op banked on different iteration dims.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

LI = 32                 # image patches -- the long stream
LT = 16                 # text tokens -- the short one, deliberately different
D = 16                  # model width == head dim, one head
MH = 2 * D              # per-stream MLP hidden
SCALE = 0.25            # 1 / sqrt(D)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def flux_dual_stream_attn_kernel(img_ptr, txt_ptr, wqkv_ptr, wo_ptr,
                                     w1_ptr, w2_ptr, img_o_ptr, txt_o_ptr,
                                     LI: tl.constexpr, LT: tl.constexpr,
                                     D: tl.constexpr, MH: tl.constexpr,
                                     SCALE: tl.constexpr):
        ri = tl.arange(0, LI)
        rt = tl.arange(0, LT)
        rd = tl.arange(0, D)
        rm = tl.arange(0, MH)

        img = tl.load(img_ptr + ri[:, None] * D + rd[None, :])        # [LI, D]
        txt = tl.load(txt_ptr + rt[:, None] * D + rd[None, :])        # [LT, D]

        # TWO PROJECTIONS, TWO WEIGHT TENSORS. Stream 0 is image, 1 is text;
        # 3D is not a power of two, so q/k/v are three column-offset loads of
        # each [D, 3D] weight rather than one wide arange.
        wqi = wqkv_ptr + 0 * D * 3 * D
        wqt = wqkv_ptr + 1 * D * 3 * D
        off = rd[:, None] * (3 * D) + rd[None, :]
        qi = tl.dot(img, tl.load(wqi + off + 0 * D))
        ki = tl.dot(img, tl.load(wqi + off + 1 * D))
        vi = tl.dot(img, tl.load(wqi + off + 2 * D))
        qt = tl.dot(txt, tl.load(wqt + off + 0 * D))
        kt = tl.dot(txt, tl.load(wqt + off + 1 * D))
        vt = tl.dot(txt, tl.load(wqt + off + 2 * D))

        # THE JOIN. k is a matmul RESULT, so the transpose is on a value:
        # the array's output layout has to be flipped without a round trip.
        kit = tl.trans(ki)                                            # [D, LI]
        ktt = tl.trans(kt)                                            # [D, LT]

        # Image queries over the concatenated key sequence [txt ; img].
        s_it = tl.dot(qi, ktt) * SCALE                                # [LI, LT]
        s_ii = tl.dot(qi, kit) * SCALE                                # [LI, LI]
        m = tl.maximum(tl.max(s_it, axis=1), tl.max(s_ii, axis=1))
        p_it = tl.exp(s_it - m[:, None])
        p_ii = tl.exp(s_ii - m[:, None])
        l = tl.sum(p_it, axis=1) + tl.sum(p_ii, axis=1)
        a_i = (tl.dot(p_it, vt) + tl.dot(p_ii, vi)) / l[:, None]      # [LI, D]

        # Text queries over the SAME concatenated sequence -- different query
        # tile, different score shapes, the same two value tensors.
        s_tt = tl.dot(qt, ktt) * SCALE                                # [LT, LT]
        s_ti = tl.dot(qt, kit) * SCALE                                # [LT, LI]
        m = tl.maximum(tl.max(s_tt, axis=1), tl.max(s_ti, axis=1))
        p_tt = tl.exp(s_tt - m[:, None])
        p_ti = tl.exp(s_ti - m[:, None])
        l = tl.sum(p_tt, axis=1) + tl.sum(p_ti, axis=1)
        a_t = (tl.dot(p_tt, vt) + tl.dot(p_ti, vi)) / l[:, None]      # [LT, D]

        # THE SPLIT. From here the two streams share nothing at all.
        wod = rd[:, None] * D + rd[None, :]
        w1d = rd[:, None] * MH + rm[None, :]
        w2d = rm[:, None] * D + rd[None, :]

        h = img + tl.dot(a_i, tl.load(wo_ptr + 0 * D * D + wod))
        u = tl.dot(h, tl.load(w1_ptr + 0 * D * MH + w1d))
        g = 0.7978845608028654 * (u + 0.044715 * u * u * u)
        act = 0.5 * u * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * g)) - 1.0))
        h = h + tl.dot(act, tl.load(w2_ptr + 0 * MH * D + w2d))
        tl.store(img_o_ptr + ri[:, None] * D + rd[None, :], h)

        h = txt + tl.dot(a_t, tl.load(wo_ptr + 1 * D * D + wod))
        u = tl.dot(h, tl.load(w1_ptr + 1 * D * MH + w1d))
        g = 0.7978845608028654 * (u + 0.044715 * u * u * u)
        act = 0.5 * u * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * g)) - 1.0))
        h = h + tl.dot(act, tl.load(w2_ptr + 1 * MH * D + w2d))
        tl.store(txt_o_ptr + rt[:, None] * D + rd[None, :], h)

    return flux_dual_stream_attn_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)

    def rn(n, s):
        return torch.randn(n, generator=g, dtype=torch.float32) * s

    return {"img": rn(LI * D, 0.5), "txt": rn(LT * D, 0.5),
            "wqkv": rn(2 * D * 3 * D, 0.2), "wo": rn(2 * D * D, 0.2),
            "w1": rn(2 * D * MH, 0.2), "w2": rn(2 * MH * D, 0.2)}


def reference(inputs):
    import torch
    img = inputs["img"].reshape(LI, D)
    txt = inputs["txt"].reshape(LT, D)
    wqkv = inputs["wqkv"].reshape(2, D, 3 * D)
    wo = inputs["wo"].reshape(2, D, D)
    w1 = inputs["w1"].reshape(2, D, MH)
    w2 = inputs["w2"].reshape(2, MH, D)

    zi = img @ wqkv[0]
    zt = txt @ wqkv[1]
    qi, ki, vi = zi[:, 0:D], zi[:, D:2 * D], zi[:, 2 * D:3 * D]
    qt, kt, vt = zt[:, 0:D], zt[:, D:2 * D], zt[:, 2 * D:3 * D]

    # The reference DOES concatenate: one softmax over the joint sequence, which
    # is the definition the block-wise form in the kernel has to reproduce.
    q = torch.cat([qt, qi], 0)
    k = torch.cat([kt, ki], 0)
    v = torch.cat([vt, vi], 0)
    a = torch.softmax((q @ k.t()) * SCALE, dim=1) @ v
    a_t, a_i = a[:LT], a[LT:]

    def gelu(z):
        return torch.nn.functional.gelu(z, approximate="tanh")

    hi = img + a_i @ wo[0]
    hi = hi + gelu(hi @ w1[0]) @ w2[0]
    ht = txt + a_t @ wo[1]
    ht = ht + gelu(ht @ w1[1]) @ w2[1]
    return {"img_o": hi.reshape(-1).contiguous(),
            "txt_o": ht.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="flux_dual_stream_attn_kernel", kernel=kernel,
    signature={"img_ptr": "*fp32", "txt_ptr": "*fp32", "wqkv_ptr": "*fp32",
               "wo_ptr": "*fp32", "w1_ptr": "*fp32", "w2_ptr": "*fp32",
               "img_o_ptr": "*fp32", "txt_o_ptr": "*fp32", "LI": "constexpr",
               "LT": "constexpr", "D": "constexpr", "MH": "constexpr",
               "SCALE": "constexpr"},
    constexprs={"LI": LI, "LT": LT, "D": D, "MH": MH, "SCALE": SCALE},
    args=[Arg("img", IN, "float32", (LI * D,)),
          Arg("txt", IN, "float32", (LT * D,)),
          Arg("wqkv", IN, "float32", (2 * D * 3 * D,)),
          Arg("wo", IN, "float32", (2 * D * D,)),
          Arg("w1", IN, "float32", (2 * D * MH,)),
          Arg("w2", IN, "float32", (2 * MH * D,)),
          Arg("img_o", OUT, "float32", (LI * D,)),
          Arg("txt_o", OUT, "float32", (LT * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
    expect="blocked",
    expect_why="matmul K=16 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
