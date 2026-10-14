"""KIVI -- keys quantised PER CHANNEL, values PER TOKEN. Two reduction axes, one
attention.

    sk[d] = max_s |K[d, s]| / 7        keys:   reduce over TOKENS, one scale per
                                               feature channel
    sv[s] = max_d |V[s, d]| / 7        values: reduce over CHANNELS, one scale
                                               per token
    p = softmax((q @ Kdeq))            Kdeq is [D, S], so no transpose
    o = p @ Vdeq

WHY THE TWO HALVES OF THE SAME CACHE GET OPPOSITE TREATMENT
  The KV cache is the memory bill of long-context decoding -- at 128k tokens it
  is larger than the weights -- so it gets quantised, and 2 bits is the target.
  Liu et al. measured what actually breaks and found the two tensors are not
  alike:

    KEYS have per-CHANNEL outliers. A handful of feature dimensions carry
    magnitudes an order above the rest, and they carry them in EVERY token --
    the same channels, consistently. Quantising a key vector per token means one
    scale covering both the outlier channels and the ordinary ones, and the
    ordinary ones collapse to a couple of levels. Quantising per channel puts
    each outlier in its own scale where it costs nothing.

    VALUES have no such structure -- their magnitudes are flat across channels.
    But the value tensor is consumed by `p @ V`, a sum over TOKENS, and a
    per-channel value scale would put a different scale on every term of that
    sum. Per token keeps the scale constant along the contraction, so it factors
    out of each row's contribution and the error stays bounded per token instead
    of accumulating across the cache.

  So the axis choice is not symmetric and is not a tuning knob: it follows from
  where the outliers are for K and from which axis is contracted for V.

THE LAYOUT CONSEQUENCE, AND IT IS THE WHOLE POINT FOR THIS BACKEND
  A reduction here has to stay on the INNER axis of a 2-D tile -- dim 0 is the
  lane axis and there is no lane-crossing primitive. `max over tokens` and
  `max over channels` cannot both be inner on one [S, D] tile.

  KIVI's answer is the one real implementations use: STORE THE KEY CACHE
  CHANNEL-MAJOR, as [D, S], and the value cache token-major, as [S, D]. Then
  both reductions are `tl.max(..., axis=1)` on their own tile and neither
  crosses a lane. Two identical-looking folds that mean opposite things because
  the two tensors are laid out differently -- that is what this kernel puts in
  front of the backend, and no other file here has two tensors with
  disagreeing majorness in the same kernel.

  The layout pays for itself twice: with K already [D, S], the score matrix is
  `q @ Kdeq` with D contracted and NO transpose, which is the shape the matrix
  unit wants anyway. attention.py and paged_attention.py both carry K as
  [S, D] and eat the transpose.

INT4 IN INT32 CONTAINERS. Real KIVI is 2 or 4 bits packed; this ABI has no
sub-byte type and no i8, so the quantised tensors are int32-containered values in
[-7, 7]. Rounding is `floor(x/s + 0.5)` written out rather than a `round` call,
so the reference can state exactly the same thing -- half-to-even vs half-up is a
real difference and it should not be hiding inside an intrinsic. `kq` and `vq`
are stored: they are integer-valued and must be bit-exact, so a mismatch in them
is a lowering fault and only `o` can carry arithmetic drift.

RESIDUAL. Real KIVI keeps the most recent few tokens unquantised in a fp16
residual buffer, because the group a token belongs to is not full until the group
closes. That is a control-flow concern, not an arithmetic one, and
sliding_window.py already covers the ring it needs; it is left out here.

KIVI: two quantisation schemes on opposite axes in one kernel, made to coexist by storing K channel-major and V token-major so both folds stay on the inner axis. kq/vq are integral and exact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, S, D = 16, 32, 16        # queries, cached tokens, head dim
QMAX = 7.0                  # int4 symmetric, so levels are -7 .. 7


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kv_cache_quant_mixed_axis_kernel(q_ptr, kt_ptr, v_ptr, kq_ptr, vq_ptr,
                                         o_ptr, T: tl.constexpr,
                                         S: tl.constexpr, D: tl.constexpr,
                                         QMAX: tl.constexpr):
        rt = tl.arange(0, T)
        rs = tl.arange(0, S)
        rd = tl.arange(0, D)

        # ---- KEYS: channel-major [D, S], so "per channel" is the inner fold.
        kt = tl.load(kt_ptr + rd[:, None] * S + rs[None, :])          # [D, S]
        sk = tl.maximum(tl.max(tl.abs(kt), axis=1), 1e-20) / QMAX     # [D]
        kq = tl.floor(kt / sk[:, None] + 0.5)
        kq = tl.minimum(tl.maximum(kq, -QMAX), QMAX)
        tl.store(kq_ptr + rd[:, None] * S + rs[None, :], kq.to(tl.int32))
        kd = kq * sk[:, None]                                          # [D, S]

        # ---- VALUES: token-major [S, D], so "per token" is the inner fold.
        # Same tl.max(axis=1), opposite meaning -- the difference is the layout.
        v = tl.load(v_ptr + rs[:, None] * D + rd[None, :])            # [S, D]
        sv = tl.maximum(tl.max(tl.abs(v), axis=1), 1e-20) / QMAX      # [S]
        vq = tl.floor(v / sv[:, None] + 0.5)
        vq = tl.minimum(tl.maximum(vq, -QMAX), QMAX)
        tl.store(vq_ptr + rs[:, None] * D + rd[None, :], vq.to(tl.int32))
        vd = vq * sv[:, None]                                          # [S, D]

        # ---- One attention across the pair. K is already [D, S], so the score
        # matmul contracts D with no transpose; the key scale broadcasts along
        # the CONTRACTED axis and so is inside the dot, while the value scale
        # broadcasts along the contracted axis of the second dot and so is
        # inside that one. Neither can become an epilogue.
        q = tl.load(q_ptr + rt[:, None] * D + rd[None, :])            # [T, D]
        sc = tl.dot(q, kd) * (1.0 / 4.0)                              # 1/sqrt(D)
        m = tl.max(sc, axis=1)
        ex = tl.exp(sc - m[:, None])
        p = ex / tl.sum(ex, axis=1)[:, None]
        tl.store(o_ptr + rt[:, None] * D + rd[None, :], tl.dot(p, vd))

    return kv_cache_quant_mixed_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Keys get the per-channel outliers that motivate the whole scheme: a few
    # feature dims are 20x the rest, in every token.
    kt = torch.randn(D, S, generator=g, dtype=torch.float32)
    heavy = torch.zeros(D, 1)
    heavy[2] = heavy[9] = 1.0
    kt = kt * (1.0 + 19.0 * heavy)
    return {"q": torch.randn(T, D, generator=g, dtype=torch.float32),
            "kt": kt,
            "v": torch.randn(S, D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    kt, v, q = inputs["kt"], inputs["v"], inputs["q"]

    sk = kt.abs().amax(dim=1).clamp_min(1e-20) / QMAX          # per CHANNEL
    kq = torch.floor(kt / sk[:, None] + 0.5).clamp(-QMAX, QMAX)
    kd = kq * sk[:, None]

    sv = v.abs().amax(dim=1).clamp_min(1e-20) / QMAX           # per TOKEN
    vq = torch.floor(v / sv[:, None] + 0.5).clamp(-QMAX, QMAX)
    vd = vq * sv[:, None]

    sc = (q @ kd) * (1.0 / 4.0)
    p = torch.softmax(sc, dim=-1)
    return {"kq": kq.to(torch.int32).contiguous(),
            "vq": vq.to(torch.int32).contiguous(),
            "o": (p @ vd).contiguous()}


SPEC = KernelSpec(
    name="kv_cache_quant_mixed_axis_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "kt_ptr": "*fp32", "v_ptr": "*fp32",
               "kq_ptr": "*i32", "vq_ptr": "*i32", "o_ptr": "*fp32",
               "T": "constexpr", "S": "constexpr", "D": "constexpr",
               "QMAX": "constexpr"},
    constexprs={"T": T, "S": S, "D": D, "QMAX": QMAX},
    args=[Arg("q", IN, "float32", (T, D)),
          Arg("kt", IN, "float32", (D, S)),
          Arg("v", IN, "float32", (S, D)),
          Arg("kq", OUT, "int32", (D, S)),
          Arg("vq", OUT, "int32", (S, D)),
          Arg("o", OUT, "float32", (T, D))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
