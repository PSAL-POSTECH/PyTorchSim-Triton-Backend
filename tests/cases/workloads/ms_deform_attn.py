"""Multi-scale deformable attention -- RT-DETR's and D-FINE's whole decoder.

    for each query q, each level l, each of P learned points:
        p     = sampling_loc[q, l, i]           normalised, in [0, 1]
        v_qli = bilinear(value_l, p * (W_l, H_l) - 0.5)
    out[q] = sum over l, i of  attn_w[q, l, i] * v_qli

torch equivalent: the `ms_deform_attn_core_pytorch` in Deformable-DETR --
F.grid_sample per level, then a weighted sum over the L*P samples.

ATTENTION WITH NO QK^T IN IT. attention.py computes a score for every (query,
key) pair and is quadratic; this one never forms a score at all. The decoder
PREDICTS where to look -- P offsets per query per level, straight out of a
linear layer -- and predicts the weight to give each sample too, already
softmaxed over the L*P samples together. So the cost is O(Q * L * P), which is
what lets a detector attend over a 100x100 feature map, and there is no softmax
in the kernel because the weights arrived normalised.

  That makes it structurally the opposite of every other attention here.
  sparse_attention.py is still a dense score matrix with a mask on it: the
  candidates are chosen by a rule the compiler can see. Here the candidates are
  CONTINUOUS COORDINATES read out of memory, so there is no index set at all,
  only addresses.

THE GATHER HAS A CHANNEL AXIS, which is what separates it from grid_sample.py.
There the tile IS the output plane and each corner is one scalar per lane. Here
the sampled row is `value[idx, :]` -- one gathered ROW INDEX broadcast across D
contiguous channels, giving `idx[:, None] * D + d[None, :]`. That is a gather
of contiguous runs rather than of scalars, the shape a DMA engine can actually
do something with, and it is not the shape grid_sample asks for.

ALL LEVELS SHARE ONE FLAT VALUE BUFFER. The feature pyramid is concatenated
into a single [S, D] tensor and a level is a base offset plus a (H_l, W_l), so
one query touches TWO DIFFERENT RESOLUTIONS in the same accumulator. The levels
therefore cannot be a loop over a tensor of shapes: `LH`, `LW` and `BASE` are
geometry and have to be compile-time. They are passed to a @triton.jit helper as
tl.constexpr and the helper is called once per level with literals -- the same
reason this file cannot index a module-level tuple of shapes from inside the
jit body, which would be reading a global.

Zero padding, because the predicted point routinely lands off the map: an
out-of-range corner contributes nothing, so a sample walking off the edge fades.
One head, to keep the tile two-dimensional -- the head axis is a batch and
gqa_repeat_kv already covers what a head axis does to addressing.

WHY IT DIES ON SPIKE, AND IT IS NOT THIS KERNEL'S BUG. It reached the suite as
`spike: command failed with exit code 255` and nothing else -- the harness threw
away the failing command's own output, and spike writes its whole trap report to
STDOUT, which was never carried. It now reads:

    User load segfault @ 0x00000020d0800f30

READ THE ADDRESS. Strip the high bits and it is 0xd0800f30 -- an
address inside the DRAM the kernel was handed. What is set on top is exactly
2^37, and 37 is 32 + log2(32), where 32 is the byte scale the DMA applies to one
indirect offset (`offset_stride` x element size). That is the whole diagnosis:

    address = base + zext_i64(i32 offset) * scale
                     ^^^^ should be sext

An indirect (gather) transfer takes its per-element offsets from an i32 tile in
the scratchpad, and the DMA adds them to the DRAM base AS UNSIGNED. A NEGATIVE
offset therefore lands 2^32 * scale above where it belongs instead of below the
base -- the right address with one spurious bit.

WHERE THE NEGATIVE OFFSET COMES FROM. Bilinear sampling. A normalised location below the first sample gives a
floor of -1, and the four-corner gather addresses it before the in-bounds
weight zeroes its contribution.

THE SAME FAULT, THE SAME SHAPE, IN FIVE KERNELS: causal_conv1d (bit 34),
conformer_conv (bit 34), rwkv_token_shift (bit 34), ms_deform_attn (bit 37,
offset_stride 8) and unsupported/mask_negative_offset (bit 34), which is the
minimal reproducer of it. convert_accesses_to_transfer already clamps the DMA's BOX so
an out-of-tile position never reads the index buffer past its extent; nothing
clamps the index VALUE. Clamping it into [0, numel) at the point the index tile
is built would be sound -- an in-bounds gather is unaffected, and an
out-of-bounds one was already reading undefined data that the mask discards --
and it is the fix all five want. Not attempted here: it changes the IR of every
indirect transfer in the suite.

Attention with no score matrix: a handful of learned continuous points per query, bilinearly gathered across two resolutions in one flat buffer, weighted by a softmax computed somewhere else.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

Q = 16                          # queries
D = 8                           # channels per head
L = 2                           # feature levels
P = 4                           # sampled points per level
H0, W0 = 8, 8                   # level 0
H1, W1 = 4, 4                   # level 1
S = H0 * W0 + H1 * W1           # the concatenated value buffer
BASE1 = H0 * W0


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _level(v_ptr, loc_ptr, aw_ptr, qi, d,
               L: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
               Q: tl.constexpr, LVL: tl.constexpr,
               LH: tl.constexpr, LW: tl.constexpr, BASE: tl.constexpr):
        """The P samples of ONE level. Geometry is constexpr; see module doc."""
        acc = tl.zeros((Q, D), dtype=tl.float32)
        for i in tl.static_range(P):
            b = (qi * L + LVL) * P + i
            gx = tl.load(loc_ptr + b * 2 + 0)
            gy = tl.load(loc_ptr + b * 2 + 1)
            aw = tl.load(aw_ptr + b)

            # Deformable-DETR normalises to [0, 1] and then goes through
            # grid_sample's align_corners=False, which composes to exactly this.
            ix = gx * LW - 0.5
            iy = gy * LH - 0.5
            fx = tl.floor(ix)
            fy = tl.floor(iy)
            wx = ix - fx
            wy = iy - fy
            x0 = fx.to(tl.int32)
            y0 = fy.to(tl.int32)
            x1 = x0 + 1
            y1 = y0 + 1

            ox0 = (x0 >= 0) & (x0 < LW)
            ox1 = (x1 >= 0) & (x1 < LW)
            oy0 = (y0 >= 0) & (y0 < LH)
            oy1 = (y1 >= 0) & (y1 < LH)
            cx0 = tl.minimum(tl.maximum(x0, 0), LW - 1)
            cx1 = tl.minimum(tl.maximum(x1, 0), LW - 1)
            cy0 = tl.minimum(tl.maximum(y0, 0), LH - 1)
            cy1 = tl.minimum(tl.maximum(y1, 0), LH - 1)

            # One gathered ROW per query, D contiguous channels wide.
            r00 = (BASE + cy0 * LW + cx0)[:, None] * D + d[None, :]
            r01 = (BASE + cy0 * LW + cx1)[:, None] * D + d[None, :]
            r10 = (BASE + cy1 * LW + cx0)[:, None] * D + d[None, :]
            r11 = (BASE + cy1 * LW + cx1)[:, None] * D + d[None, :]
            v00 = tl.load(v_ptr + r00, mask=(oy0 & ox0)[:, None], other=0.0)
            v01 = tl.load(v_ptr + r01, mask=(oy0 & ox1)[:, None], other=0.0)
            v10 = tl.load(v_ptr + r10, mask=(oy1 & ox0)[:, None], other=0.0)
            v11 = tl.load(v_ptr + r11, mask=(oy1 & ox1)[:, None], other=0.0)

            s = (v00 * ((1.0 - wy) * (1.0 - wx))[:, None]
                 + v01 * ((1.0 - wy) * wx)[:, None]
                 + v10 * (wy * (1.0 - wx))[:, None]
                 + v11 * (wy * wx)[:, None])
            acc += s * aw[:, None]
        return acc

    @triton.jit
    def ms_deform_attn_kernel(v_ptr, loc_ptr, aw_ptr, o_ptr,
                              Q: tl.constexpr, D: tl.constexpr,
                              L: tl.constexpr, P: tl.constexpr,
                              H0: tl.constexpr, W0: tl.constexpr,
                              H1: tl.constexpr, W1: tl.constexpr,
                              BASE1: tl.constexpr):
        qi = tl.arange(0, Q)
        d = tl.arange(0, D)
        # Two resolutions, one accumulator. The literals are the point.
        acc = _level(v_ptr, loc_ptr, aw_ptr, qi, d, L, P, D, Q, 0, H0, W0, 0)
        acc += _level(v_ptr, loc_ptr, aw_ptr, qi, d, L, P, D, Q, 1, H1, W1,
                      BASE1)
        tl.store(o_ptr + qi[:, None] * D + d[None, :], acc)

    return ms_deform_attn_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Points overshoot [0, 1] a little, the way reference_point + offset does,
    # so the zero-padding path is exercised.
    loc = torch.rand(Q * L * P * 2, generator=g, dtype=torch.float32)
    loc = loc * 1.14 - 0.07
    w = torch.randn(Q, L * P, generator=g, dtype=torch.float32)
    return {"v": torch.randn(S * D, generator=g, dtype=torch.float32),
            "loc": loc,
            # The op receives weights already softmaxed over the L*P samples.
            "aw": torch.softmax(w, dim=1).reshape(-1).contiguous()}


def reference(inputs):
    import torch
    F = torch.nn.functional
    v = inputs["v"].reshape(S, D)
    loc = inputs["loc"].reshape(Q, L, P, 2)
    aw = inputs["aw"].reshape(Q, L, P)

    shapes = [(H0, W0, 0), (H1, W1, BASE1)]
    out = torch.zeros(Q, D)
    for lid, (lh, lw, base) in enumerate(shapes):
        vl = v[base:base + lh * lw].t().reshape(1, D, lh, lw)
        grid = (2.0 * loc[:, lid] - 1.0).reshape(1, Q, P, 2)
        s = F.grid_sample(vl, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=False)          # (1, D, Q, P)
        out = out + (s[0] * aw[:, lid][None, :, :]).sum(-1).t()
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="ms_deform_attn_kernel", kernel=kernel,
    signature={"v_ptr": "*fp32", "loc_ptr": "*fp32", "aw_ptr": "*fp32",
               "o_ptr": "*fp32", "Q": "constexpr", "D": "constexpr",
               "L": "constexpr", "P": "constexpr", "H0": "constexpr",
               "W0": "constexpr", "H1": "constexpr", "W1": "constexpr",
               "BASE1": "constexpr"},
    constexprs={"Q": Q, "D": D, "L": L, "P": P, "H0": H0, "W0": W0,
                "H1": H1, "W1": W1, "BASE1": BASE1},
    args=[Arg("v", IN, "float32", (S * D,)),
          Arg("loc", IN, "float32", (Q * L * P * 2,)),
          Arg("aw", IN, "float32", (Q * L * P,)),
          Arg("o", OUT, "float32", (Q * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
