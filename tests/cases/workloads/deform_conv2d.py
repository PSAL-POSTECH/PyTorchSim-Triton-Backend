"""DCNv2 -- a 3x3 convolution whose nine taps have moved, and been re-weighted.

    p     = (oy - 1 + kh, ox - 1 + kw)            where a normal conv reads
    p'    = p + offset[kh, kw, oy, ox]            where THIS one reads
    out   = sum_k w[k] * mask[k, oy, ox] * bilinear(x, p'_k)

torch equivalent: torchvision.ops.deform_conv2d(x, offset, w, mask=mask)

TWO THINGS ARE LEARNED THAT ARE NORMALLY GEOMETRY. A convolution's tap positions
are a compile-time constant -- conv2d.py reads nine SHIFTED TILES, and the shift
is a literal added to the lane index. Dai et al. make the displacement a tensor
predicted by a side branch, so the offsets are float, different at every output
pixel, and different for every one of the nine taps. v2 adds the modulation
mask: a per-tap per-pixel scalar in [0, 1] that lets the layer turn a tap OFF,
which is what makes it work on occlusion.

WHAT THAT COSTS THE COMPILER. conv2d's nine reads are nine affine tiles and the
whole thing stays one strided descriptor family. Here each tap is FOUR GATHERS
at an address that came out of memory, so a 3x3 convolution becomes thirty-six
unstructured loads -- and they are not shared between taps, because tap (0,0)
and tap (0,1) have independent offsets and land nowhere near each other.

  This is grid_sample.py's gather, but NINE OF THEM PER OUTPUT PIXEL, each with
  its own coordinate field, and the results folded by a conv's weights instead
  of being the answer. grid_sample asks whether the backend can do the gather at
  all; this asks whether it can do it nine times in a static loop and keep the
  accumulator in a register.

THE SAMPLE IS ZERO-PADDED PER CORNER, matching torchvision exactly: a tap whose
offset walks it off the plane fades out rather than clamping to the border. The
convolution's own `pad=1` is folded into the coordinate (`oy - 1 + kh`) instead
of materialising a padded input, so the padding and the offsets share one
out-of-bounds test -- which is why the padding is not a separate mask here.

  The bilinear body is a @triton.jit HELPER taking H and W as tl.constexpr.
  Inlining it nine times by hand would be nine copies of the same masking; as a
  helper called with literal geometry it stays one readable block and triton
  specialises it per call site.

Single input and output channel: the channel axis of a deformable conv is an
ordinary conv reduction and conv2d.py already covers it. What is new is the
address, and the address does not depend on the channel.

A convolution whose tap positions are a tensor: nine independent coordinate fields, thirty-six gathers, one accumulator. conv2d's shifted tiles with the shift taken out of memory.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

H, W = 8, 8                     # input plane
OH, OW = 8, 8                   # stride 1, pad 1, 3x3 -> same size
K = 3                           # kernel edge
PAD = 1


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _bilin(x_ptr, py, px, H: tl.constexpr, W: tl.constexpr):
        """Zero-padded bilinear sample at a float coordinate tile."""
        fy = tl.floor(py)
        fx = tl.floor(px)
        wy = py - fy
        wx = px - fx
        y0 = fy.to(tl.int32)
        x0 = fx.to(tl.int32)
        y1 = y0 + 1
        x1 = x0 + 1

        oy0 = (y0 >= 0) & (y0 < H)
        oy1 = (y1 >= 0) & (y1 < H)
        ox0 = (x0 >= 0) & (x0 < W)
        ox1 = (x1 >= 0) & (x1 < W)
        cy0 = tl.minimum(tl.maximum(y0, 0), H - 1)
        cy1 = tl.minimum(tl.maximum(y1, 0), H - 1)
        cx0 = tl.minimum(tl.maximum(x0, 0), W - 1)
        cx1 = tl.minimum(tl.maximum(x1, 0), W - 1)

        v00 = tl.load(x_ptr + cy0 * W + cx0, mask=oy0 & ox0, other=0.0)
        v01 = tl.load(x_ptr + cy0 * W + cx1, mask=oy0 & ox1, other=0.0)
        v10 = tl.load(x_ptr + cy1 * W + cx0, mask=oy1 & ox0, other=0.0)
        v11 = tl.load(x_ptr + cy1 * W + cx1, mask=oy1 & ox1, other=0.0)
        return (v00 * ((1.0 - wy) * (1.0 - wx)) + v01 * ((1.0 - wy) * wx)
                + v10 * (wy * (1.0 - wx)) + v11 * (wy * wx))

    @triton.jit
    def deform_conv2d_kernel(x_ptr, off_ptr, msk_ptr, w_ptr, o_ptr,
                             H: tl.constexpr, W: tl.constexpr,
                             OH: tl.constexpr, OW: tl.constexpr,
                             K: tl.constexpr, PAD: tl.constexpr):
        oy = tl.arange(0, OH)
        ox = tl.arange(0, OW)
        p = oy[:, None] * OW + ox[None, :]
        acc = tl.zeros((OH, OW), dtype=tl.float32)

        for kh in tl.static_range(K):
            for kw in tl.static_range(K):
                t = kh * K + kw
                # torchvision's layout: offset is [2 * K * K, OH, OW] with dy
                # before dx for each tap, mask is [K * K, OH, OW].
                dy = tl.load(off_ptr + (2 * t + 0) * (OH * OW) + p)
                dx = tl.load(off_ptr + (2 * t + 1) * (OH * OW) + p)
                m = tl.load(msk_ptr + t * (OH * OW) + p)

                # `pad` lives in the coordinate, not in a padded input, so it
                # shares the sampler's out-of-bounds test.
                py = (oy[:, None] - PAD + kh).to(tl.float32) + dy
                px = (ox[None, :] - PAD + kw).to(tl.float32) + dx

                acc += tl.load(w_ptr + t) * m * _bilin(x_ptr, py, px, H, W)

        tl.store(o_ptr + p, acc)

    return deform_conv2d_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = K * K * OH * OW
    return {
        # Offsets around one pixel: big enough to leave the tap's own cell and
        # to walk off the plane at the border, small enough to stay a conv.
        "off": torch.randn(2 * n, generator=g, dtype=torch.float32) * 0.9,
        # DCNv2's modulation is sigmoid-gated, so it lives in [0, 1].
        "msk": torch.rand(n, generator=g, dtype=torch.float32),
        "x": torch.randn(H * W, generator=g, dtype=torch.float32),
        "w": torch.randn(K * K, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    import torch
    x = inputs["x"].reshape(H, W)
    off = inputs["off"].reshape(2 * K * K, OH, OW)
    msk = inputs["msk"].reshape(K * K, OH, OW)
    w = inputs["w"].reshape(K * K)

    oy = torch.arange(OH).reshape(OH, 1).float()
    ox = torch.arange(OW).reshape(1, OW).float()
    acc = torch.zeros(OH, OW)
    for t in range(K * K):
        kh, kw = t // K, t % K
        py = oy - PAD + kh + off[2 * t + 0]
        px = ox - PAD + kw + off[2 * t + 1]
        y0 = torch.floor(py)
        x0 = torch.floor(px)
        wy, wx = py - y0, px - x0
        y0, x0 = y0.long(), x0.long()
        y1, x1 = y0 + 1, x0 + 1
        xf = x.reshape(-1)

        def tap(yy, xx):
            ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            idx = yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1)
            return torch.where(ok, xf[idx], torch.zeros(()))

        v = (tap(y0, x0) * (1 - wy) * (1 - wx) + tap(y0, x1) * (1 - wy) * wx
             + tap(y1, x0) * wy * (1 - wx) + tap(y1, x1) * wy * wx)
        acc = acc + w[t] * msk[t] * v
    return {"o": acc.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="deform_conv2d_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "off_ptr": "*fp32", "msk_ptr": "*fp32",
               "w_ptr": "*fp32", "o_ptr": "*fp32",
               "H": "constexpr", "W": "constexpr", "OH": "constexpr",
               "OW": "constexpr", "K": "constexpr", "PAD": "constexpr"},
    constexprs={"H": H, "W": W, "OH": OH, "OW": OW, "K": K, "PAD": PAD},
    args=[Arg("x", IN, "float32", (H * W,)),
          Arg("off", IN, "float32", (2 * K * K * OH * OW,)),
          Arg("msk", IN, "float32", (K * K * OH * OW,)),
          Arg("w", IN, "float32", (K * K,)),
          Arg("o", OUT, "float32", (OH * OW,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
