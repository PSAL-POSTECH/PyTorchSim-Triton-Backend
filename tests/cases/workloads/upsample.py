"""Nearest and bilinear upsampling, and the transposed convolution behind them.

U-Net's decoder has to undo the encoder's pooling, and there are two ways:

    upsample    out[y, x] = f(in[y/2, x/2])       fixed, no parameters
    transposed  out = sum over taps, stride 2     learned, and the source of
                conv                              checkerboard artefacts

NEAREST is a GATHER WITH A DIVIDED INDEX. `out[y, x] = in[y // 2, x // 2]`, so
every input pixel is read four times and the address is not affine in the output
coordinate -- `y // 2` is where the structured-pointer analysis usually gives up
and the access becomes an unstructured gather. That is the interesting part for
this backend; the arithmetic is a copy.

BILINEAR is four gathers and three lerps.

    fy = (y + 0.5) / 2 - 0.5, clamped        the source coordinate, half-pixel
    y0 = floor(fy), wy = fy - y0             aligned so the corners behave

    out = (1-wy)(1-wx) in[y0, x0] + (1-wy) wx in[y0, x1]
        +    wy (1-wx) in[y1, x0] +    wy  wx in[y1, x1]

The half-pixel offset is `align_corners=False`, which is torch's default and the
only convention under which upsampling twice equals upsampling once by four.

TRANSPOSED CONVOLUTION is a convolution whose stride lands on the OUTPUT.

    out[2y + kh, 2x + kw] += w[kh, kw] * in[y, x]

Written as a scatter it needs an accumulate; written as a gather -- which is what
this kernel does, and what every implementation does -- it is a normal
convolution over an input with holes:

    out[o] = sum over taps t where (o - t) is even  of  w[t] * in[(o - t) / 2]

CHECKERBOARD. With kernel 4 and stride 2 each output gets two taps; with kernel 3
and stride 2 the outputs alternate between two taps and one, so a uniform input
produces a striped output before any training. It is why U-Net implementations
have largely moved to `upsample + conv3x3`, which cannot have the problem.

torch equivalent: F.interpolate(x, scale_factor=2, mode='nearest'), then
                  F.interpolate(x, scale_factor=2, mode='bilinear',
                                align_corners=False)

U-Net decoder: nearest (a divided index) and bilinear (four gathers and three lerps, half-pixel aligned).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

H, W = 16, 16
OH, OW = H * 2, W * 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def upsample_kernel(x_ptr, n_ptr, b_ptr,
                        H: tl.constexpr, W: tl.constexpr,
                        OH: tl.constexpr, OW: tl.constexpr):
        oy = tl.arange(0, OH)
        ox = tl.arange(0, OW)

        # NEAREST: the source index is the output index halved.
        sy = oy // 2
        sx = ox // 2
        near = tl.load(x_ptr + sy[:, None] * W + sx[None, :])
        tl.store(n_ptr + oy[:, None] * OW + ox[None, :], near)

        # BILINEAR, half-pixel aligned: fy = (oy + 0.5)/2 - 0.5 = oy/2 - 0.25.
        fy = oy.to(tl.float32) * 0.5 - 0.25
        fx = ox.to(tl.float32) * 0.5 - 0.25
        fy = tl.maximum(fy, 0.0)
        fx = tl.maximum(fx, 0.0)
        y0 = fy.to(tl.int32)
        x0 = fx.to(tl.int32)
        y1 = tl.minimum(y0 + 1, H - 1)
        x1 = tl.minimum(x0 + 1, W - 1)
        wy = fy - y0.to(tl.float32)
        wx = fx - x0.to(tl.float32)

        v00 = tl.load(x_ptr + y0[:, None] * W + x0[None, :])
        v01 = tl.load(x_ptr + y0[:, None] * W + x1[None, :])
        v10 = tl.load(x_ptr + y1[:, None] * W + x0[None, :])
        v11 = tl.load(x_ptr + y1[:, None] * W + x1[None, :])
        top = v00 + (v01 - v00) * wx[None, :]
        bot = v10 + (v11 - v10) * wx[None, :]
        out = top + (bot - top) * wy[:, None]
        tl.store(b_ptr + oy[:, None] * OW + ox[None, :], out)

    return upsample_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(H * W, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(1, 1, H, W)
    n = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
    b = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear",
                                        align_corners=False)
    return {"near": n.reshape(-1), "bilin": b.reshape(-1)}


SPEC = KernelSpec(
    name="upsample_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "n_ptr": "*fp32", "b_ptr": "*fp32",
               "H": "constexpr", "W": "constexpr", "OH": "constexpr",
               "OW": "constexpr"},
    constexprs={"H": H, "W": W, "OH": OH, "OW": OW},
    args=[Arg("x", IN, "float32", (H * W,)),
          Arg("near", OUT, "float32", (OH * OW,)),
          Arg("bilin", OUT, "float32", (OH * OW,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    # AN `outs` NEVER READ IS NO OPERAND, as upstream's op has none; while it was
    # one, it took a layout of its own and passed it on -- four crossings, none needed.
    # The one move left is the [16,2,16,2] -> [32,32] reshape's permute.
    expect_ir={"08-transformed.mlir": {"pytorchsim_tpu.xlu_move": 1}},
)
