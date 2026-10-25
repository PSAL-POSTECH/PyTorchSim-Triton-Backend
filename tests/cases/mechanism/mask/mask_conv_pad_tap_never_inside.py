"""A padded conv's tap that never lands inside the image: its digit bound is empty, not unread.

ConvNeXtV2's last stage is 2x2 under a 7x7 kernel: for |dh| >= 2 the mask `0 <= (x // 2) % 2 + dh < 2`
holds for no row. MaskAnalysis read no box from a bound the digit never meets, so 40 of 49 taps
(80 loads) fell to gathers, past the 64 records a TOG declares. Such a bound is an empty axis: the
load reads nothing and keeps the fill. Here 2x2 images under a 5x5 kernel, 16 of 25 taps empty.

torch equivalent: o[n, h, w, c] = sum over taps of (pad(x)[n, c, h+dh, w+dw] + b[c]*inside) * k[c, tap].
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, H, W, C = 4, 2, 2, 12
XBLOCK, RBLOCK = 8, 16
TAPS, PAD = 5, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_conv_pad_tap_never_inside(x_ptr, b_ptr, k_ptr, o_ptr, XBLOCK: tl.constexpr,
                                        RBLOCK: tl.constexpr, C: tl.constexpr,
                                        TAPS: tl.constexpr, PAD: tl.constexpr, H: tl.constexpr):
        x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        c = tl.arange(0, RBLOCK)[None, :]
        cm = c < C
        h = (x // H) % H
        w = x % H
        acc = tl.zeros([XBLOCK, RBLOCK], tl.float32)
        for t in tl.static_range(TAPS * TAPS):
            dh = t // TAPS - PAD
            dw = t % TAPS - PAD
            m = (h + dh >= 0) & (h + dh < H) & (w + dw >= 0) & (w + dw < H)
            v = tl.load(x_ptr + (dh * H + dw + x % (H * H) + H * H * c + H * H * C * (x // (H * H))), cm & m,
                        other=0.0)
            bias = tl.load(b_ptr + tl.broadcast_to(c, [XBLOCK, RBLOCK]), cm & m, other=0.0)
            k = tl.load(k_ptr + (t + TAPS * TAPS * c), cm, other=0.0)
            acc += (v + bias) * k
        tl.store(o_ptr + (c + C * x), acc, cm)

    return mask_conv_pad_tap_never_inside


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N * C * H * W, generator=g, dtype=torch.float32),
            "b": torch.randn(C, generator=g, dtype=torch.float32),
            "k": torch.randn(C * TAPS * TAPS, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = torch.nn.functional.pad(inputs["x"].reshape(N, C, H, W), (PAD, PAD, PAD, PAD))
    inside = torch.nn.functional.pad(torch.ones(N, 1, H, W), (PAD, PAD, PAD, PAD))
    b = inputs["b"].reshape(1, C, 1, 1)
    k = inputs["k"].reshape(C, TAPS * TAPS)
    o = torch.zeros(N, C, H, W)
    for t in range(TAPS * TAPS):
        dh, dw = t // TAPS, t % TAPS
        win = (x + b * inside)[:, :, dh:dh + H, dw:dw + W]
        o = o + win * k[:, t].reshape(1, C, 1, 1)
    return {"o": o.permute(0, 2, 3, 1).reshape(-1).clone()}


SPEC = KernelSpec(
    name="mask_conv_pad_tap_never_inside", kernel=kernel,
    signature={"x_ptr": "*fp32", "b_ptr": "*fp32", "k_ptr": "*fp32", "o_ptr": "*fp32",
               "XBLOCK": "constexpr", "RBLOCK": "constexpr", "C": "constexpr",
               "TAPS": "constexpr", "PAD": "constexpr", "H": "constexpr"},
    constexprs={"XBLOCK": XBLOCK, "RBLOCK": RBLOCK, "C": C, "TAPS": TAPS, "PAD": PAD, "H": H},
    args=[Arg("x", IN, "float32", (N * C * H * W,)),
          Arg("b", IN, "float32", (C,)),
          Arg("k", IN, "float32", (C * TAPS * TAPS,)),
          Arg("o", OUT, "float32", (N * H * W * C,))],
    grid=(N * H * W // XBLOCK,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0}},
)
