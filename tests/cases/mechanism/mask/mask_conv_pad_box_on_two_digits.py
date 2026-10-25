"""A padded depthwise convolution as Inductor writes it: each tap masked on two digits of the row.

ConvNeXt's 7x7 depthwise conv, fused with its layer norm, reads x[n, c, h + dh, w + dw] at
`(x % 64) + dh*8 + dw + 64*c + 64*C*(x // 64)` with x = n*64 + h*8 + w, and masks each tap with
`0 <= (x // 8) % 8 + dh < 8` and `0 <= x % 8 + dw < 8`, then a bias load at `c` broadcast over x
under the same mask. MaskAnalysis read a remainder only alone, `x % K < c`, so every tap fell to a
gather: 98 for the 49 taps of the model's kernel, past the 64 records a TOG declares. The mask is a
box on the digits of x cut at 8 and 64: the access is one structured load of [N, 8, 8, C] with
bounds on the h and w digits.

torch equivalent: o[n, h, w, c] = sum over taps of (pad(x)[n, c, h+dh, w+dw] + b[c]*inside) * k[c, tap].
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, H, W, C = 4, 8, 8, 12
XBLOCK, RBLOCK = 128, 16
TAPS, PAD = 3, 1


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_conv_pad_box_on_two_digits(x_ptr, b_ptr, k_ptr, o_ptr, XBLOCK: tl.constexpr,
                                        RBLOCK: tl.constexpr, C: tl.constexpr,
                                        TAPS: tl.constexpr, PAD: tl.constexpr):
        x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        c = tl.arange(0, RBLOCK)[None, :]
        cm = c < C
        h = (x // 8) % 8
        w = x % 8
        acc = tl.zeros([XBLOCK, RBLOCK], tl.float32)
        for t in tl.static_range(TAPS * TAPS):
            dh = t // TAPS - PAD
            dw = t % TAPS - PAD
            m = (h + dh >= 0) & (h + dh < 8) & (w + dw >= 0) & (w + dw < 8)
            v = tl.load(x_ptr + (dh * 8 + dw + x % 64 + 64 * c + 64 * C * (x // 64)), cm & m,
                        other=0.0)
            bias = tl.load(b_ptr + tl.broadcast_to(c, [XBLOCK, RBLOCK]), cm & m, other=0.0)
            k = tl.load(k_ptr + (t + TAPS * TAPS * c), cm, other=0.0)
            acc += (v + bias) * k
        tl.store(o_ptr + (c + C * x), acc, cm)

    return mask_conv_pad_box_on_two_digits


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
    name="mask_conv_pad_box_on_two_digits", kernel=kernel,
    signature={"x_ptr": "*fp32", "b_ptr": "*fp32", "k_ptr": "*fp32", "o_ptr": "*fp32",
               "XBLOCK": "constexpr", "RBLOCK": "constexpr", "C": "constexpr",
               "TAPS": "constexpr", "PAD": "constexpr"},
    constexprs={"XBLOCK": XBLOCK, "RBLOCK": RBLOCK, "C": C, "TAPS": TAPS, "PAD": PAD},
    args=[Arg("x", IN, "float32", (N * C * H * W,)),
          Arg("b", IN, "float32", (C,)),
          Arg("k", IN, "float32", (C * TAPS * TAPS,)),
          Arg("o", OUT, "float32", (N * H * W * C,))],
    grid=(N * H * W // XBLOCK,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0}},
)
