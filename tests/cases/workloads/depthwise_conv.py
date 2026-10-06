"""3x3 depthwise convolution and ReLU6 -- what MobileNetV2 is built out of.

    out[c, y, x] = sum_kh sum_kw  w[c, kh, kw] * in[c, y+kh, x+kw]

THERE IS NO CHANNEL SUM, and that is the whole op. A standard conv folds the
input channels; a depthwise one gives each channel its own 3x3 filter and never
mixes them. The weight is `[C, 3, 3]` -- one filter per channel -- and it is
`groups == C`, the extreme of a grouped convolution.

  It exists to be cheap. Splitting a standard 3x3 into depthwise + 1x1
  pointwise costs `ci*9 + ci*co` against `ci*co*9`, a ratio of `1/co + 1/9`, so
  about an eighth at 256 channels. That is how MobileNetV2 does ResNet-50's job
  with 3.5M parameters instead of 25.5M.

AND IT HAPPENS TO SUIT THIS MACHINE. With the channels on the lanes, lane `c`
holds channel `c`'s whole plane in its own bank, the tap `w[c, kh, kw]` is one
scalar per channel -- a channel broadcast, the free direction -- and the spatial
shift is an offset INSIDE the lane. No channel sum means no lane-to-lane
communication, where a standard conv has to fold across the banks. The op was
designed to save parameters and the lane structure came out for free.

The tile is RANK 3 (`[C, OH, OW]`), which is also worth having: the lane axis is
dim 0, the vector axis is dim 2, and dim 1 is a plain loop in between -- the one
arrangement a rank-3 tile is allowed here, since a middle lane axis has no
layout.

RELU6 is `min(max(x, 0), 6)`, and the 6 is about QUANTISATION rather than about
the function: a fixed `[0, 6]` output range lets an int8 scale be chosen ahead of
time, and one outlier cannot ruin it. MobileNetV2 uses it everywhere ReLU would
go -- except after the projecting 1x1, which has no activation at all (the
"linear bottleneck": ReLU on a NARROW layer destroys information that a wide one
would have kept a copy of).

torch equivalent: F.conv2d(x, w, groups=C), then clamp(0, 6)

MobileNetV2's only genuinely new op. No channel sum means no lane-to-lane communication; the per-channel tap is the free broadcast direction, and the tile is rank 3 with the middle axis a plain loop.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C, H, W = 8, 10, 10
OH = OW = 8                      # (10 - 3) + 1


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def depthwise_conv_kernel(x_ptr, w_ptr, o_ptr, r_ptr,
                              C: tl.constexpr, H: tl.constexpr,
                              W: tl.constexpr, OH: tl.constexpr,
                              OW: tl.constexpr):
        c = tl.arange(0, C)[:, None, None]
        oh = tl.arange(0, OH)[None, :, None]
        ow = tl.arange(0, OW)[None, None, :]

        acc = tl.zeros((C, OH, OW), dtype=tl.float32)
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                t = tl.load(x_ptr + c * (H * W) + (oh + kh) * W + ow + kw)
                # one scalar per channel: a channel broadcast, not a channel sum
                wt = tl.load(w_ptr + c * 9 + kh * 3 + kw)
                acc += wt * t
        idx = c * (OH * OW) + oh * OW + ow
        tl.store(o_ptr + idx, acc)
        tl.store(r_ptr + idx, tl.minimum(tl.maximum(acc, 0.0), 6.0))

    return depthwise_conv_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(C * H * W, generator=g, dtype=torch.float32),
            "w": torch.randn(C * 9, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(1, C, H, W)
    w = inputs["w"].reshape(C, 1, 3, 3)
    o = torch.nn.functional.conv2d(x, w, groups=C)
    return {"o": o.reshape(-1), "r": o.clamp(0.0, 6.0).reshape(-1)}


SPEC = KernelSpec(
    name="depthwise_conv_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "r_ptr": "*fp32", "C": "constexpr", "H": "constexpr",
               "W": "constexpr", "OH": "constexpr", "OW": "constexpr"},
    constexprs={"C": C, "H": H, "W": W, "OH": OH, "OW": OW},
    args=[Arg("x", IN, "float32", (C * H * W,)),
          Arg("w", IN, "float32", (C * 9,)),
          Arg("o", OUT, "float32", (C * OH * OW,)),
          Arg("r", OUT, "float32", (C * OH * OW,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
