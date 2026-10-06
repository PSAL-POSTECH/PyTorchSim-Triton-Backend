"""Direct 2-D convolution, 3x3 -- nine OVERLAPPING loads of one buffer.

torch equivalent: F.conv2d(x, w) on a single channel, no padding, stride 1

WHY A CONVOLUTION AND NOT ANOTHER MATMUL.  Every other kernel in this suite
reads each input element of a tile exactly once. A convolution reads each one
nine times, from nine tiles that are SHIFTED COPIES of each other:

    for kh, kw:  acc += w[kh, kw] * x[(oh + kh) * W + (ow + kw)]

Each of those is its own `tts.make_tptr` with the same strides and a different
scalar offset, so nine structured tiles overlap in DRAM and the offsets differ by
`kh * W + kw` -- a compile-time constant that PtrAnalysis has to fold into the
base rather than leave as tensor arithmetic. It is the one addressing shape here
that is affine, aliased, and repeated, and the row shift `+ W` is what makes the
tiles non-contiguous with each other: a DMA that fetched the union once would be
a different (and much better) kernel, and this one is the baseline that does not.

THE FILTER IS A SCALAR PER TAP, not a tile. `w[kh, kw]` is read with a scalar
load off a runtime pointer inside a fully unrolled constexpr loop, so the kernel
also carries nine scalar loads feeding nine splats -- the scalar path, which is
DRAM-direct rather than DMA (see the Phase 4 note in the README), interleaved
with the tile path in the same body.

SHAPES.  The output tile is 128 rows (the lane axis, one program) by 32 columns
(in-lane). Padding is none, so the input is 130 x 34, and W = 34 is deliberately
not a power of two: the row stride then cannot be folded into a shift and every
tile offset is a real multiply.

Complex-op tier: nine overlapping structured tiles off one buffer, with a scalar filter tap splat into each.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

OH, OW = 128, 32     # output, and the tile: OH on the lanes, OW in-lane
KH, KW = 3, 3
H, W = OH + KH - 1, OW + KW - 1     # 130 x 34, no padding


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def conv2d_kernel(x_ptr, w_ptr, o_ptr, W: tl.constexpr, OW: tl.constexpr,
                      OH: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr):
        oh = tl.arange(0, OH)
        ow = tl.arange(0, OW)
        acc = tl.zeros((OH, OW), dtype=tl.float32)
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                # One shifted structured tile per tap.
                x = tl.load(x_ptr + (oh[:, None] + kh) * W + (ow[None, :] + kw))
                acc += tl.load(w_ptr + kh * KW + kw) * x
        tl.store(o_ptr + oh[:, None] * OW + ow[None, :], acc)

    return conv2d_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(H * W, generator=g, dtype=torch.float32),
        "w": torch.randn(KH * KW, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    import torch
    import torch.nn.functional as F
    x = inputs["x"].reshape(1, 1, H, W)
    w = inputs["w"].reshape(1, 1, KH, KW)
    return {"o": F.conv2d(x, w).reshape(-1)}


SPEC = KernelSpec(
    name="conv2d_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "W": "constexpr", "OW": "constexpr", "OH": "constexpr",
               "KH": "constexpr", "KW": "constexpr"},
    constexprs={"W": W, "OW": OW, "OH": OH, "KH": KH, "KW": KW},
    args=[
        Arg("x", IN, "float32", (H * W,)),
        Arg("w", IN, "float32", (KH * KW,)),
        Arg("o", OUT, "float32", (OH * OW,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    # Nine accumulations in a different order than cuDNN's.
    rtol=1e-4, atol=1e-4,
)
