"""The 3x3 of a ResNet bottleneck: nine shifted tiles per INPUT CHANNEL.

`conv2d.py` does one input channel and one filter -- the spatial part on its own.
A bottleneck's convolutions are the CHANNEL part, and they are two different
computations wearing the same name.

3x3 OVER INPUT CHANNELS is nine shifted tiles per input channel, accumulated.
The loop nest is (input channel, tap): each tap reads the plane offset by (kh,
kw), scales it by one weight, and adds. So the spatial structure is the same
trick conv2d.py uses and what is new is the accumulation ACROSS channels, which
is a plain running sum in registers and never touches the reduction machinery.

The 1x1 is a different computation entirely -- a matmul over channels -- and it
has its own file, `conv1x1.py`. Two of the three convolutions in every
bottleneck are that one, so the dispatch this backend makes (vector lanes here,
systolic array there) falls along exactly this line.

torch equivalent: F.conv2d(x, k) with one output channel.

The spatial convolution of a bottleneck: nine shifted tiles per input channel, accumulated in registers. The channel sum never touches the reduction machinery.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

CI, W_, OH, OW = 4, 10, 8, 8         # (10 - 3) + 1 == 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def conv_channels_kernel(x_ptr, k_ptr, o_ptr,
                             CI: tl.constexpr, W: tl.constexpr,
                             OH: tl.constexpr, OW: tl.constexpr):
        oh = tl.arange(0, OH)
        ow = tl.arange(0, OW)
        acc = tl.zeros((OH, OW), dtype=tl.float32)
        for ci in tl.static_range(CI):
            for kh in tl.static_range(3):
                for kw in tl.static_range(3):
                    t = tl.load(x_ptr + ci * (W * W)
                                + (oh[:, None] + kh) * W + ow[None, :] + kw)
                    acc += tl.load(k_ptr + (ci * 3 + kh) * 3 + kw) * t
        tl.store(o_ptr + oh[:, None] * OW + ow[None, :], acc)

    return conv_channels_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(CI * W_ * W_, generator=g, dtype=torch.float32),
            "k": torch.randn(CI * 9, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(1, CI, W_, W_)
    k = inputs["k"].reshape(1, CI, 3, 3)
    return {"o": torch.nn.functional.conv2d(x, k).reshape(-1)}


SPEC = KernelSpec(
    name="conv_channels_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "k_ptr": "*fp32", "o_ptr": "*fp32",
               "CI": "constexpr", "W": "constexpr", "OH": "constexpr",
               "OW": "constexpr"},
    constexprs={"CI": CI, "W": W_, "OH": OH, "OW": OW},
    args=[Arg("x", IN, "float32", (CI * W_ * W_,)),
          Arg("k", IN, "float32", (CI * 9,)),
          Arg("o", OUT, "float32", (OH * OW,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
