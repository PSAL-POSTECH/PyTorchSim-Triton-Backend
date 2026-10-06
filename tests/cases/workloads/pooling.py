"""maxpool 3x3 s2 and global average pool -- the two poolings in a ResNet.

There are exactly two in the whole network and they sit at opposite ends:
`maxpool` right after the stem, `avgpool` right before the classifier. They are
also opposite in shape, which is why one file holds both.

MAXPOOL IS A SLIDING WINDOW, like a convolution with `max` instead of `+`. Each
tap is a SHIFTED TILE of the input -- the same trick conv2d.py uses -- and the
nine taps fold into a running maximum. Stride 2 makes each tap's read stride 2
along both axes, which is a strided DRAM read and not a strided scratchpad one:
the descriptor takes a stride per axis, so the input tile arrives already
subsampled and every lane still holds contiguous elements.

GLOBAL AVERAGE POOL IS A REDUCTION over all of H*W, one value per channel. On a
[C, HW] tile with the channels on the lanes it is the free direction -- each
lane owns a whole channel and folds it in its own bank. It is also the shape
that made `tl.sum` interesting: the result is one scalar per lane, which stays
in a vector, and NOT one scalar for the whole tile, which could not.

  A ResNet's avgpool is 7x7 -> 1x1 over 2048 channels, so the tile is tall and
  short: the parallelism is entirely in the channels and none of it in the
  reduction. Which is the arrangement to want, and the reason a rank-1 fold
  (one lane working, the rest idle) is the case to avoid.

torch equivalent: F.max_pool2d(x, 3, stride=2), and x.mean(-1)

A ResNet's only two poolings, and they are opposite shapes: a sliding-window max over shifted tiles with a stride-2 DRAM read, and a reduction whose result is one value per lane.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# TRITON'S arange MUST BE A POWER OF TWO, so the shapes here are 8-sided where
# a real ResNet is 7-sided: its avgpool folds 7x7 and its maxpool output is
# 56x56. Nothing about either op depends on the extent -- the window, the stride
# and the fold direction are what is under test -- and the alternative is
# masking every tile to hide a tail, which measures the mask.
H = W = 17                      # input plane; (17 - 3) // 2 + 1 == 8
OH = OW = 8
C, HW = 8, 64                   # avgpool: channels x 8*8 (ResNet: 2048 x 7*7)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def pooling_kernel(x_ptr, m_ptr, f_ptr, a_ptr,
                       W: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
                       C: tl.constexpr, HW: tl.constexpr):
        oh = tl.arange(0, OH)
        ow = tl.arange(0, OW)
        # tap (0, 0) seeds the running max; stride 2 is in the index arithmetic
        acc = tl.load(x_ptr + (oh[:, None] * 2) * W + ow[None, :] * 2)
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                t = tl.load(x_ptr + (oh[:, None] * 2 + kh) * W
                            + (ow[None, :] * 2 + kw))
                acc = tl.maximum(acc, t)
        tl.store(m_ptr + oh[:, None] * OW + ow[None, :], acc)

        c = tl.arange(0, C)
        p = tl.arange(0, HW)
        f = tl.load(f_ptr + c[:, None] * HW + p[None, :])
        tl.store(a_ptr + c, tl.sum(f, axis=1) / HW)

    return pooling_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(H * W, generator=g, dtype=torch.float32),
            "f": torch.randn(C * HW, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(1, 1, H, W)
    m = torch.nn.functional.max_pool2d(x, 3, stride=2)
    return {"m": m.reshape(-1),
            "a": inputs["f"].reshape(C, HW).mean(-1)}


SPEC = KernelSpec(
    name="pooling_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "m_ptr": "*fp32", "f_ptr": "*fp32",
               "a_ptr": "*fp32", "W": "constexpr", "OH": "constexpr",
               "OW": "constexpr", "C": "constexpr", "HW": "constexpr"},
    constexprs={"W": W, "OH": OH, "OW": OW, "C": C, "HW": HW},
    args=[Arg("x", IN, "float32", (H * W,)), Arg("m", OUT, "float32", (OH * OW,)),
          Arg("f", IN, "float32", (C * HW,)), Arg("a", OUT, "float32", (C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
