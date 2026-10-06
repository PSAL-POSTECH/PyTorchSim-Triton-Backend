"""The ShuffleNetV2 unit -- split, branch, CONCAT, shuffle.

    x  ->  x1 | x2                      channel split, half each
           x1 carried through untouched
           x2 -> 1x1 -> dw3x3 -> 1x1    the branch (BN folded, ReLU kept)
    out =  concat(x1, x2') -> shuffle

FOUR GUIDELINES, AND THEY ARE ABOUT MEMORY RATHER THAN FLOPS. The V2 paper
measures wall clock instead of counting multiplies and gets a different answer
from V1:

  G1  equal channel widths in and out minimise memory traffic per FLOP, so the
      1x1s here do NOT expand and contract the way MobileNetV2's do
  G2  too many groups raises traffic, so the grouped 1x1 of V1 is GONE
  G3  fragmenting a block into many small branches costs parallelism
  G4  element-wise ops are not free, so the residual ADD becomes a CONCAT --
      and the concat is why the split exists

The split is what makes the concat free of a copy: half the channels never
change, so "carrying" them is an address, not a move. The shuffle at the end is
what stops the untouched half from being untouched forever.

WHAT IS NEW HERE FOR THE BACKEND. Not the convolutions -- those are conv1x1 and
depthwise_conv. It is that the output is assembled from TWO tensors of different
provenance: one straight from DRAM, one computed. Concat along the channel axis
with the channels on the lanes means the two halves land in different banks, so
the assembly is a layout question rather than an arithmetic one.

torch equivalent:
    x1, x2 = x.chunk(2, dim=0)
    x2 = relu(x2 * w1)                      the 1x1s, folded to a per-channel scale
    x2 = relu(x2 * w2)
    out = cat([x1, x2]).reshape(2, C // 2, HW).transpose(0, 1).reshape(-1)

ShuffleNetV2: channel split, a branch, and an output assembled from two tensors at strided channel positions -- concat + shuffle in one.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C, HW = 32, 64
H = C // 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def shufflenet_unit_kernel(x_ptr, w1_ptr, w2_ptr, o_ptr,
                               C: tl.constexpr, HW: tl.constexpr):
        H: tl.constexpr = C // 2
        h = tl.arange(0, H)
        p = tl.arange(0, HW)

        # CHANNEL SPLIT: two reads of one tensor, at different channel offsets.
        x1 = tl.load(x_ptr + h[:, None] * HW + p[None, :])
        x2 = tl.load(x_ptr + (h + H)[:, None] * HW + p[None, :])

        # The branch. The two 1x1s are folded to a per-channel scale, since the
        # channel matmul itself is conv1x1's kernel and not what this one tests.
        w1 = tl.load(w1_ptr + h)
        w2 = tl.load(w2_ptr + h)
        y = tl.maximum(x2 * w1[:, None], 0.0)
        y = tl.maximum(y * w2[:, None], 0.0)

        # CONCAT, then shuffle. Written as two stores at the shuffled channel
        # positions: out channel `2i` is x1[i] and `2i+1` is y[i], which IS the
        # G=2 shuffle of `cat([x1, y])`.
        tl.store(o_ptr + (2 * h)[:, None] * HW + p[None, :], x1)
        tl.store(o_ptr + (2 * h + 1)[:, None] * HW + p[None, :], y)

    return shufflenet_unit_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(C * HW, generator=g, dtype=torch.float32),
            "w1": torch.randn(H, generator=g, dtype=torch.float32),
            "w2": torch.randn(H, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(C, HW)
    x1, x2 = x[:H], x[H:]
    y = torch.relu(x2 * inputs["w1"][:, None])
    y = torch.relu(y * inputs["w2"][:, None])
    out = torch.empty_like(x)
    out[0::2] = x1
    out[1::2] = y
    return {"o": out.reshape(-1)}


SPEC = KernelSpec(
    name="shufflenet_unit_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "w1_ptr": "*fp32", "w2_ptr": "*fp32",
               "o_ptr": "*fp32", "C": "constexpr", "HW": "constexpr"},
    constexprs={"C": C, "HW": HW},
    args=[Arg("x", IN, "float32", (C * HW,)),
          Arg("w1", IN, "float32", (H,)), Arg("w2", IN, "float32", (H,)),
          Arg("o", OUT, "float32", (C * HW,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
