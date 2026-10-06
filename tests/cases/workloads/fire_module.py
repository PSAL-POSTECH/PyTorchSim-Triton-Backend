"""SqueezeNet's fire module -- squeeze, then two expands, then concat.

    squeeze   1x1   C -> S          S is small, typically C/8
    expand    1x1   S -> E1   and   3x3   S -> E3
    out       cat(e1, e3)           E1 + E3 channels, usually 4S each

THREE STRATEGIES FOR FEWER PARAMETERS, and the module is all three at once.
Iandola et al. (2016) set out to match AlexNet with 50x fewer parameters, and
their reasoning is arithmetic rather than architectural:

  S1  replace 3x3 with 1x1 where you can        9x fewer weights per filter
  S2  shrink the INPUT channel count to the     a 3x3 layer costs
      3x3s that remain                          C_in * C_out * 9
  S3  downsample late, so the layers that do    bigger activations, better
      have parameters see large maps            accuracy for the same weights

The squeeze is S2: it narrows C to S before anything expensive runs. The two
expands are S1: half the output comes from 1x1 filters that cost a ninth.

WHY BOTH SIZES. A pure 1x1 stack has no spatial receptive field at all -- it can
only mix channels. The 3x3 half is what sees neighbours; the 1x1 half is what
makes it affordable. Concatenating rather than adding keeps the two kinds of
feature separable for the next layer, the same argument DenseNet makes.

WHAT IS NEW HERE FOR THE BACKEND. The output is assembled from two branches with
DIFFERENT SHAPES of computation -- one is pointwise, one has spatial taps -- so
the two halves of the output tile are produced by different loop nests and then
have to land in one buffer. dense_block concatenates pieces of different
provenance; this concatenates pieces of different STRUCTURE.

torch equivalent: s = relu(conv1x1(x)); cat([relu(conv1x1(s)),
                                             relu(conv3x3(s))], dim=0)

SqueezeNet: two expand branches of different STRUCTURE -- one pointwise, one with spatial taps -- concatenated into one tile.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, HW = 16, 66                 # squeezed channels, flattened plane
E = 16                         # each expand branch's channels
OW = HW - 2                    # VALID 3x3, and 64 so tl.arange stays a power of two
OC = E * 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def fire_module_kernel(s_ptr, w1_ptr, w3_ptr, o_ptr,
                           S: tl.constexpr, HW: tl.constexpr,
                           E: tl.constexpr, OW: tl.constexpr,
                           OC: tl.constexpr):
        e = tl.arange(0, E)
        p = tl.arange(0, OW)
        # The squeeze itself is a channel matmul -- conv1x1's kernel -- so this
        # one starts from its output and tests the two expands and the join.
        # The 1x1 branch reads the same window the VALID 3x3 branch produces,
        # so the two halves are the same shape and the concat is a concat.
        s = tl.load(s_ptr + e[:, None] * HW + (p + 1)[None, :])

        # EXPAND 1x1: pointwise, one weight per channel.
        b1 = tl.maximum(s * tl.load(w1_ptr + e)[:, None], 0.0)

        # EXPAND 3x3: spatial taps along the flattened plane, per channel. A
        # different loop nest over the same input, which is the point. VALID
        # rather than masked -- a shifted tile keeps every address in range,
        # which is how every other tap kernel here is written (depthwise_conv).
        b3 = tl.zeros((E, OW), dtype=tl.float32)
        for t in tl.static_range(3):
            v = tl.load(s_ptr + e[:, None] * HW + (p + t)[None, :])
            b3 += v * tl.load(w3_ptr + t * E + e)[:, None]
        b3 = tl.maximum(b3, 0.0)

        # CONCAT: the two branches into one output, at their own channel ranges.
        tl.store(o_ptr + e[:, None] * OW + p[None, :], b1)
        tl.store(o_ptr + (e + E)[:, None] * OW + p[None, :], b3)

    return fire_module_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"s": torch.randn(S * HW, generator=g, dtype=torch.float32),
            "w1": torch.randn(E, generator=g, dtype=torch.float32),
            "w3": torch.randn(3 * E, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    s = inputs["s"].reshape(S, HW)
    w3 = inputs["w3"].reshape(3, E)
    b1 = torch.relu(s[:, 1:HW - 1] * inputs["w1"][:, None])
    b3 = torch.zeros(E, OW)
    for t in range(3):
        b3 += s[:, t:t + OW] * w3[t][:, None]
    b3 = torch.relu(b3)
    return {"o": torch.cat([b1, b3], dim=0).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="fire_module_kernel",
    kernel=kernel,
    signature={"s_ptr": "*fp32", "w1_ptr": "*fp32", "w3_ptr": "*fp32",
               "o_ptr": "*fp32", "S": "constexpr", "HW": "constexpr",
               "E": "constexpr", "OW": "constexpr", "OC": "constexpr"},
    constexprs={"S": S, "HW": HW, "E": E, "OW": OW, "OC": OC},
    args=[Arg("s", IN, "float32", (S * HW,)),
          Arg("w1", IN, "float32", (E,)),
          Arg("w3", IN, "float32", (3 * E,)),
          Arg("o", OUT, "float32", (OC * OW,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
