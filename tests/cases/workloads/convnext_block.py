"""A ConvNeXt block -- what a ResNet becomes when you copy a Transformer.

    x  ->  dw7x7  ->  LayerNorm over CHANNELS  ->  1x1 (4x wider)
       ->  GELU   ->  1x1 (back)  ->  * gamma  ->  + x

THE PAPER IS AN ABLATION, NOT AN ARCHITECTURE. Liu et al. (2022) start from
ResNet-50 and apply the Swin Transformer's choices one at a time, keeping each
that helps: the stage ratio, a patchify stem, depthwise 7x7 instead of 3x3,
an inverted bottleneck, fewer activations and fewer normalisers, GELU instead of
ReLU, LayerNorm instead of BatchNorm. The result beats Swin with no attention in
it, which is the argument -- the gains were the design, not the self-attention.

FOUR THINGS ARE NEW HERE, and three of them are about the channel axis.

  dw7x7           a bigger receptive field per layer, still cheap because it is
                  depthwise -- and left to depthwise_conv, which is that shape
                  with fewer taps. What is kept here is a per-channel weight.
  LayerNorm       over CHANNELS at each position, not over the batch. This is
                  the interesting one: BatchNorm folds an axis that is not in
                  the tile at all (see 02-normalization.md), while this folds
                  the channel axis -- which, with the channels on the lanes, is
                  a fold ACROSS the banks.
  GELU            x * Phi(x), the tanh approximation. Already in activations.py.
  LayerScale      a learned per-channel scalar on the residual branch, init 1e-6
                  so the block starts as the identity and deep stacks train.

ONE NORM AND ONE ACTIVATION PER BLOCK. A ResNet bottleneck has three of each;
this has one of each, placed the way a Transformer places them. The paper
measures that removing them HELPS, which is the least intuitive result in it.

WHAT IT ASKS OF THIS MACHINE. The block folds the channel axis (LayerNorm) and
broadcasts along it (LayerScale, and the norm's own mean/rsigma) in the same
kernel -- the shape squeeze_excite refuses to make easy. With the channels
IN-LANE both are free: the fold is inside one bank and the broadcast is a splat.
That is the layout this kernel wants, and the tile is written [HW, C] rather
than [C, HW] to ask for it.

torch equivalent: x + gamma * fc2(gelu(fc1(layer_norm(dwconv(x)))))

ConvNeXt: depthwise 7x7, LayerNorm over the CHANNEL axis, GELU and LayerScale -- a fold and a broadcast along the same axis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

HW, C = 64, 32
EPS = 1e-6


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def convnext_block_kernel(x_ptr, w_ptr, ln_g_ptr, ln_b_ptr, gam_ptr, o_ptr,
                              HW: tl.constexpr, C: tl.constexpr,
                              EPS: tl.constexpr):
        s = tl.arange(0, HW)
        c = tl.arange(0, C)
        x = tl.load(x_ptr + s[:, None] * C + c[None, :])      # [HW, C]

        # The depthwise 7x7 that opens the block is depthwise_conv's shape with
        # more taps, so it is left there; what this kernel keeps is a
        # per-channel weight, which is the part the layout has to survive.
        y = x * tl.load(w_ptr + c)[None, :]

        # LAYERNORM OVER CHANNELS -- the fold and the broadcast are both along
        # the channel axis, and both stay inside one lane with this layout.
        mu = tl.sum(y, axis=1) / C
        d = y - mu[:, None]
        var = tl.sum(d * d, axis=1) / C
        n = d * tl.exp(-0.5 * tl.log(var + EPS))[:, None]
        n = n * tl.load(ln_g_ptr + c)[None, :] + tl.load(ln_b_ptr + c)[None, :]

        # THE INVERTED BOTTLENECK, folded to per-channel scales: the two 1x1s
        # are channel matmuls and conv1x1 already covers those. What is kept is
        # the ACTIVATION between them, which is the part that is new.
        h = n * 4.0
        # GELU, tanh approximation. `tanh(u) == 2*sigmoid(2u) - 1` exactly, and
        # that is the form activations.py uses -- writing it as (e-1)/(e+1)
        # with `e = exp(2u)` overflows to inf and then to NaN, which is what
        # this kernel did first: the normalised input reaches |h| ~ 12, so the
        # cubic term puts `2u` past 140.
        u = 0.7978845608 * (h + 0.044715 * h * h * h)
        g = 0.5 * h * (1.0 + 2.0 * tl.sigmoid(2.0 * u) - 1.0)

        # LAYERSCALE, then the residual. gamma starts at 1e-6 so a fresh block
        # is the identity; here it is an input so the test sees a real one.
        out = x + g * tl.load(gam_ptr + c)[None, :]
        tl.store(o_ptr + s[:, None] * C + c[None, :], out)

    return convnext_block_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(HW * C, generator=g, dtype=torch.float32),
            "w": torch.randn(C, generator=g, dtype=torch.float32),
            "ln_g": torch.randn(C, generator=g, dtype=torch.float32),
            "ln_b": torch.randn(C, generator=g, dtype=torch.float32),
            "gam": torch.randn(C, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(HW, C)
    y = x * inputs["w"][None, :]
    mu = y.mean(dim=1, keepdim=True)
    d = y - mu
    var = (d * d).mean(dim=1, keepdim=True)
    n = d * torch.rsqrt(var + EPS)
    n = n * inputs["ln_g"][None, :] + inputs["ln_b"][None, :]
    h = n * 4.0
    gl = 0.5 * h * (1.0 + torch.tanh(0.7978845608 * (h + 0.044715 * h ** 3)))
    out = x + gl * inputs["gam"][None, :]
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="convnext_block_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "ln_g_ptr": "*fp32",
               "ln_b_ptr": "*fp32", "gam_ptr": "*fp32", "o_ptr": "*fp32",
               "HW": "constexpr", "C": "constexpr", "EPS": "constexpr"},
    constexprs={"HW": HW, "C": C, "EPS": EPS},
    args=[Arg("x", IN, "float32", (HW * C,)),
          Arg("w", IN, "float32", (C,)),
          Arg("ln_g", IN, "float32", (C,)), Arg("ln_b", IN, "float32", (C,)),
          Arg("gam", IN, "float32", (C,)),
          Arg("o", OUT, "float32", (HW * C,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
