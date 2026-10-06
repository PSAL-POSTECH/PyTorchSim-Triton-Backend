"""BatchNorm + residual + ReLU -- what closes every ResNet bottleneck.

    y = relu(x * s + b + identity)

WHY BATCHNORM IS AN AFFINE HERE AND NOT A REDUCTION
  In training it normalises by the batch's own mean and variance. In INFERENCE
  the running statistics are constants, so

      (x - mean) / sqrt(var + eps) * gamma + beta
        = x * [gamma / sqrt(var + eps)] + [beta - mean * that]
        = x * s + b

  collapses to two per-channel constants and the reduction disappears. That is
  why this takes `s` and `b` as channel vectors: writing the unfolded form would
  put a reduction in a kernel that does not have one, and would measure
  something ResNet inference never runs.

WHAT IT EXERCISES
  A per-CHANNEL broadcast against a [C, HW] tile -- `s[:, None]`, the column
  direction. With dim 0 (channels) on the lanes, each lane holds one channel's
  whole row and its own scale, so the splat stays inside the bank and costs
  nothing. The second arm does the same scale along the OTHER axis
  (`s[None, :]` against a [HW, C] tile) because that is the layout a
  channels-last model hands over, and only one of the two is free.

  The residual add is elementwise and the ReLU is a max against zero; both are
  here because the point is the FUSED shape -- 16 blocks per forward pass end
  with exactly this, so it is the pattern to be fast at, not the three ops
  separately.

torch equivalent: relay = x * s[:, None] + b[:, None]; relu(relay + idn)

The fused tail of a ResNet bottleneck, 16 of them per forward pass. BatchNorm as the per-channel affine inference actually runs, and the ONE scale vector broadcast along both axes, which needs two stagings of one buffer and used to be refused.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C, HW = 8, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def bn_relu_add_kernel(x_ptr, s_ptr, b_ptr, idn_ptr, o_ptr, t_ptr,
                           C: tl.constexpr, HW: tl.constexpr):
        c = tl.arange(0, C)
        p = tl.arange(0, HW)

        # channels-first: [C, HW], the scale runs down the lane axis
        i0 = c[:, None] * HW + p[None, :]
        x = tl.load(x_ptr + i0)
        s = tl.load(s_ptr + c)
        b = tl.load(b_ptr + c)
        idn = tl.load(idn_ptr + i0)
        y = x * s[:, None] + b[:, None] + idn
        tl.store(o_ptr + i0, tl.maximum(y, 0.0))

        # channels-last: [HW, C], the same scale across the vector axis
        i1 = p[:, None] * C + c[None, :]
        xt = tl.load(t_ptr + i1)
        yt = xt * s[None, :] + b[None, :]
        tl.store(t_ptr + i1, tl.maximum(yt, 0.0))

    return bn_relu_add_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(C * HW, generator=g, dtype=torch.float32),
            "s": torch.randn(C, generator=g, dtype=torch.float32),
            "b": torch.randn(C, generator=g, dtype=torch.float32),
            "idn": torch.randn(C * HW, generator=g, dtype=torch.float32),
            "t": torch.randn(HW * C, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(C, HW)
    s, b = inputs["s"], inputs["b"]
    idn = inputs["idn"].reshape(C, HW)
    o = torch.relu(x * s[:, None] + b[:, None] + idn)
    t = inputs["t"].reshape(HW, C)
    return {"o": o.reshape(-1),
            "t": torch.relu(t * s[None, :] + b[None, :]).reshape(-1)}


SPEC = KernelSpec(
    name="bn_relu_add_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "s_ptr": "*fp32", "b_ptr": "*fp32",
               "idn_ptr": "*fp32", "o_ptr": "*fp32", "t_ptr": "*fp32",
               "C": "constexpr", "HW": "constexpr"},
    constexprs={"C": C, "HW": HW},
    args=[Arg("x", IN, "float32", (C * HW,)), Arg("s", IN, "float32", (C,)),
          Arg("b", IN, "float32", (C,)), Arg("idn", IN, "float32", (C * HW,)),
          Arg("o", OUT, "float32", (C * HW,)),
          Arg("t", "inout", "float32", (HW * C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
