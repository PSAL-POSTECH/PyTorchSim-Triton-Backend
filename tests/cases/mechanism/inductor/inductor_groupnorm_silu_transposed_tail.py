"""Inductor's group-norm epilogue on a channel-last conv output read transposed (y0 + 64*x1), one
2048-wide x block over 1280 live channels, per-group stats gathered through x1 // 40.

WHAT IT PINS. pytorchsim-drop-unread-padding joins the axes one generic walks with one dim; a
view axis that is the laps of a tile's lane axis ((LAP, axis)) was not joined, so the 128x10 view of a
tile cut to 1280 fed a 128x4x16 broadcast that was not, and verify failed in _cut.
Before: "operand #1 has shape's dimension #2 to be 10, but found 16". After: the class keeps 2048 whole.

torch equivalent: silu(((x^T + b) - mean[c // 40]) * rsqrt(var[c // 40] / 2560 + eps) * w + beta).
Found by Diffusion's triton_poi_fused_convolution_native_group_norm_silu_86 (YBLOCK 4, XBLOCK 2048).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

Y, X, GS = 64, 1280, 40


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def inductor_groupnorm_silu_transposed_tail_kernel(
            in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1,
            YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
        ynumel = 64
        xnumel = 1280
        yoffset = tl.program_id(1) * YBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        ymask = yindex < ynumel
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        xmask = xindex < xnumel
        x1 = xindex
        y0 = yindex
        tmp0 = tl.load(in_ptr0 + (y0 + 64 * x1), xmask & ymask).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x1), xmask, other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (x1 // 40), xmask, other=0.0)
        tmp6 = tl.load(in_ptr3 + (x1 // 40), xmask, other=1.0)
        tmp13 = tl.load(in_ptr4 + (x1), xmask, other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr5 + (x1), xmask, other=0.0).to(tl.float32)
        tmp5 = tmp0 + tmp1 - tmp4
        tmp8 = tmp6 / tl.full([1, 1], 2560.0, tl.float32)
        tmp11 = 1.0 / tl.sqrt(tmp8 + tl.full([1, 1], 1e-05, tl.float32))
        tmp18 = tmp5 * tmp11 * tmp13 + tmp16
        tmp25 = tmp18 / (tl.exp(-tmp18) + tl.full([1, 1], 1.0, tl.float32))
        tl.store(out_ptr1 + (x1 + 1280 * y0), tmp25.to(tl.bfloat16), xmask & ymask)

    return inductor_groupnorm_silu_transposed_tail_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda n, dt: torch.randn(n, generator=g).to(dt)
    return {"in_ptr0": r(Y * X, torch.bfloat16), "in_ptr1": r(X, torch.bfloat16),
            "in_ptr2": r(X // GS, torch.float32), "in_ptr3": r(X // GS, torch.float32).abs() * 2560 + 1,
            "in_ptr4": r(X, torch.bfloat16), "in_ptr5": r(X, torch.bfloat16)}


def reference(inputs):
    import torch
    f = lambda k: inputs[k].float()
    c = torch.arange(X) // GS
    x = f("in_ptr0").reshape(X, Y).t()
    v = (x + f("in_ptr1")[None] - f("in_ptr2")[c][None]) * torch.rsqrt(
        f("in_ptr3")[c][None] / 2560 + 1e-5) * f("in_ptr4")[None] + f("in_ptr5")[None]
    return {"out_ptr1": (v / (torch.exp(-v) + 1)).to(torch.bfloat16).reshape(-1)}


SPEC = KernelSpec(
    name="inductor_groupnorm_silu_transposed_tail_kernel", kernel=kernel,
    signature={"in_ptr0": "*bf16", "in_ptr1": "*bf16", "in_ptr2": "*fp32", "in_ptr3": "*fp32",
               "in_ptr4": "*bf16", "in_ptr5": "*bf16", "out_ptr1": "*bf16",
               "YBLOCK": "constexpr", "XBLOCK": "constexpr"},
    constexprs={"YBLOCK": 4, "XBLOCK": 2048},
    args=[Arg("in_ptr0", IN, "bfloat16", (Y * X,)), Arg("in_ptr1", IN, "bfloat16", (X,)),
          Arg("in_ptr2", IN, "float32", (X // GS,)), Arg("in_ptr3", IN, "float32", (X // GS,)),
          Arg("in_ptr4", IN, "bfloat16", (X,)), Arg("in_ptr5", IN, "bfloat16", (X,)),
          Arg("out_ptr1", OUT, "bfloat16", (Y * X,))],
    grid=(1, 16), parallel_axes=("y", "x"),
    reference=reference, make_inputs=make_inputs,
)
