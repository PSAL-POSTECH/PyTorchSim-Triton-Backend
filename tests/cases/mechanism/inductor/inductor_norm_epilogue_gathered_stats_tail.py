"""A layer-norm epilogue over channels-first rows whose numel is no power of two of its block.

    x     [B, C, W] flat, xnumel = B*C*W = 2*6*256 = 3072 of an XBLOCK of 4096
    mean  [B, W], var [B, W]   read at x0 + W*x2: a gather into a [lanes, laps, ...] tile
                               the epilogue reads through a collapse that merges its axes

pytorchsim-drop-unread-padding cut the 1-D buffers of the epilogue's generic to 3072 while the
collapsed views of the gathered stats, read whole, kept 4096: verify then refused the generic
("operand #2 has shape's dimension #0 to be 3072, but found 4096"). From ConvNeXtV2's fused
convolution + native_layer_norm + permute (96 channels, 49152 of 65536). The merged tile is
now cut on its outermost axis below the lanes, [128, 8, 2, 2] -> [128, 6, 2, 2]; cut on the
lane axis instead ([96, 8, 2, 2]) its rows land in other lanes (max_abs_err 3170 on Spike).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT, INOUT

B, C, W = 2, 6, 256
XNUMEL = B * C * W
XBLOCK = 4096


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def inductor_norm_epilogue_gathered_stats_tail_kernel(
            in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, XBLOCK: tl.constexpr):
        xnumel = 3072
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x1 = ((xindex // 256) % 6)
        x0 = (xindex % 256)
        x2 = xindex // 1536
        tmp0 = tl.load(in_out_ptr0 + (xindex), xmask)
        tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0 + 256 * x2), xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr2 + (x0 + 256 * x2), xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1], 6.0, tl.float32)
        tmp7 = (tmp5 / tmp6)
        tmp8 = tl.full([1], 1e-06, tl.float32)
        tmp9 = tmp7 + tmp8
        tmp10 = tl.math.rsqrt(tmp9)
        tmp11 = tmp4 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tl.store(in_out_ptr0 + (xindex), tmp15, xmask)

    return inductor_norm_epilogue_gathered_stats_tail_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(XNUMEL, generator=g),
            "bias": torch.randn(C, generator=g),
            "mean": torch.randn(B * W, generator=g),
            "var": torch.rand(B * W, generator=g) + 0.5,
            "weight": torch.randn(C, generator=g),
            "shift": torch.randn(C, generator=g)}


def reference(inputs):
    import torch
    x = inputs["x"].view(B, C, W) + inputs["bias"].view(1, C, 1)
    mean, var = inputs["mean"].view(B, 1, W), inputs["var"].view(B, 1, W)
    y = (x - mean) * torch.rsqrt(var / C + 1e-6)
    y = y * inputs["weight"].view(1, C, 1) + inputs["shift"].view(1, C, 1)
    return {"x": y.reshape(-1)}


SPEC = KernelSpec(
    name="inductor_norm_epilogue_gathered_stats_tail_kernel", kernel=kernel,
    signature={"in_out_ptr0": "*fp32", "in_ptr0": "*fp32", "in_ptr1": "*fp32",
               "in_ptr2": "*fp32", "in_ptr3": "*fp32", "in_ptr4": "*fp32",
               "XBLOCK": "constexpr"},
    constexprs={"XBLOCK": XBLOCK},
    args=[Arg("x", INOUT, "float32", (XNUMEL,)), Arg("bias", IN, "float32", (C,)),
          Arg("mean", IN, "float32", (B * W,)), Arg("var", IN, "float32", (B * W,)),
          Arg("weight", IN, "float32", (C,)), Arg("shift", IN, "float32", (C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
