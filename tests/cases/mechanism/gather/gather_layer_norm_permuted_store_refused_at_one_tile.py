"""ConvNeXt's layer norm over 96 channels, stored permuted, as Inductor tiled it: a compiler gap at XBLOCK 8192 (open, ours to fix), built at 64.

WHAT IT IS. A two-pass mean and variance over R0_BLOCK-wide chunks, the result stored at `x2 + 2*x4 + 4*r + 384*x3 +
10752*x5` under the tail mask. At XBLOCK 8192 the layout cuts the x axis into digits and that store is a gathering
DMA over a cut dim, which materialize_layouts refuses (Reason.DIGIT_REFUSED).

WHAT IT PINS. The refusal is a decision about THIS tile: tests/runtime/triton_backend_cases.py compiles the file at
XBLOCK 64 and gets an ELF, and at 8192 gets RefusedConfig, so autotune drops the 8192 config and goes on.

WHAT WOULD BREAK THIS. Cutting the x axis at 8192 without the gather refusing (then this builds and the test of
the drop has nothing to drop); or refusing at 64 as well (then no config of the kernel compiles).

torch equivalent: o[x2 + 2*x4 + 4*r + 384*x3 + 10752*x5] = layer_norm(a + b)[x, r] * w + bias, x = x2 + 2*x3 + 56*x4 + 112*x5
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

XN, RN = 6272, 96
XBLOCK, R0_BLOCK = 8192, 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_layer_norm_permuted_store_refused_at_one_tile(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2,
                                                             XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
        xnumel = 6272
        r0_numel = 96
        xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        x0 = xindex
        acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        acc2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + 96*x0), r0_mask & xmask, other=0.0).to(tl.float32)
            tmp1 = tl.load(in_ptr1 + (r0_1 + 96*x0), r0_mask & xmask, other=0.0).to(tl.float32)
            tmp2 = tmp0 + tmp1
            acc += tl.where(r0_mask & xmask, tmp2, 0.0)
            acc2 += tl.where(r0_mask & xmask, tmp2 * tmp2, 0.0)
        tmp5 = (tl.sum(acc, 1) / 96.0)[:, None]
        tmp9 = (tl.sum(acc2, 1) / 96.0)[:, None] - tmp5 * tmp5
        x2 = (xindex % 2)
        x3 = ((xindex // 2) % 28)
        x4 = ((xindex // 56) % 2)
        x5 = xindex // 112
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            r0_1 = r0_index
            tmp11 = tl.load(in_ptr0 + (r0_1 + 96*x0), r0_mask & xmask, other=0.0).to(tl.float32)
            tmp12 = tl.load(in_ptr1 + (r0_1 + 96*x0), r0_mask & xmask, other=0.0).to(tl.float32)
            tmp22 = tl.load(in_ptr2 + (r0_1), r0_mask, other=0.0).to(tl.float32)
            tmp25 = tl.load(in_ptr3 + (r0_1), r0_mask, other=0.0).to(tl.float32)
            tmp15 = (tmp11 + tmp12) - tmp5
            tmp20 = tl.math.rsqrt(tmp9 + 1e-06)
            tmp27 = tmp15 * tmp20 * tmp22 + tmp25
            tl.store(out_ptr2 + (x2 + 2*x4 + 4*r0_1 + 384*x3 + 10752*x5), tmp27.to(tl.bfloat16), r0_mask & xmask)
    return gather_layer_norm_permuted_store_refused_at_one_tile


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).to(torch.bfloat16)
    return {"a": r(XN * RN), "b": r(XN * RN), "w": r(RN), "bias": r(RN)}


def reference(inputs):
    import torch
    x = (inputs["a"].float() + inputs["b"].float()).reshape(XN, RN)
    y = torch.nn.functional.layer_norm(x, (RN,), inputs["w"].float(), inputs["bias"].float(), 1e-6)
    i = torch.arange(XN)
    at = (i % 2) + 2 * ((i // 56) % 2) + 384 * ((i // 2) % 28) + 10752 * (i // 112)
    o = torch.zeros(602112)
    o[at[:, None] + 4 * torch.arange(RN)[None, :]] = y
    return {"o": o.to(torch.bfloat16)}


SPEC = KernelSpec(
    name="gather_layer_norm_permuted_store_refused_at_one_tile", kernel=kernel,
    signature={"in_ptr0": "*bf16", "in_ptr1": "*bf16", "in_ptr2": "*bf16", "in_ptr3": "*bf16",
               "out_ptr2": "*bf16", "XBLOCK": "constexpr", "R0_BLOCK": "constexpr"},
    constexprs={"XBLOCK": XBLOCK, "R0_BLOCK": R0_BLOCK},
    args=[Arg("a", IN, "bfloat16", (XN * RN,)), Arg("b", IN, "bfloat16", (XN * RN,)),
          Arg("w", IN, "bfloat16", (RN,)), Arg("bias", IN, "bfloat16", (RN,)),
          Arg("o", OUT, "bfloat16", (602112,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
    expect="open", expect_stop=Stop("transform", Reason.DIGIT_REFUSED),
    expect_why="the permuted store is a gathering DMA over a dim cut into digits at XBLOCK 8192; XBLOCK 64 builds",
)
