"""Known regression pinned (gemma3 q/k norm + RoPE, w of 256): (1 + w) is made again in the rows' layout (no
convert left, 05), 7,140 -> 7,461 cycles: the gather it replaces sat in its own XLU tile, hidden beside the VPU's
work; only TOG tiling shows that, and this 05 is shaped as gemma2's, which gains. Flip to 1 if that is fixed."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, D, H = 1024, 256, 128


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def layout_remat_small_weight_beside_xlu_kernel(o_ptr, x_ptr, w_ptr, cos_ptr, sin_ptr, XBLOCK: tl.constexpr):
        x0 = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        r = tl.arange(0, 256)[None, :]
        x = tl.load(x_ptr + (r + 256 * x0)).to(tl.float32)
        w = tl.load(w_ptr + r).to(tl.float32)
        cs = tl.load(cos_ptr + (r + 256 * x0)).to(tl.float32)
        sn = tl.load(sin_ptr + (r + 256 * x0)).to(tl.float32)
        rs = libdevice.rsqrt(tl.sum(x * x, 1)[:, None] / 256.0 + 1e-06)
        lo = r < 128
        hi = r >= 128
        xh = tl.load(x_ptr + (128 + 256 * x0 + r), lo, other=0.0).to(tl.float32)
        wh = tl.load(w_ptr + tl.broadcast_to(128 + r, [XBLOCK, 256]), lo, other=0.0).to(tl.float32)
        xl = tl.load(x_ptr + (256 * x0 + (r - 128)), hi, other=0.0).to(tl.float32)
        wl = tl.load(w_ptr + tl.broadcast_to(r - 128, [XBLOCK, 256]), hi, other=0.0).to(tl.float32)
        rot = tl.where(lo, -(xh * rs * (wh + 1.0)), tl.where(hi, xl * rs * (wl + 1.0), 0.0))
        n = x * rs * (w + 1.0)
        tl.store(o_ptr + (r + 256 * x0), (n * cs + rot * sn).to(tl.bfloat16))

    return layout_remat_small_weight_beside_xlu_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32).to(torch.bfloat16)
    return {"x": r(M, D), "w": r(D), "cos": r(M, D), "sin": r(M, D)}


def reference(inputs):
    import torch
    x, w = inputs["x"].float(), inputs["w"].float()
    n = x * torch.rsqrt((x * x).mean(1, keepdim=True) + 1e-6) * (w + 1.0)
    rot = torch.cat([-n[:, H:], n[:, :H]], 1)
    return {"o": (n * inputs["cos"].float() + rot * inputs["sin"].float()).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="layout_remat_small_weight_beside_xlu_kernel",
    kernel=kernel,
    signature={"o_ptr": "*bf16", "x_ptr": "*bf16", "w_ptr": "*bf16", "cos_ptr": "*bf16", "sin_ptr": "*bf16",
               "XBLOCK": "constexpr"},
    constexprs={"XBLOCK": M},
    args=[
        Arg("o", OUT, "bfloat16", (M, D)),
        Arg("x", IN, "bfloat16", (M, D)),
        Arg("w", IN, "bfloat16", (D,)),
        Arg("cos", IN, "bfloat16", (M, D)),
        Arg("sin", IN, "bfloat16", (M, D)),
    ],
    grid=(1, 1, 1),
    reference=reference,
    make_inputs=make_inputs,
    rtol=3e-2, atol=3e-2,
    expect_ir={"05-layouts.mlir": {"ttg.convert_layout": 0}},
    expect_stage="timing",
)
