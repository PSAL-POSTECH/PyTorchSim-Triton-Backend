"""Gemma's RMSNorm, x * rsqrt(mean(x^2) + eps) * (1 + w): w + 1 is a load and one add, so it is made again
in the rows' layout, every lane reading w whole, and no convert gathers it across the lanes. Breaks if
remove-layout-conversions keeps (1 + w) in the columns' layout and converts it (an XLU all-gather)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, BM = 512, 1024, 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def layout_remat_one_plus_weight_kernel(x_ptr, w_ptr, o_ptr, N: tl.constexpr, BM: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)[:, None]
        cols = tl.arange(0, N)[None, :]
        x = tl.load(x_ptr + rows * N + cols).to(tl.float32)
        ms = tl.sum(x * x, 1)[:, None] / N
        y = x * tl.math.rsqrt(ms + 1e-6)
        w = tl.load(w_ptr + cols).to(tl.float32)
        tl.store(o_ptr + rows * N + cols, (y * (w + 1.0)).to(tl.bfloat16))

    return layout_remat_one_plus_weight_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32).to(torch.bfloat16),
            "w": torch.randn(N, generator=g, dtype=torch.float32).to(torch.bfloat16)}


def reference(inputs):
    import torch
    x, w = inputs["x"].to(torch.float32), inputs["w"].to(torch.float32)
    y = x * torch.rsqrt((x * x).mean(1, keepdim=True) + 1e-6)
    return {"o": (y * (w + 1.0)).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="layout_remat_one_plus_weight_kernel",
    kernel=kernel,
    signature={"x_ptr": "*bf16", "w_ptr": "*bf16", "o_ptr": "*bf16", "N": "constexpr", "BM": "constexpr"},
    constexprs={"N": N, "BM": BM},
    args=[
        Arg("x", IN, "bfloat16", (M, N)),
        Arg("w", IN, "bfloat16", (N,)),
        Arg("o", OUT, "bfloat16", (M, N)),
    ],
    grid=(M // BM, 1, 1),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
    expect_ir={"05-layouts.mlir": {"ttg.convert_layout": 0}},
)
