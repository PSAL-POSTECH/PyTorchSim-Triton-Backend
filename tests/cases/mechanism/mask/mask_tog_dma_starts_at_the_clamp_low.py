"""A load masked from below, `x_ptr + ((-64) + r)` with `r >= 64` (Qwen3 RoPE's rotate_half): the TOG
DMA reads x[0..63], from the clamp's low corner. Breaks if the TOG sizes the DMA to the mask but keeps
the unmasked start -64 (tog.cpp names -64, togcheck finds x read from -64)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, H = 128, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_tog_dma_starts_at_the_clamp_low(x_ptr, y_ptr, N: tl.constexpr, H: tl.constexpr):
        r = tl.arange(0, N)
        tl.store(y_ptr + r, tl.load(x_ptr + ((-H) + r), mask=r >= H, other=0.0))

    return mask_tog_dma_starts_at_the_clamp_low


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(H, generator=g)}


def reference(inputs):
    import torch
    return {"y": torch.cat([torch.zeros(N - H), inputs["x"]])}


SPEC = KernelSpec(
    name="mask_tog_dma_starts_at_the_clamp_low", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "N": "constexpr", "H": "constexpr"},
    constexprs={"N": N, "H": H},
    args=[Arg("x", IN, "float32", (H,)), Arg("y", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"tog.cpp": {"-64": 0}}, expect_stage="togcheck",
)
