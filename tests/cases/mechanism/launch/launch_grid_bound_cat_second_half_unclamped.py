"""A cat whose numel mask is elided (numel % BLOCK == 0): the first half reads under `x < N`, the second under
`x >= N` with no high clamp, as Yolov5's fused add_cat_convolution_silu. The grid is 2N / BLOCK, not the
ceil(N / BLOCK) the `N - pid * BLOCK` clamp counts. Breaks if grid_bound takes one clamp as the numel mask."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK = 128
N = 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def launch_grid_bound_cat_second_half_unclamped_kernel(a_ptr, b_ptr, o_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
        x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        lo = x < N
        hi = x >= N
        va = tl.load(a_ptr + x, mask=lo, other=0.0)
        vb = tl.load(b_ptr + (x - N), mask=hi, other=0.0)
        tl.store(o_ptr + x, tl.where(lo, va, vb))

    return launch_grid_bound_cat_second_half_unclamped_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(N, generator=g, dtype=torch.float32), "b": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.cat([inputs["a"], inputs["b"]])}


SPEC = KernelSpec(
    name="launch_grid_bound_cat_second_half_unclamped_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr", "BLOCK": "constexpr"},
    constexprs={"N": N, "BLOCK": BLOCK},
    args=[Arg("a", IN, "float32", (N,)), Arg("b", IN, "float32", (N,)), Arg("o", OUT, "float32", (2 * N,))],
    grid=(2 * N // BLOCK,), reference=reference, make_inputs=make_inputs,
)
