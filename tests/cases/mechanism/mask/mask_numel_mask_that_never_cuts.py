"""Inductor's numel mask `x < N` on a pointwise kernel whose block divides N: no program cuts, so no clamp is left.

Inductor keeps the mask whenever N is not a multiple of ITS max block (4096), whatever block the kernel runs at.
Here N = 3 * BLOCK = 6144, not a multiple of 4096: at every program id below the grid's bound the box is the whole tile, and
torchsim-resolve-masks makes its ends static. Breaks if a transfer keeps a runtime clamp (masked_axes).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK, GRID = 2048, 3
N = BLOCK * GRID


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_numel_mask_that_never_cuts_kernel(a_ptr, b_ptr, o_ptr, BLOCK: tl.constexpr, N: tl.constexpr):
        x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = x < N
        a = tl.load(a_ptr + x, m, other=0.0)
        b = tl.load(b_ptr + x, m, other=0.0)
        tl.store(o_ptr + x, a * b + 1.5, m)

    return mask_numel_mask_that_never_cuts_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(N, generator=g, dtype=torch.float32),
            "b": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["a"] * inputs["b"] + 1.5}


SPEC = KernelSpec(
    name="mask_numel_mask_that_never_cuts_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32", "BLOCK": "constexpr", "N": "constexpr"},
    constexprs={"BLOCK": BLOCK, "N": N},
    args=[Arg("a", IN, "float32", (N,)), Arg("b", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(GRID,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-transformed.mlir": {"masked_axes": 0}},
)
