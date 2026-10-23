"""A mask whose bound reads `tl.num_programs`: `cols < num_programs(0) * BLOCK - CUT`.

Guards the masked extent of the last program in the TOG: it follows the grid the TOG is run with.
The last of 3 programs moves 88 of its 128 elements; a TOG that cannot compute the bound moves 128.

WHAT FAILS WITHOUT IT
  Not this file's own stages: values pass either way, and togcheck accepts the unclamped 128 as an
  approximation. tests/passes/grid_extent_cases.py records this kernel's TOG and reads the dims.

ARITHMETIC
  o[c] = x[c] + 1 for c < 3 * 128 - 40, else 0. Exact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK = 128
NX = 3
N = NX * BLOCK
CUT = 40


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_bound_from_num_programs_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr, CUT: tl.constexpr):
        cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < tl.num_programs(0) * BLOCK - CUT
        x = tl.load(x_ptr + cols, mask=mask, other=0.0)
        tl.store(o_ptr + cols, x + 1.0, mask=mask)

    return mask_bound_from_num_programs_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    o = torch.zeros(N)
    o[:N - CUT] = inputs["x"][:N - CUT] + 1.0
    return {"o": o}


SPEC = KernelSpec(
    name="mask_bound_from_num_programs_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BLOCK": "constexpr", "CUT": "constexpr"},
    constexprs={"BLOCK": BLOCK, "CUT": CUT},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(NX,), reference=reference, make_inputs=make_inputs,
)
