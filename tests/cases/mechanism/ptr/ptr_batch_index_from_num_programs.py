"""A row address built from `tl.num_programs`: the batch index `pid_y + pid_z * num_programs(1)`.

Guards that a kernel whose addresses read a number of programs has a TOG at all (Inductor's bmm
template rebuilds its batch index this way), and that the number read is its own axis's extent.

WHY THE GRID IS (2, 3, 4)
  Three different extents and two axes past one program, so an extent read off another axis, a
  program id read for an extent, or an extent off by one is a different row.

WHY THE STORE IS MIRRORED
  The load's row is q = pid_y + pid_z * NY and the store's is NY * NZ - 1 - q. A number too small
  sends a store below row 0 and one too large sends a transfer past the last row, and togcheck's
  footprint names either; a plain `q` on both sides would only fold rows together.

WHAT THIS DOES NOT SEE
  Where the number comes from. A TOG that bakes the launch grid in gives the same addresses on
  this grid: tests/passes/grid_extent_cases.py replays this kernel's TOG on a second grid.

ARITHMETIC
  o[NY * NZ - 1 - q, c] = x[q, c] + 1 for the 12 rows q and the 256 columns c. Exact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK = 128
NX, NY, NZ = 2, 3, 4
ROWS, COLS = NY * NZ, NX * BLOCK


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_batch_index_from_num_programs_kernel(x_ptr, o_ptr, COLS: tl.constexpr, BLOCK: tl.constexpr):
        q = tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)
        last = tl.num_programs(1) * tl.num_programs(2) - 1
        cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + q * COLS + cols)
        tl.store(o_ptr + (last - q) * COLS + cols, x + 1.0)

    return ptr_batch_index_from_num_programs_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(ROWS, COLS, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.flip(inputs["x"], dims=(0,)) + 1.0}


SPEC = KernelSpec(
    name="ptr_batch_index_from_num_programs_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "COLS": "constexpr", "BLOCK": "constexpr"},
    constexprs={"COLS": COLS, "BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (ROWS, COLS)), Arg("o", OUT, "float32", (ROWS, COLS))],
    grid=(NX, NY, NZ), reference=reference, make_inputs=make_inputs,
    expect_stage="togcheck",
)
