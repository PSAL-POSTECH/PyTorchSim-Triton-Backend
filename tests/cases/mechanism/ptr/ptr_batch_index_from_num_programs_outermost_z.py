"""The batch index `pid_y + pid_z * num_programs(1)` on a grid launched with z outermost.

Guards the pairing of a number of programs with its axis when the launch order is not (x, y, z):
`parallel_axes=("z", "y", "x")` is the order Inductor's templates are launched in.

WHY THE GRID IS (4, 3, 2)
  Three different extents with the fewest programs on the outermost axis. A z loop or a
  num_programs(2) that takes the x extent reaches row 11 of 6, and togcheck's footprint names it;
  with the larger extent on z the exchanged extents would stay inside the arguments.

ARITHMETIC
  As ptr_batch_index_from_num_programs: o[NY * NZ - 1 - q, c] = x[q, c] + 1, 6 rows of 512. Exact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK = 128
NX, NY, NZ = 4, 3, 2
ROWS, COLS = NY * NZ, NX * BLOCK


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_batch_index_from_num_programs_outermost_z_kernel(x_ptr, o_ptr, COLS: tl.constexpr, BLOCK: tl.constexpr):
        q = tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)
        last = tl.num_programs(1) * tl.num_programs(2) - 1
        cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + q * COLS + cols)
        tl.store(o_ptr + (last - q) * COLS + cols, x + 1.0)

    return ptr_batch_index_from_num_programs_outermost_z_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(ROWS, COLS, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.flip(inputs["x"], dims=(0,)) + 1.0}


SPEC = KernelSpec(
    name="ptr_batch_index_from_num_programs_outermost_z_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "COLS": "constexpr", "BLOCK": "constexpr"},
    constexprs={"COLS": COLS, "BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (ROWS, COLS)), Arg("o", OUT, "float32", (ROWS, COLS))],
    grid=(NX, NY, NZ), parallel_axes=("z", "y", "x"), reference=reference, make_inputs=make_inputs,
    expect_stage="togcheck",
)
