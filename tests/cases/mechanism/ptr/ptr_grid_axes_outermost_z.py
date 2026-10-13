"""A (4, 3, 2) grid launched with z outermost: `parallel_axes=("z", "y", "x")`.

Guards the shape a TOG is run with: one extent per parallel axis in the order of the axes, so the
z loop runs to the z extent. Handed the grid as (x, y, z) the z loop runs to 4 and the x loop to 2.

WHY THE GRID IS (4, 3, 2)
  Three different extents, and the outermost axis has the fewest programs: a z loop that runs to
  the x extent reaches row 2 + 3 * 3 = 11 of 6, which togcheck's footprint names. With the larger
  extent on z the swapped loops would stay inside the arguments and only fold blocks together.

ARITHMETIC
  o[pid_y + pid_z * NY, pid_x * 128 + c] = x[the same] + 1, 6 rows of 512. Exact.
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
    def ptr_grid_axes_outermost_z_kernel(x_ptr, o_ptr, NY: tl.constexpr, COLS: tl.constexpr,
                                         BLOCK: tl.constexpr):
        row = tl.program_id(1) + tl.program_id(2) * NY
        offs = row * COLS + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(o_ptr + offs, tl.load(x_ptr + offs) + 1.0)

    return ptr_grid_axes_outermost_z_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(ROWS, COLS, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"] + 1.0}


SPEC = KernelSpec(
    name="ptr_grid_axes_outermost_z_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "NY": "constexpr", "COLS": "constexpr",
               "BLOCK": "constexpr"},
    constexprs={"NY": NY, "COLS": COLS, "BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (ROWS, COLS)), Arg("o", OUT, "float32", (ROWS, COLS))],
    grid=(NX, NY, NZ), parallel_axes=("z", "y", "x"), reference=reference,
    make_inputs=make_inputs, expect_stage="togcheck",
)
