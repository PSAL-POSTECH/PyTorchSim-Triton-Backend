"""x*y*z over 16 rows of 65536 f32: three 262144-byte tiles per lane against a 524288 budget, so the
binary refuses with a SpadOverflow and failure.json keeps usage and budget (tests/runtime/failure_cases).
Breaks if the refusal stops being a SpadOverflow, or if the tile stops riding one row per lane."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

M, N = 16, 65536


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def spad_over_budget(x_ptr, y_ptr, z_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        off = rm[:, None] * N + rn[None, :]
        x = tl.load(x_ptr + off)
        y = tl.load(y_ptr + off)
        z = tl.load(z_ptr + off)
        tl.store(o_ptr + off, x * y * z)

    return spad_over_budget


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32),
            "y": torch.randn(M, N, generator=g, dtype=torch.float32),
            "z": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"] * inputs["y"] * inputs["z"]}


SPEC = KernelSpec(
    name="spad_over_budget", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "z_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("y", IN, "float32", (M, N)),
          Arg("z", IN, "float32", (M, N)),
          Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="blocked",
    expect_stop=Stop("binary", Reason.SPAD_OVERFLOW),
    expect_why="three 262144-byte/lane tiles against a double-buffer budget of "
               "half the scratchpad: the tile is bigger than the machine, and "
               "the refusal leaves its usage and budget in failure.json for the "
               "caller that can shrink it",
)
