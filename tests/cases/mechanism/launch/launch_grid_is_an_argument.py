"""The grid is a launch argument, not baked in: the wrapper reads gx/gy/gz from argv and kernel.json
states the grid_bound the compile assumed (here ceil(300 / 128) programs), past which a launch is refused.
Breaks if the wrapper bakes spec.grid again or the bound is left out of the manifest."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK = 128
N = 300


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def launch_grid_is_an_argument_kernel(x_ptr, o_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
        cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < N
        tl.store(o_ptr + cols, tl.load(x_ptr + cols, mask=mask, other=0.0) + 1.0, mask=mask)

    return launch_grid_is_an_argument_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"] + 1.0}


SPEC = KernelSpec(
    name="launch_grid_is_an_argument_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr", "BLOCK": "constexpr"},
    constexprs={"N": N, "BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(-(-N // BLOCK),), reference=reference, make_inputs=make_inputs,
    expect_ir={"kernel.json": {"grid_bound": 1}, "11-wrapper.c": {"atoi": 3}},
)
