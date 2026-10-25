"""Pins one vector over a lane's whole run of an elementwise tile: [4096, 4] over 128 lanes leaves
each lane 32 rows of 4, one contiguous run of 128 elements, so exp/mulf/truncf vectorize as
vector<128xf32> (legalize cuts that to what an instruction issues) instead of 32 trips of
vector<4xf32>. Breaks if b09_vectorize goes back to the innermost dim alone."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C = 4096, 4
SCALE = 0.125


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def lane_rows_one_run(x_ptr, y_ptr, R: tl.constexpr, C: tl.constexpr):
        rc = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
        tl.store(y_ptr + rc, (tl.exp(tl.load(x_ptr + rc)) * 0.125).to(tl.bfloat16))

    return lane_rows_one_run


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R, C, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"y": (torch.exp(inputs["x"]) * SCALE).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="lane_rows_one_run", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*bf16", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (R, C)), Arg("y", OUT, "bfloat16", (R, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
    expect_ir={"08-passes/10-pytorchsim-vectorize.mlir": {"vector<128xf32>": 5, "vector<4xf32>": 0}},
)
