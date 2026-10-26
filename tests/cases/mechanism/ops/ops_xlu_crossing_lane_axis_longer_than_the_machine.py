"""A [512, 16] tile summed over both axes on 256 lanes: one sum reads it transposed, a crossing
whose lane axis is two lane-fulls. materialize_layouts writes one pass per lane-full, lower_to_xlu
reads them. Breaks if the transpose is priced as one round or its passes re-derived below.
The tile comes out of a loop: computed straight from the load, each sum would load its own
layout (upstream rematerializes a load rather than convert) and nothing would cross.

A crossing of a tile whose lane axis is twice the machine's lanes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C = 512, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_crossing_lane_axis_longer_than_the_machine_kernel(
            x_ptr, rows_ptr, cols_ptr, R: tl.constexpr, C: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :])
        y = tl.zeros((R, C), tl.float32)
        for _ in range(2):
            y += x * x + x
        tl.store(rows_ptr + tl.arange(0, R), tl.sum(y, axis=1))
        tl.store(cols_ptr + tl.arange(0, C), tl.sum(y, axis=0))

    return ops_xlu_crossing_lane_axis_longer_than_the_machine_kernel


def make_inputs(torch, seed=0):
    return {"x": torch.randn(R, C, generator=torch.Generator().manual_seed(seed))}


def reference(inputs):
    y = 2 * (inputs["x"] * inputs["x"] + inputs["x"])
    return {"rows": y.sum(dim=1), "cols": y.sum(dim=0)}


SPEC = KernelSpec(
    name="ops_xlu_crossing_lane_axis_longer_than_the_machine_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "rows_ptr": "*fp32", "cols_ptr": "*fp32",
               "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (R, C)), Arg("rows", OUT, "float32", (R,)),
          Arg("cols", OUT, "float32", (C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-3, expect_stage="timing",
    expect_ir={"08-transformed.mlir": {'mode = "transpose"': 1}},
)
