"""Pins a loop that holds no DMA and whose trip count is read from data as a loop the TOG replays:
n = 3 trips of compute over a tile loaded before it, the bound a runtime attribute and the sample one
trip. Breaks if the loop is refused, or timed inside one segment on the sample's data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, N = 8, 128, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_compute_trip_from_data_tog(x_ptr, n_ptr, y_ptr, R: tl.constexpr, C: tl.constexpr):
        offs = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
        x = tl.load(x_ptr + offs)
        y = x
        for _ in range(tl.load(n_ptr)):
            y = tl.exp(y * 0.5) - x
        tl.store(y_ptr + offs, y)

    return loop_compute_trip_from_data_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R, C, generator=g) * 0.5, "n": torch.tensor([N], dtype=torch.int32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    y = x
    for _ in range(int(inputs["n"][0])):
        y = torch.exp(y * 0.5) - x
    return {"y": y}


SPEC = KernelSpec(
    name="loop_compute_trip_from_data_tog", kernel=kernel,
    signature={"x_ptr": "*fp32", "n_ptr": "*i32", "y_ptr": "*fp32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (R, C)), Arg("n", IN, "int32", (1,)), Arg("y", OUT, "float32", (R, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
)
