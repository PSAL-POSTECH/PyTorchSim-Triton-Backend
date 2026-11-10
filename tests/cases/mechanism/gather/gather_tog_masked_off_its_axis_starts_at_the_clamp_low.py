"""A row gather `x_ptr + idx[:, None] * 40 + ((-24) + j)` masked off its gather axis (`j >= 24`): its index
snapshot is one entry per row and the TOG's DMA moves Dims(8, 40) from 0, not [8, 64] from -24 (x[-24..-1]).
Breaks if the snapshot goes back to the whole tile or build_skeleton stops sizing a gather's off-row clamp."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

V, N, C, CL = 32, 8, 64, 24
W = C - CL


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_tog_masked_off_its_axis_starts_at_the_clamp_low(x_ptr, idx_ptr, y_ptr, N: tl.constexpr,
                                                        C: tl.constexpr, CL: tl.constexpr, W: tl.constexpr):
        i = tl.arange(0, N)
        j = tl.arange(0, C)[None, :]
        rows = tl.load(idx_ptr + i)
        v = tl.load(x_ptr + rows[:, None] * W + ((-CL) + j), mask=j >= CL, other=0.0)
        tl.store(y_ptr + i[:, None] * C + j, v)

    return gather_tog_masked_off_its_axis_starts_at_the_clamp_low


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(V, W, generator=g),
            "idx": torch.randint(0, V, (N,), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    y = torch.zeros(N, C)
    y[:, CL:] = inputs["x"][inputs["idx"].long()]
    return {"y": y}


SPEC = KernelSpec(
    name="gather_tog_masked_off_its_axis_starts_at_the_clamp_low", kernel=kernel,
    signature={"x_ptr": "*fp32", "idx_ptr": "*i32", "y_ptr": "*fp32", "N": "constexpr", "C": "constexpr",
               "CL": "constexpr", "W": "constexpr"},
    constexprs={"N": N, "C": C, "CL": CL, "W": W},
    args=[Arg("x", IN, "float32", (V, W)), Arg("idx", IN, "int32", (N,)), Arg("y", OUT, "float32", (N, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"tog.cpp": {"-24": 0, "togsim::Dims(8, 64)": 1}}, expect_stage="togcheck",
)
