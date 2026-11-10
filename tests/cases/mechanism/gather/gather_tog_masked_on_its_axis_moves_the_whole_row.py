"""BLOCKED, PINNED AS IT IS: a row gather masked on its gather axis, rows `i >= 3` of 8. TOGSim's Gather
reads its row from entry 0, one entry per position, and has no start position on its axis, so the TOG keeps
the static [8, 64] (masked rows request row 0); tog.cpp's two Dims(8, 64) become one when it can say so."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

V, N, C, NL = 32, 8, 64, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_tog_masked_on_its_axis_moves_the_whole_row(x_ptr, idx_ptr, y_ptr, N: tl.constexpr,
                                                          C: tl.constexpr, NL: tl.constexpr):
        i = tl.arange(0, N)
        j = tl.arange(0, C)[None, :]
        live = i >= NL
        rows = tl.load(idx_ptr + i, mask=live, other=0)
        v = tl.load(x_ptr + rows[:, None] * C + j, mask=live[:, None], other=0.0)
        tl.store(y_ptr + i[:, None] * C + j, v)

    return gather_tog_masked_on_its_axis_moves_the_whole_row


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(V, C, generator=g),
            "idx": torch.randint(0, V, (N,), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    y = torch.zeros(N, C)
    y[NL:] = inputs["x"][inputs["idx"][NL:].long()]
    return {"y": y}


SPEC = KernelSpec(
    name="gather_tog_masked_on_its_axis_moves_the_whole_row", kernel=kernel,
    signature={"x_ptr": "*fp32", "idx_ptr": "*i32", "y_ptr": "*fp32", "N": "constexpr", "C": "constexpr",
               "NL": "constexpr"},
    constexprs={"N": N, "C": C, "NL": NL},
    args=[Arg("x", IN, "float32", (V, C)), Arg("idx", IN, "int32", (N,)), Arg("y", OUT, "float32", (N, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"tog.cpp": {"togsim::Dims(8, 64)": 2}}, expect_stage="togcheck",
)
