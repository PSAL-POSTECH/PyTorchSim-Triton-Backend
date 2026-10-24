"""A row summed out of a loaded i64 tile, lanes on its columns, then read by every lane of a tile
whose sum runs along those columns: the conversion TritonGPU writes between the two reduces is
the XLU all-gather, the wide element in two halves. Breaks if p02 lays it out as a widened copy."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, Y = 16, 32, 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_all_gather_i64_loaded_row_kernel(x_ptr, w_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr, Y: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :])
        s = tl.sum(x, axis=0)
        yc = tl.arange(0, Y)[:, None] * C + tl.arange(0, C)[None, :]
        t = s[None, :] * (tl.arange(0, C)[None, :] + 1).to(tl.int64) + tl.load(w_ptr + yc)
        tl.store(o_ptr + tl.arange(0, Y), tl.sum(t, axis=1))

    return ops_xlu_all_gather_i64_loaded_row_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(-1000, 1000, (R, C), generator=g, dtype=torch.int64),
            "w": torch.randint(-1000, 1000, (Y, C), generator=g, dtype=torch.int64)}


def reference(inputs):
    import torch
    s = inputs["x"].sum(0) * (torch.arange(C, dtype=torch.int64) + 1)
    return {"o": (s[None, :] + inputs["w"]).sum(1)}


SPEC = KernelSpec(
    name="ops_xlu_all_gather_i64_loaded_row_kernel", kernel=kernel,
    signature={"x_ptr": "*i64", "w_ptr": "*i64", "o_ptr": "*i64",
               "R": "constexpr", "C": "constexpr", "Y": "constexpr"},
    constexprs={"R": R, "C": C, "Y": Y},
    args=[Arg("x", IN, "int64", (R, C)), Arg("w", IN, "int64", (Y, C)),
          Arg("o", OUT, "int64", (Y,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # ONE GATHER AND NO ROUND TRIP.
    expect_ir={"08-transformed.mlir": {"xlu_op = 5 : i64": 1,
                                       "xlu_op = 4 : i64": 0}},
)
