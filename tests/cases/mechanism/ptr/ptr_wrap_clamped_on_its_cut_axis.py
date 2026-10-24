"""A static mask on the wrapped axis itself, load and store: each piece's box meets the clamp, the
load's lanes past it get one fill piece, the store just drops them -- five pieces, not two gathers.
Breaks if p01 sends a statically masked cut axis per-element again, or p11 refuses the clamp."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, W, O, M = 32, 24, 4, 26
S, SO, SM = 16, 12, 14


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_clamped_on_its_cut_axis_kernel(x_ptr, o_ptr, y_ptr, N: tl.constexpr,
                                                W: tl.constexpr, O: tl.constexpr,
                                                M: tl.constexpr, S: tl.constexpr,
                                                SO: tl.constexpr, SM: tl.constexpr):
        p = tl.arange(0, N)
        v = tl.load(x_ptr + (p + O) % W, mask=p < M, other=-1.0)
        tl.store(o_ptr + p, v)
        q = tl.arange(0, S)
        tl.store(y_ptr + (q + SO) % W, tl.load(x_ptr + q) + 1.0, mask=q < SM)

    return ptr_wrap_clamped_on_its_cut_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(W, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    p = torch.arange(N)
    y = torch.zeros(W, dtype=torch.float32)
    q = torch.arange(SM)
    y[(q + SO) % W] = x[q] + 1.0
    return {"o": torch.where(p < M, x[(p + O) % W], torch.tensor(-1.0)), "y": y}


SPEC = KernelSpec(
    name="ptr_wrap_clamped_on_its_cut_axis_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "y_ptr": "*fp32", "N": "constexpr",
               "W": "constexpr", "O": "constexpr", "M": "constexpr", "S": "constexpr",
               "SO": "constexpr", "SM": "constexpr"},
    constexprs={"N": N, "W": W, "O": O, "M": M, "S": S, "SO": SO, "SM": SM},
    args=[Arg("x", IN, "float32", (W,)), Arg("o", OUT, "float32", (N,)),
          Arg("y", OUT, "float32", (W,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"09-custom.mlir": {"pytorchsim.transfer": 7}},
)
