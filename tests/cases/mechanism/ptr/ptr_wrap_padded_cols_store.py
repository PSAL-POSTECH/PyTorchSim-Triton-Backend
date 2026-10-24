"""96 of 128 columns stored at (c + 150) % 200 of each 200-wide row: the cut store's clamp takes
96 columns, so fit-to-numel cuts every buffer's in-lane axis to 96 as for a plain store.
Breaks if fit-to-numel skips a cut transfer again: the buffers stay 128 wide."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, INOUT

M, C, CB, W, O = 128, 96, 128, 200, 150


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_padded_cols_store_kernel(x_ptr, y_ptr, M: tl.constexpr, C: tl.constexpr,
                                          CB: tl.constexpr, W: tl.constexpr,
                                          O: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, CB)[None, :]
        m = cols < C
        x = tl.load(x_ptr + rows * C + cols, m, other=0.0)
        tl.store(y_ptr + rows * W + (cols + O) % W, x * 2.0 + 1.0, m)

    return ptr_wrap_padded_cols_store_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * C, generator=g, dtype=torch.float32),
            "y": torch.randn(M * W, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    y = inputs["y"].clone().reshape(M, W)
    y[:, (torch.arange(C) + O) % W] = (inputs["x"] * 2.0 + 1.0).reshape(M, C)
    return {"y": y.reshape(-1)}


SPEC = KernelSpec(
    name="ptr_wrap_padded_cols_store_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32",
               "M": "constexpr", "C": "constexpr", "CB": "constexpr", "W": "constexpr",
               "O": "constexpr"},
    constexprs={"M": M, "C": C, "CB": CB, "W": W, "O": O},
    args=[Arg("x", IN, "float32", (M * C,)), Arg("y", INOUT, "float32", (M * W,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=0, atol=0,
    expect_ir={"08-transformed.mlir": {"memref<128x128xf32": 0, "memref<128x96xf32": 6},
               "09-custom.mlir": {"pytorchsim.transfer": 3}},
)
