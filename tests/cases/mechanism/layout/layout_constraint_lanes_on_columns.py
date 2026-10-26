"""A kernel keeps a 128x128 tile with its lanes on axis 1 (layout_constraint, lane_layout(1)) where the
compiler alone puts them on axis 0: the load, the arithmetic after it and the store all take the kernel's
layout, no convert is left, and the pin is gone. Breaks if the constraint is dropped or does not propagate."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 128


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import pytorchsim

    @triton.jit
    def layout_constraint_columns_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        x = pytorchsim.layout_constraint(tl.load(x_ptr + offs), pytorchsim.lane_layout(1))
        tl.store(o_ptr + offs, x * 2.0 + 1.0)

    return layout_constraint_columns_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"] * 2.0 + 1.0}


SPEC = KernelSpec(
    name="layout_constraint_columns_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"04-anchored.mlir": {"pytorchsim_extern.layout_constraint": 1},
               "05-layouts.mlir": {"pytorchsim_extern.layout_constraint": 0, "ttg.convert_layout": 0,
                                   "threadsPerWarp = [128, 1]": 0}},
)
