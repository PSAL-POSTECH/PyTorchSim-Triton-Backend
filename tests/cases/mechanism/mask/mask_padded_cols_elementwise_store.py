"""An elementwise op on 96 columns riding a block of 128, loaded and stored under the same column
mask: the store takes only 96 columns, so every buffer's in-lane axis is cut to 96."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, C, CB = 128, 96, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_padded_cols_elementwise_store_kernel(x_ptr, y_ptr, M: tl.constexpr,
                                                  C: tl.constexpr, CB: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, CB)[None, :]
        m = cols < C
        x = tl.load(x_ptr + rows * C + cols, m, other=0.0)
        tl.store(y_ptr + rows * C + cols, x * 2.0 + 1.0, m)

    return mask_padded_cols_elementwise_store_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * C, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"y": inputs["x"] * 2.0 + 1.0}


SPEC = KernelSpec(
    name="mask_padded_cols_elementwise_store_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32",
               "M": "constexpr", "C": "constexpr", "CB": "constexpr"},
    constexprs={"M": M, "C": C, "CB": CB},
    args=[Arg("x", IN, "float32", (M * C,)), Arg("y", OUT, "float32", (M * C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=0, atol=0,
)
