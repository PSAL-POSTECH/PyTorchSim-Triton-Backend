"""A row max over 384 columns riding a block of 512: the loads fill past the columns with -inf,
the max's identity, so the in-lane axis is cut to 384 and the max reads only the data."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, C, CB = 128, 384, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_padded_row_max_kernel(x_ptr, y_ptr, M: tl.constexpr, C: tl.constexpr,
                                   CB: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, CB)[None, :]
        x = tl.load(x_ptr + rows * C + cols, cols < C, other=float("-inf"))
        tl.store(y_ptr + rows, tl.max(x, 1)[:, None])

    return mask_padded_row_max_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * C, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"y": inputs["x"].reshape(M, C).amax(dim=1)}


SPEC = KernelSpec(
    name="mask_padded_row_max_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32",
               "M": "constexpr", "C": "constexpr", "CB": "constexpr"},
    constexprs={"M": M, "C": C, "CB": CB},
    args=[Arg("x", IN, "float32", (M * C,)), Arg("y", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=0, atol=0,
)
