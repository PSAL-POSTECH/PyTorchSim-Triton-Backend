"""A row sum whose padding is NOT its identity: past the 384 columns the loads fill with 1.0, so
the sum over the block of 512 counts 128 ones a row. pytorchsim-drop-unread-padding must leave that axis
whole -- cutting it would drop the ones."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, C, CB = 128, 384, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_padded_sum_of_ones_past_the_mask_kernel(x_ptr, y_ptr, M: tl.constexpr,
                                                     C: tl.constexpr, CB: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, CB)[None, :]
        x = tl.load(x_ptr + rows * C + cols, cols < C, other=1.0)
        tl.store(y_ptr + rows, tl.sum(x, 1)[:, None])

    return mask_padded_sum_of_ones_past_the_mask_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * C, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"y": inputs["x"].reshape(M, C).sum(dim=1) + float(CB - C)}


SPEC = KernelSpec(
    name="mask_padded_sum_of_ones_past_the_mask_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32",
               "M": "constexpr", "C": "constexpr", "CB": "constexpr"},
    constexprs={"M": M, "C": C, "CB": CB},
    args=[Arg("x", IN, "float32", (M * C,)), Arg("y", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-3,
)
