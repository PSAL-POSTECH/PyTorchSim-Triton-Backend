"""A softmax over 48 of a block of 64 columns, its one mask read by the max, the sum, the load and
the store, as Inductor writes it. pytorchsim-fuse makes the mask in each reader's body, so
pytorchsim-fit-to-numel cuts every tile to the 48 columns; the padding never reaches a reduce."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, C, CB = 128, 48, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_shared_by_two_reduces_kernel(x_ptr, y_ptr, M: tl.constexpr, C: tl.constexpr,
                                          CB: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, CB)[None, :]
        mask = cols < C
        x = tl.load(x_ptr + rows * C + cols, mask, other=0.0)
        m = tl.max(tl.where(mask, x, float("-inf")), 1)[:, None]
        e = tl.exp(x - m)
        s = tl.sum(tl.where(mask, e, 0.0), 1)[:, None]
        tl.store(y_ptr + rows * C + cols, e / s, mask)

    return mask_shared_by_two_reduces_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * C, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"y": inputs["x"].reshape(M, C).softmax(dim=1).reshape(-1)}


SPEC = KernelSpec(
    name="mask_shared_by_two_reduces_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32",
               "M": "constexpr", "C": "constexpr", "CB": "constexpr"},
    constexprs={"M": M, "C": C, "CB": CB},
    args=[Arg("x", IN, "float32", (M * C,)), Arg("y", OUT, "float32", (M * C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-5,
)
