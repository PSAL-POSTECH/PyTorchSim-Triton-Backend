"""A row sum over 327 rows riding a block of 512 (4 rows a lane), its columns 447 of 512. The
rows' padding cannot be cut: 327 is no whole number of rows a lane, and a buffer cut to 384 (3 a
lane) would put the masked load's partial lanes elsewhere than the ops read them. Only the
columns are cut (to 448); the sums must be the unpadded ones."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, RB, CB = 327, 447, 512, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_padded_rows_partial_lanes_kernel(x_ptr, y_ptr, R: tl.constexpr, C: tl.constexpr,
                                              RB: tl.constexpr, CB: tl.constexpr):
        rows = tl.arange(0, RB)[:, None]
        cols = tl.arange(0, CB)[None, :]
        m = (rows < R) & (cols < C)
        x = tl.load(x_ptr + rows * C + cols, m, other=0.0)
        tl.store(y_ptr + rows, tl.sum(x, 1)[:, None], rows < R)

    return mask_padded_rows_partial_lanes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R * C, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"y": inputs["x"].reshape(R, C).sum(dim=1)}


SPEC = KernelSpec(
    name="mask_padded_rows_partial_lanes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32",
               "R": "constexpr", "C": "constexpr", "RB": "constexpr", "CB": "constexpr"},
    constexprs={"R": R, "C": C, "RB": RB, "CB": CB},
    args=[Arg("x", IN, "float32", (R * C,)), Arg("y", OUT, "float32", (R,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-3,
)
