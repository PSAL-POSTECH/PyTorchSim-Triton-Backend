"""A kernel states a tile's layout with tl.extra.torchsim.constrain: x * 2 + 1 on a 128x128 tile,
whose default layout puts the lanes on its rows, constrained to the lanes across its columns before
the store. The anchor pins the tile there and the layouts around it meet it. torch: x * 2 + 1."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 128


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import torchsim

    @triton.jit
    def constrain_columns_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        y = tl.load(x_ptr + offs) * 2.0 + 1.0
        y = torchsim.constrain(y, torchsim.lane_layout(1))
        tl.store(o_ptr + offs, y)

    return constrain_columns_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"] * 2 + 1}


SPEC = KernelSpec(
    name="constrain_columns_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
