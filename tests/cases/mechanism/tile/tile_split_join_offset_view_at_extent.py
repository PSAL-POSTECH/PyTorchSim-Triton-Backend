"""A [128, 2] bf16 tile split into its columns and joined back swapped (o = x.flip(1)); the second
column is a [128, 1] view at offset 1. Breaks if legalize widens its one-element loop past the extent:
the vector<2xbf16> reaches the next row, and the containment check refuses bytes [2, 6) of a 4-byte slot."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_split_join_offset_view_at_extent_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                                     N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        idx = rm[:, None] * N + rn[None, :]
        x = tl.load(x_ptr + idx)
        a, b = tl.split(tl.reshape(x, (M, 1, 2)))
        y = tl.reshape(tl.join(b, a), (M, N))
        tl.store(o_ptr + idx, y)

    return tile_split_join_offset_view_at_extent_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.bfloat16)}


def reference(inputs):
    return {"o": inputs["x"].flip(1)}


SPEC = KernelSpec(
    name="tile_split_join_offset_view_at_extent_kernel", kernel=kernel,
    signature={"x_ptr": "*bf16", "o_ptr": "*bf16", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "bfloat16", (M, N)), Arg("o", OUT, "bfloat16", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
