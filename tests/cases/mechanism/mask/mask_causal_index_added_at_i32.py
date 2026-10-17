"""A causal mask over a 128x128 fp32 tile: the column index n = start + arange is added in a loop
and compared, as i32, with the row index. The integer range analysis proves n fits i32, the
narrowest integer width the function already has, so the add runs at i32 and not at the index
width (XLEN) only to be cast down for the compare -- one e64 vtype fewer in the kernel. Pins the
three index_castui the narrowing leaves in 08-transformed and the one e64,m8 vsetvli left in
11-kernel.s. torch: x.masked_fill(triu(ones, 1), -1e30)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_causal_index_added_at_i32(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
        tl.store(o_ptr + rm[:, None] * N + rn[None, :],
                 tl.where(rm[:, None] >= rn[None, :], x, -1.0e30))

    return mask_causal_index_added_at_i32


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": inputs["x"].masked_fill(torch.ones(M, N, dtype=torch.bool).triu(1), -1.0e30)}


SPEC = KernelSpec(
    name="mask_causal_index_added_at_i32", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=0, atol=0,
    expect_ir={"08-transformed.mlir": {"arith.index_castui": 3},
               "11-kernel.s": {"e64, m8": 1}},
)
