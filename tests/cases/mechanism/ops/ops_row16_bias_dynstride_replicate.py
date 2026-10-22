"""A [16] row loaded with a runtime stride, so into its own UNIFORM [1,16] buffer, added to every row
of a [128,16] tile. Breaks (127 of 128 lanes wrong) if the UNIFORM MVIN fills lane 0 alone; before,
the broadcast of that row went through the cross-lane unit as a replicate move."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
BM = 128; BN = 16; SB = 2
def kernel():
    import triton
    import triton.language as tl
    @triton.jit
    def ops_row16_bias_dynstride_replicate_kernel(x_ptr, b_ptr, o_ptr, sb, BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        b = tl.load(b_ptr + rn * sb)
        x = tl.load(x_ptr + rm[:, None] * BN + rn[None, :])
        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], x + b[None, :])
    return ops_row16_bias_dynstride_replicate_kernel
def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BM * BN, generator=g), "b": torch.randn(BN * SB, generator=g) + 3.0}
def reference(inputs):
    return {"o": (inputs["x"].reshape(BM, BN) + inputs["b"][::SB][None, :]).reshape(-1)}
SPEC = KernelSpec(name="ops_row16_bias_dynstride_replicate_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32", "sb": "i32", "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (BM * BN,)), Arg("b", IN, "float32", (BN * SB,)), Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-5, atol=1e-5,
    extra={"scalar_args": [("sb", "int32_t")], "scalar_values": {"sb": SB}}, expect="pass")
