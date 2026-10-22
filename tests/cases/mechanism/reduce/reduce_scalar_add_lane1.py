"""A loaded scalar added to a [128] row on 128 lanes, one element a lane: lanes 1-127 read the scalar's
one-bank buffer from their own banks. Breaks (127 of 128 lanes wrong) if a one-bank MVIN fills lane 0's
bank alone; while the reader is a lane-0 scalar load (T2), before vector<1> readers, it passes either way."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
BM = 128
def kernel():
    import triton
    import triton.language as tl
    @triton.jit
    def reduce_scalar_add_lane1_kernel(x_ptr, s_ptr, o_ptr, BM: tl.constexpr):
        rows = tl.arange(0, BM)
        s = tl.load(s_ptr)
        x = tl.load(x_ptr + rows)
        tl.store(o_ptr + rows, x * s + s)
    return reduce_scalar_add_lane1_kernel
def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BM, generator=g), "s": torch.tensor([0.75])}
def reference(inputs):
    s = inputs["s"][0]
    return {"o": inputs["x"] * s + s}
SPEC = KernelSpec(name="reduce_scalar_add_lane1_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "s_ptr": "*fp32", "o_ptr": "*fp32", "BM": "constexpr"},
    constexprs={"BM": BM},
    args=[Arg("x", IN, "float32", (BM,)), Arg("s", IN, "float32", (1,)), Arg("o", OUT, "float32", (BM,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-5, atol=1e-5, expect="pass")
