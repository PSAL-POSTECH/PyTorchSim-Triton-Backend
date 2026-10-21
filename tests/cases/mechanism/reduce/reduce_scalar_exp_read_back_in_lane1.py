"""exp of a loaded scalar stored into a one-element buffer, read back and added to a [128] row: lanes 1-127
read that buffer from their own banks. Breaks (127 of 128 lanes wrong) if a one-bank MVIN fills lane 0's
bank alone; while the reader is a lane-0 scalar load (T2), before vector<1> readers, it passes either way."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
BM = 128
def kernel():
    import triton
    import triton.language as tl
    @triton.jit
    def reduce_scalar_exp_read_back_in_lane1_kernel(s_ptr, x_ptr, t_ptr, o_ptr, BM: tl.constexpr):
        r = tl.arange(0, BM)
        s = tl.load(s_ptr)
        e = tl.exp(s)
        tl.store(t_ptr, e)
        x = tl.load(x_ptr + r)
        tl.store(o_ptr + r, x + e)
    return reduce_scalar_exp_read_back_in_lane1_kernel
def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"s": torch.tensor([0.75]), "x": torch.randn(BM, generator=g)}
def reference(inputs):
    e = inputs["s"].exp()
    return {"t": e, "o": inputs["x"] + e[0]}
SPEC = KernelSpec(name="reduce_scalar_exp_read_back_in_lane1_kernel", kernel=kernel,
    signature={"s_ptr": "*fp32", "x_ptr": "*fp32", "t_ptr": "*fp32", "o_ptr": "*fp32", "BM": "constexpr"},
    constexprs={"BM": BM},
    args=[Arg("s", IN, "float32", (1,)), Arg("x", IN, "float32", (BM,)), Arg("t", OUT, "float32", (1,)), Arg("o", OUT, "float32", (BM,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-5, atol=1e-5, expect="pass")
