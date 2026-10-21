"""A loaded scalar's exp, broadcast over a [128,16] tile and added: lanes 1-127 read the scalar's one-bank
buffer from their own banks. Breaks (127 of 128 lanes wrong) if a one-bank MVIN fills lane 0's bank
alone; while the reader is a lane-0 scalar load (T2), before vector<1> readers, it passes either way."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
N, BM = 16, 128
def kernel():
    import triton
    import triton.language as tl
    @triton.jit
    def reduce_scalar_exp_store_kernel(x_ptr, s_ptr, o_ptr, N: tl.constexpr, BM: tl.constexpr):
        rows = tl.arange(0, BM)
        cols = tl.arange(0, N)
        s = tl.load(s_ptr)
        e = tl.exp(s) * 0.5
        x = tl.load(x_ptr + rows[:, None] * N + cols[None, :])
        tl.store(o_ptr + rows[:, None] * N + cols[None, :], x + e)
    return reduce_scalar_exp_store_kernel
def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BM * N, generator=g), "s": torch.tensor([0.75])}
def reference(inputs):
    import torch
    return {"o": inputs["x"] + torch.exp(inputs["s"][0]) * 0.5}
SPEC = KernelSpec(name="reduce_scalar_exp_store_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "s_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr", "BM": "constexpr"},
    constexprs={"N": N, "BM": BM},
    args=[Arg("x", IN, "float32", (BM * N,)), Arg("s", IN, "float32", (1,)), Arg("o", OUT, "float32", (BM * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4, expect="pass")
