"""A NaN-propagating max folded across the lanes, its one-element result stored and added to a [128] row
read in every lane. Breaks (127 of 128 lanes wrong, at poison off and 0xff alike) if a scalar store to
the scratchpad fills lane 0's bank alone; while the reader is a lane-0 scalar load it passes either way."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
BM = 128
BN = 16
def kernel():
    import triton
    import triton.language as tl
    @triton.jit
    def _maxp(a, b):
        return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)
    @triton.jit
    def reduce_fold_init_maxp_read_in_lane1_kernel(x_ptr, y_ptr, o_ptr, BM: tl.constexpr, BN: tl.constexpr):
        r = tl.arange(0, BM)
        c = tl.arange(0, BN)
        x = tl.load(x_ptr + r[:, None] * BN + c[None, :])
        m = tl.reduce(x, None, _maxp)
        y = tl.load(y_ptr + r)
        tl.store(o_ptr + r, y + m)
    return reduce_fold_init_maxp_read_in_lane1_kernel
def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BM * BN, generator=g) - 8.0, "y": torch.randn(BM, generator=g)}
def reference(inputs):
    return {"o": inputs["y"] + inputs["x"].max()}
SPEC = KernelSpec(name="reduce_fold_init_maxp_read_in_lane1_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32", "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (BM * BN,)), Arg("y", IN, "float32", (BM,)), Arg("o", OUT, "float32", (BM,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-5, atol=1e-5, expect="pass")
