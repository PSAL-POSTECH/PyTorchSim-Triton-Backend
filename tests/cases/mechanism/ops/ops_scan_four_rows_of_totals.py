"""A scan four times the lanes long: folded to [4, 256], its row totals are cut from the last
column -- a view at bank depth 255 -- and shifted along the lanes by the XLU. Breaks if an XLU
end is read from its bank's start: rows 2 and 3 then add the wrong running total."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_scan_four_rows_of_totals_kernel(x_ptr, o_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        tl.store(o_ptr + r, tl.cumsum(tl.load(x_ptr + r), axis=0))

    return ops_scan_four_rows_of_totals_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.rand(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].double().cumsum(0).float()}


SPEC = KernelSpec(
    name="ops_scan_four_rows_of_totals_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
