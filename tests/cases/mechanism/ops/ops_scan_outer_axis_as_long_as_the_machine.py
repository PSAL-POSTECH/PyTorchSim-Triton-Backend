"""ops_scan_outer_axis with the scanned axis exactly the lanes long: a [128, 4] tile scanned
along axis 0, the axis on the lanes. The ladder's padded tile is [192, 4], one and a half
lane-fulls each four rows deep, so the fold puts the second lane-full four rows down and the bank
reserves eight rows per lane -- two whole lane-fulls -- not the six that 192 x 4 elements spread
over 128 lanes would need. Breaks if the reserve is computed from the element count: the
permute's deeper rows then land in the next buffer.

A rank-2 scan along a lane axis as long as the machine, the padded tile deeper than one row.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_scan_outer_axis_as_long_as_the_machine_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                                          N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        idx = rm[:, None] * N + rn[None, :]
        x = tl.load(x_ptr + idx)
        tl.store(o_ptr + idx, tl.cumprod(x * 0.001 + 1.0, axis=0))

    return ops_scan_outer_axis_as_long_as_the_machine_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": (inputs["x"].double() * 0.001 + 1.0).cumprod(dim=0).float()}


SPEC = KernelSpec(
    name="ops_scan_outer_axis_as_long_as_the_machine_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
)
