"""`tl.sort` on a row -- triton's own bitonic network, and the one Inductor emits."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def sort_row_tl_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.sort(x, dim=1))

    return sort_row_tl_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].sort(dim=1).values}


SPEC = KernelSpec(
    name="sort_row_tl_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-passes/04-pytorchsim-materialize-layouts.mlir": {"pytorchsim.convert_layout": 0}},
)
