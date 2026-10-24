"""A row sum of 8 rows of 1024: 8 rows would fill 8 of 128 lanes, so the row's columns go on the lanes,
each lane sums its laps and the lanes fold. Breaks if few rows keep the rows on the lanes again.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C = 8, 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_few_rows_fold_on_lanes_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :])
        tl.store(o_ptr + tl.arange(0, R), tl.sum(x, axis=1))

    return reduce_few_rows_fold_on_lanes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R, C, generator=g)}


def reference(inputs):
    x = inputs["x"]
    return {"o": x.sum(1)}


SPEC = KernelSpec(
    name="reduce_few_rows_fold_on_lanes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (R, C)), Arg("o", OUT, "float32", (R,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    # THE LAYOUT IS THE CLAIM: the columns on the lanes make the row sum a fold (rows on the lanes fold nothing).
    expect_ir={"08-transformed.mlir": {"pytorchsim.move_kind = \"fold\"": 1}},
)
