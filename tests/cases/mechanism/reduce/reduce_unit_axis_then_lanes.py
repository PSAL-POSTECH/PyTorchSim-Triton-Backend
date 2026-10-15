"""A [128, 8, 1] tile folded over its unit axis, then over the lanes: lowering leaves an llvm.extractelement of a
vector<1xf32> lane value, which TO_LLVM refuses (lower stage, measured on 6ebafec); the [128, 8] tile without the
unit axis compiles and matches. validation/vector writes its C = 128 softmax without the axis until this passes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R = 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_unit_axis_then_lanes_kernel(x_ptr, o_ptr):
        q = tl.arange(0, 128)
        r = tl.arange(0, 8)
        k = tl.arange(0, 1)
        x = tl.load(x_ptr + r[None, :, None] * 128 + q[:, None, None] + k[None, None, :])
        tl.store(o_ptr + r, tl.max(tl.max(x, axis=2), axis=0))

    return reduce_unit_axis_then_lanes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R * 128, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].reshape(R, 128).max(dim=1).values}


SPEC = KernelSpec(
    name="reduce_unit_axis_then_lanes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32"}, constexprs={},
    args=[Arg("x", IN, "float32", (R * 128,)), Arg("o", OUT, "float32", (R,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="blocked",
    expect_why="a fold over a unit axis before the lanes leaves an extractelement of a vector<1> lane value",
)
