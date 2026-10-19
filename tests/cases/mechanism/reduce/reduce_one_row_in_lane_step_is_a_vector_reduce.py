"""A max over one row of 8192: each lane holds 64 of the row and maxes them before the lanes fold. The
in-lane step reads views without the trailing unit dims, so it is one vector reduce a lane, not 64
steps of one element. Breaks if the unit dim is walked again.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C = 8192


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_one_row_in_lane_step_is_a_vector_reduce_kernel(x_ptr, o_ptr, C: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, C)[None, :])
        tl.store(o_ptr + tl.arange(0, 1), tl.max(x, axis=1))

    return reduce_one_row_in_lane_step_is_a_vector_reduce_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(1, C, generator=g)}


def reference(inputs):
    x = inputs["x"]
    return {"o": x.amax(1)}


SPEC = KernelSpec(
    name="reduce_one_row_in_lane_step_is_a_vector_reduce_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "C": "constexpr"},
    constexprs={"C": C},
    args=[Arg("x", IN, "float32", (1, C)), Arg("o", OUT, "float32", (1,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    # THE IN-LANE STEP IS THE CLAIM: one vector reduce over a lane's 64 elements.
    expect_ir={"08-transformed.mlir": {"vector.multi_reduction": 1}},
)
