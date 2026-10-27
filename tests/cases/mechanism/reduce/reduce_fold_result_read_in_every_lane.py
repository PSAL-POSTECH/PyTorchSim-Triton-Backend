"""A max over one row of 2048 on the lanes, its result read by every element: the fold's pop gives every
lane the max and each lane copies it into its own bank, the one row repeated over the 16 laps by one
load of stride 0. Breaks if the broadcast of a fold's result goes back to a move through the
cross-lane unit (pytorchsim_tpu.xlu_move), or is written a row at a time.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 2048


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_fold_result_read_in_every_lane_kernel(x_ptr, o_ptr, N: tl.constexpr):
        r = tl.arange(0, N)[None, :]
        x = tl.load(x_ptr + r)
        tl.store(o_ptr + r, x - tl.max(x, axis=1)[:, None])

    return reduce_fold_result_read_in_every_lane_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(1, N, generator=g)}


def reference(inputs):
    x = inputs["x"]
    return {"o": x - x.amax(1, keepdim=True)}


SPEC = KernelSpec(
    name="reduce_fold_result_read_in_every_lane_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (1, N)), Arg("o", OUT, "float32", (1, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    # THE COPY IS THE CLAIM: one fold, no cross-lane move, and no row written as a vector of one.
    expect_ir={"08-transformed.mlir": {"pytorchsim.distributed_reduction": 1, "pytorchsim_tpu.xlu_move": 0,
                               "vector<1xf32>, memref": 0}},
)
