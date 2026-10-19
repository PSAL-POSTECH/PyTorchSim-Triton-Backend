"""A max over 4 x 8192 elements, the way Inductor writes a large reduce: a [1, 8192] accumulator maxed
with each block in a loop, then reduced. The accumulator is loaded once before the loop, carried as
an iter_arg and stored once after, not read and written back around every block's max. Breaks if the
loop's accumulator goes back to the scratchpad each trip.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, B = 4 * 8192, 8192


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_chunked_accumulator_in_registers_kernel(x_ptr, o_ptr, N: tl.constexpr, B: tl.constexpr):
        r = tl.arange(0, B)[None, :]
        acc = tl.full((1, B), float("-inf"), tl.float32)
        for off in range(0, N, B):
            acc = tl.maximum(acc, tl.load(x_ptr + off + r))
        tl.store(o_ptr + tl.arange(0, 1)[:, None], tl.max(acc, axis=1)[:, None])

    return reduce_chunked_accumulator_in_registers_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g)}


def reference(inputs):
    x = inputs["x"]
    return {"o": x.amax().reshape(1, 1)}


SPEC = KernelSpec(
    name="reduce_chunked_accumulator_in_registers_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr", "B": "constexpr"},
    constexprs={"N": N, "B": B},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (1, 1))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    # THE CARRY IS THE CLAIM: the block loop carries the accumulator (a second iter_args, one scf.yield).
    expect_ir={"08-transformed.mlir": {"iter_args": 2, "scf.yield": 1}},
)
