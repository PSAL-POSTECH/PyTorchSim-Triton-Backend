"""8 rows of 1024 less their row max: the columns go on the lanes, each lane holds its 8 laps of the
8 rows, and the fold leaves the 8 maxes in every lane. The subtract walks a lane's 64 elements as one
vector, the 8 maxes repeated: they are read once and repeated in the register, no buffer is filled
with them. Breaks if the fold's result is spread through a buffer again (8 copies of a row of 8), or
if its readers walk a lap at a time.
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
    def reduce_fold_result_held_in_registers_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr):
        at = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
        x = tl.load(x_ptr + at)
        tl.store(o_ptr + at, x - tl.max(x, axis=1)[:, None])

    return reduce_fold_result_held_in_registers_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R, C, generator=g)}


def reference(inputs):
    x = inputs["x"]
    return {"o": x - x.amax(1, keepdim=True)}


SPEC = KernelSpec(
    name="reduce_fold_result_held_in_registers_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (R, C)), Arg("o", OUT, "float32", (R, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    # THE REGISTER IS THE CLAIM: one fold, its 8 maxes joined into the 64 the subtract walks, and
    # no transfer writes them anywhere.
    expect_ir={"08-transformed.mlir": {"pytorchsim.distributed_reduction": 1, "vector.shuffle": 1,
                                       "vector.transfer_write": 0, "vector<64xf32>": 3}},
)
