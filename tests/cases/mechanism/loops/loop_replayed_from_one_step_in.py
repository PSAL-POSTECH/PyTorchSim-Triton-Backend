"""A replayed loop whose start is a whole step in: scf.for 64 to 1024 step 64, a load and an add.

The cycle binary runs every replayed loop once. It used to collapse an scf.for to ub = step, so a
loop starting at lb >= step made no trip, its tile never reached gem5, and timing refused the
kernel. With ub = lb + step the tile fires once per call. Breaks if the collapse stops giving every
replayed scf.for exactly one trip.

RUNNABLE. o = x[64:128] + x[128:192] + ... + x[960:1024].

POSITIVE.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_replayed_from_one_step_in_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                              BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(BLOCK, N, BLOCK):
            acc += tl.load(x_ptr + k + r)
        tl.store(o_ptr + r, acc)

    return loop_replayed_from_one_step_in_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"][BLOCK:].reshape(-1, BLOCK).sum(0)}


SPEC = KernelSpec(
    name="loop_replayed_from_one_step_in_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr", "BLOCK": "constexpr"},
    constexprs={"N": N, "BLOCK": BLOCK},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (BLOCK,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    expect_stage="timing",
)
