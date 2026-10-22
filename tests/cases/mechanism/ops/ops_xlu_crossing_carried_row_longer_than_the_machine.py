"""A loop-carried 1024-long row reshaped to (2, 512) and summed per row: the sum keeps the row's
lanes and folds them, ONE cross-lane move (a fold) after each lane sums its four laps. Breaks if the
reshape stops on its own (lanes over two dims) as a second move."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, ROWS, STEPS = 1024, 2, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_crossing_carried_row_longer_than_the_machine_kernel(
            x_ptr, s_ptr, N: tl.constexpr, ROWS: tl.constexpr, STEPS: tl.constexpr):
        offs = tl.arange(0, N)
        v = tl.zeros((N,), dtype=tl.float32)
        for i in range(0, STEPS):
            v = v + tl.load(x_ptr + offs) * (i + 1)
        m = tl.reshape(v, (ROWS, N // ROWS))
        tl.store(s_ptr + tl.arange(0, ROWS), tl.sum(m, axis=1))

    return ops_xlu_crossing_carried_row_longer_than_the_machine_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    v = inputs["x"] * sum(i + 1 for i in range(STEPS))
    return {"s": v.reshape(ROWS, N // ROWS).sum(dim=1)}


SPEC = KernelSpec(
    name="ops_xlu_crossing_carried_row_longer_than_the_machine_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "s_ptr": "*fp32", "N": "constexpr", "ROWS": "constexpr",
               "STEPS": "constexpr"},
    constexprs={"N": N, "ROWS": ROWS, "STEPS": STEPS},
    args=[Arg("x", IN, "float32", (N,)), Arg("s", OUT, "float32", (ROWS,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # ONE CROSSING AND NO ROUND TRIP.
    expect_ir={"08-transformed.mlir": {"torchsim.crosslane_move": 1,
                                       "torchsim.move_kind = \"fold\"": 1}},
)
