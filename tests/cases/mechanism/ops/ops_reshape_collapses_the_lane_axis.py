"""`tl.reshape((2, 16) -> (32,))` on a loop result moves nothing: the one cross-lane move is the
sum over axis 0, a fold across the lanes; `* (k + 1)` makes a wrong order visible.
Breaks if the collapse is laid out below TritonGPU again: a permute appears."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS, COLS, STEPS = 2, 16, 3
N = ROWS * COLS


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_reshape_collapses_the_lane_axis_kernel(x_ptr, y_ptr, s_ptr,
                                                   ROWS: tl.constexpr,
                                                   COLS: tl.constexpr,
                                                   STEPS: tl.constexpr):
        offs = tl.arange(0, ROWS)[:, None] * COLS + tl.arange(0, COLS)[None, :]
        # CARRIED, so the reshape below reads a loop RESULT and cannot be
        # pushed back into it -- four slots of one type, two of them reachable.
        v = tl.zeros((ROWS, COLS), dtype=tl.float32)
        for i in range(0, STEPS):
            v = v + tl.load(x_ptr + offs) * (i + 1)
        # (2, 16) -> (32,): reassociation [[0, 1]], group 0 carries 2 AND 16.
        tl.store(s_ptr + tl.arange(0, COLS), tl.sum(v, axis=0))
        f = tl.reshape(v, (ROWS * COLS,))
        k = tl.arange(0, ROWS * COLS)
        tl.store(y_ptr + k, f * (k + 1).to(tl.float32))

    return ops_reshape_collapses_the_lane_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    v = inputs["x"].reshape(ROWS, COLS) * sum(i + 1 for i in range(STEPS))
    k = torch.arange(N, dtype=torch.float32) + 1
    return {"y": v.reshape(N) * k, "s": v.sum(dim=0)}


SPEC = KernelSpec(
    name="ops_reshape_collapses_the_lane_axis_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "s_ptr": "*fp32", "ROWS": "constexpr",
               "COLS": "constexpr", "STEPS": "constexpr"},
    constexprs={"ROWS": ROWS, "COLS": COLS, "STEPS": STEPS},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", OUT, "float32", (N,)),
          Arg("s", OUT, "float32", (COLS,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE PATH IS THE CLAIM: a crossing for the sum's conversion, nothing for the collapse.
    expect_ir={"08-transformed.mlir": {"torchsim.crosslane = 16 : i64": 0,
                                       "torchsim.move_kind = \"fold\"": 1}},
)
