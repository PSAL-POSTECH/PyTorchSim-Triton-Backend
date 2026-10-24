"""ops_reshape_collapses_the_lane_axis at i64: (2, 16) -> (32,) on a loop result moves nothing, and
the one cross-lane move, the sum's conversion, carries the wide element as two halves.
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
    def ops_xlu_permute_widens_i64_kernel(x_ptr, y_ptr, s_ptr, ROWS: tl.constexpr,
                                          COLS: tl.constexpr,
                                          STEPS: tl.constexpr):
        offs = tl.arange(0, ROWS)[:, None] * COLS + tl.arange(0, COLS)[None, :]
        v = tl.zeros((ROWS, COLS), dtype=tl.int64)
        for i in range(0, STEPS):
            v = v + tl.load(x_ptr + offs) * (i + 1)
        tl.store(s_ptr + tl.arange(0, COLS), tl.sum(v, axis=0))
        f = tl.reshape(v, (ROWS * COLS,))
        k = tl.arange(0, ROWS * COLS)
        tl.store(y_ptr + k, f * (k + 1).to(tl.int64))

    return ops_xlu_permute_widens_i64_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(-1000, 1000, (N,), generator=g,
                               dtype=torch.int64)}


def reference(inputs):
    import torch
    v = inputs["x"].reshape(ROWS, COLS) * sum(i + 1 for i in range(STEPS))
    k = torch.arange(N, dtype=torch.int64) + 1
    return {"y": v.reshape(N) * k, "s": v.sum(dim=0)}


SPEC = KernelSpec(
    name="ops_xlu_permute_widens_i64_kernel", kernel=kernel,
    signature={"x_ptr": "*i64", "y_ptr": "*i64", "s_ptr": "*i64", "ROWS": "constexpr",
               "COLS": "constexpr", "STEPS": "constexpr"},
    constexprs={"ROWS": ROWS, "COLS": COLS, "STEPS": STEPS},
    args=[Arg("x", IN, "int64", (N,)), Arg("y", OUT, "int64", (N,)),
          Arg("s", OUT, "int64", (COLS,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE PATH IS THE CLAIM: a crossing for the sum's conversion, nothing for the collapse.
    expect_ir={"08-transformed.mlir": {"pytorchsim.crosslane = 16 : i64": 0,
                                       "pytorchsim.crosslane = 4 : i64": 1}},
)
