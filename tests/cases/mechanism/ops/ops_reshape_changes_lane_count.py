"""A loop-carried 16x16 i64 tile reshaped to 32x8, 16 lanes to 8, moves nothing: a reshape renames
coordinates, and its result is laid out as its source composed through it.
Breaks if a pass below TritonGPU lays the result out on its own again: a cross-lane move appears."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS, COLS, DEEP, LANES, STEPS = 16, 16, 32, 8, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_reshape_changes_lane_count_kernel(
            x_ptr, w_ptr, o_ptr, ROWS: tl.constexpr, COLS: tl.constexpr,
            DEEP: tl.constexpr, LANES: tl.constexpr, STEPS: tl.constexpr):
        off = tl.arange(0, ROWS)[:, None] * COLS + tl.arange(0, COLS)[None, :]
        v = tl.zeros((ROWS, COLS), dtype=tl.int64)
        for i in range(0, STEPS):
            v = v + tl.load(x_ptr + off) * (i + 1)
        m = tl.reshape(v, (DEEP, LANES))
        out = tl.arange(0, DEEP)[:, None] * LANES + tl.arange(0, LANES)[None, :]
        tl.store(o_ptr + out, m + tl.load(w_ptr + out))

    return ops_reshape_changes_lane_count_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(-50, 50, (ROWS, COLS), generator=g,
                               dtype=torch.int64),
            "w": torch.randint(-50, 50, (DEEP, LANES), generator=g,
                               dtype=torch.int64)}


def reference(inputs):
    return {"o": (inputs["x"] * STEPS * (STEPS + 1) // 2).reshape(
        DEEP, LANES) + inputs["w"]}


SPEC = KernelSpec(
    name="ops_reshape_changes_lane_count_kernel", kernel=kernel,
    signature={"x_ptr": "*i64", "w_ptr": "*i64", "o_ptr": "*i64",
               "ROWS": "constexpr", "COLS": "constexpr", "DEEP": "constexpr",
               "LANES": "constexpr", "STEPS": "constexpr"},
    constexprs={"ROWS": ROWS, "COLS": COLS, "DEEP": DEEP, "LANES": LANES,
                "STEPS": STEPS},
    args=[Arg("x", IN, "int64", (ROWS, COLS)),
          Arg("w", IN, "int64", (DEEP, LANES)),
          Arg("o", OUT, "int64", (DEEP, LANES))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-transformed.mlir": {"pytorchsim_tpu.xlu_move": 0}},
)
