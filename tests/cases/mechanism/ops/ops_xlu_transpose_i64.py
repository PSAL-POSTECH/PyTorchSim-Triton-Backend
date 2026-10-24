"""A tile that has to change banks while its element is WIDER than the queues.

WHAT IT PINS. The cross-lane unit carries 32 bits. A push hands raw bits over at
the current SEW and `e64` is not one of the cases spike's push writes, so a
64-bit element would arrive as 0.0f -- which is why every operation but the
all-gather refused one by name. This is the transpose's half of opening that: a
rank-1 i64 row of 32 is regrouped to (2, 16), so every element changes lane and
the unit, not a DRAM round trip, has to move them.

    before   torchsim-lower-to-vcix: "its element is 64 bits and only 32 crosses
             whole", and the move falls back to a round trip it cannot take
    after    the move crosses twice -- low halves, then high halves, the same
             pattern each time -- ALL STAGES PASSED, max_abs_err 0

THE MOVE IS A `permute` AND NOT A `transpose`. A reshape RENUMBERS, and the
unit's transpose states one correspondence only -- (lane L, depth d) to (lane d,
depth L). The pattern states any of them, so a regrouped lane axis is what the
permute is for; the transpose carries a re-banking that renumbers nothing.

WHY THE LOOP. ops_reshape_splits_the_lane_axis measured this by leave-one-out:
written without the `scf.for` the reshape is ABSORBED in absorb_layout_ops and
never reaches the pass that prices it. A layout op on a loop RESULT cannot be
pushed back into its producer -- init, iter_arg, yield operand and result are ONE
type and a walk from outside the region reaches only two of the four.

WHY NOT A REDUCE OVER THE WIDE TILE. A reduce is the one operation the halves do
not answer: it COMBINES rather than permutes, so the two halves are not two
independent problems and a carry crosses between them. `_crossings_of` says so
by name, and the `tl.sum` here runs in-lane over the folded tile, which needs no
trip at all.

torch equivalent: x.reshape(2, 16).sum(1) after three scaled accumulations.

A regrouped lane axis at i64: pins the cross-lane permute carrying a wide element as its two 32-bit halves.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, ROWS, STEPS = 32, 2, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_transpose_i64_kernel(x_ptr, s_ptr, N: tl.constexpr,
                                     ROWS: tl.constexpr, STEPS: tl.constexpr):
        offs = tl.arange(0, N)
        v = tl.zeros((N,), dtype=tl.int64)
        for i in range(0, STEPS):
            v = v + tl.load(x_ptr + offs) * (i + 1)
        m = tl.reshape(v, (ROWS, N // ROWS))
        tl.store(s_ptr + tl.arange(0, ROWS), tl.sum(m, axis=1))

    return ops_xlu_transpose_i64_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(-1000, 1000, (N,), generator=g,
                               dtype=torch.int64)}


def reference(inputs):
    v = inputs["x"] * sum(i + 1 for i in range(STEPS))
    return {"s": v.reshape(ROWS, N // ROWS).sum(dim=1)}


SPEC = KernelSpec(
    name="ops_xlu_transpose_i64_kernel", kernel=kernel,
    signature={"x_ptr": "*i64", "s_ptr": "*i64", "N": "constexpr",
               "ROWS": "constexpr", "STEPS": "constexpr"},
    constexprs={"N": N, "ROWS": ROWS, "STEPS": STEPS},
    args=[Arg("x", IN, "int64", (N,)), Arg("s", OUT, "int64", (ROWS,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE MOVE IS THE CLAIM. Without it the reshape is absorbed or bought with a
    # round trip and the width never reaches the unit at all.
    expect_ir={"08-transformed.mlir": {"pytorchsim.crosslane = 16 : i64": 1}},
)
