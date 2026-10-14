"""A free backward absorption that leaves fit_layout_to_lanes a conflict it will not repair.

WHAT THIS FOUND. absorb_layout_ops's backward arm folds `tl.trans(x)` into the MVIN
that staged `x` -- one `dram_stride`, no data moved, exactly what rule 8 asks
for. The transposed tile then feeds TWO things that want opposite lane axes: a
`linalg.matmul` contracting one axis, and a `linalg.reduce` folding the other.
vectorize refuses:

    reduce over dim 1 of a tile banked on that same axis would have to fold
    across lanes: linalg.reduce ins(%4 : memref<8x32xf32, 1>)
    outs(%2 : memref<8xf32, 1>) dimensions = [1]

BEFORE THE BACKWARD ARM THIS COMPILED, and that is the whole point of the file.
The transpose survived absorb_layout_ops, fit_layout_to_lanes turned it into a DRAM
round trip, and the round trip happened to land the two readers on axes they
could both live with. So the repair existed -- it was just being reached by
accident, as a side effect of a layout op that rule 8 says must not be standing
there at all.

WHY THIS IS NOT AN ARGUMENT FOR PUTTING THE TRANSPOSE BACK. The trip was free to
remove and rule 8b says a trip a free absorption would have avoided is
absorption failure, not cost. What is missing is the repair itself, on the
buffer, decided where the cost is visible -- fit_layout_to_lanes owns that and
`_movable` skips every buffer holding a class some MUST names, which here is
this one, because the matmul's MUST names it. That is the SAME gap dl/qk_norm
sits on, written down in absorb_layout_ops's docstring, and this file is the small
form of it.

WHAT WOULD MOVE IT. A relayout offered on a buffer a MUST has claimed, chosen by
cost rather than by whether a layout op happens to be sitting in the way. Then
this passes and the marker comes off.

EXPECT="PASS" AND IT PASSES, SINCE 2026-09-04. It reached this file as a red
reproducer -- the repair was only ever arrived at by accident, as a side effect
of a layout op rule 8 says must not be standing there -- and both halves have
now landed. select_lane_axis reaches the repair on purpose (conflicts are part
of the turn's yardstick), and the wrong answer that exposed was not this
kernel's design at all.

WHAT THE WRONG ANSWER WAS, AND WHERE THE FIRST READING WENT. With the repair
reached, err 16.1495 on 128 of 128 elements, and every piece of the transposed
form checked out by hand: both stagings (32x8 stride [8,1] is x, 16x32 stride
[1,16] is y.t()), the flipped matmul (y.t()@x = out.t()), the reduce (dims=[0]
of 32x8 is the same sum as dims=[1] of 8x32), the broadcast, and the transposed
store. SPIKE_DEBUG showed lane j of eight holding column j of B, right lane and
right values, but only SIXTEEN of the thirty-two rows K has, at in-bank
positions 0..7 and 16..23 -- eight on, eight off.

That was read as "the MVIN for B moves half its tile", and it was not. The MVIN
is correct: p_dim_size (1,1,32,8), block_stride H->1, used_vlane 8, so it writes
all 32 in-bank slots of 8 lanes. What was half was the READ. lower_to_vcix
pushed with a `vector.transfer_read` of vlen/elen = 16 elements indexed on the
tile, whose last axis here is 8 -- the LANE COUNT -- so each push was clamped at
8 and padded the rest, twice, giving exactly 0..7 and 16..23. See
mechanism/ops/ops_trans_matmul_maps, which is where that is pinned.

The lesson worth the file: an eight-on/eight-off pattern in a bank names a
WIDTH, and there are two widths in reach -- the DMA's and the vector's. This
one blamed the writer and it was the reader.

torch equivalent: (x.t() @ y) + x.t().sum(1)[:, None].

Absorbing the transpose is free and correct; the two readers of the absorbed tile then need a relayout on a buffer a matmul MUST has claimed, which fit_layout_to_lanes does not offer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 32, 8, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def trans_absorbed_then_no_repair_kernel(x_ptr, y_ptr, o_ptr,
                                             M: tl.constexpr, K: tl.constexpr,
                                             N: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :])       # (M, K)
        y = tl.load(y_ptr + rm[:, None] * N + rn[None, :])       # (M, N)
        xt = tl.trans(x, 1, 0)                                    # (K, M)
        o = tl.dot(xt, y)                                         # (K, N)
        s = tl.sum(xt, axis=1)                                    # (K,)
        tl.store(o_ptr + rk[:, None] * N + rn[None, :], o + s[:, None])

    return trans_absorbed_then_no_repair_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * K, generator=g, dtype=torch.float32),
            "y": torch.randn(M * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(M, K)
    y = inputs["y"].reshape(M, N)
    xt = x.t()
    return {"o": (xt @ y + xt.sum(1)[:, None]).reshape(-1)}


SPEC = KernelSpec(
    name="trans_absorbed_then_no_repair_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "K": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "K": K, "N": N},
    args=[Arg("x", IN, "float32", (M * K,)), Arg("y", IN, "float32", (M * N,)),
          Arg("o", OUT, "float32", (K * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
    expect="blocked",
    expect_why="matmul K=32 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
