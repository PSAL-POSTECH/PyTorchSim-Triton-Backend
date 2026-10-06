"""A unit axis is a NAME, so absorbing it can never refuse -- and it used to.

WHAT THIS PINS. `absorb_layout_ops` reads a layout op as a weight per result axis
and then asks one question: does each old axis land on exactly one new axis with
the same weight and the same extent? If it does, restating any per-axis
statement is a reorder and nothing can say no. That question was asked of two
shapes of the SAME RANK and answered "no" the moment the ranks differed, which
sent every expand and collapse that only adds or drops an extent-1 axis into the
split/merge branch -- the branch that has a reading for a stride and for nothing
else, so a `linalg.generic`'s map, a `linalg.reduce`'s dimensions and a slice's
three arrays all refused it.

An axis of extent 1 has exactly one in-bounds index. It carries no element and
steps nothing, so it appearing on one side only is a change of NAME: the axis
that goes away takes nothing with it, and the axis that appears is described by
the neutral entry of whatever statement is being restated -- extent 1, offset 0,
the map result `0`, the stride that leaves its neighbours a single run.

MEASURED, on the tree before this: 33 of the 42 expand/collapse ops surviving
absorb_layout_ops across the suite differed from their operand in extent-1 axes only, and this
is the smallest of them.

    tensor<8x1xf32>  ->  tensor<8xf32>       collapse_shape [[0, 1]]

WHAT IT FAILS WITH WITHOUT IT. `SPEC.extra["forbids"]` turns absorb_layout_ops's own
declaration on for this kernel, so the pass is stopped at the moment it leaves
the op standing rather than five passes down:

    absorb_layout_ops left 1 op it declared would not survive it:
      tensor.collapse_shape x1

WHERE IT GOES. Backwards, into the MVIN that staged the tile: the descriptor
drops the row that indexed the unit axis and the tile is staged `memref<8xf32>`
instead of `memref<8x1xf32>`. That direction is not a coin toss -- absorb_layout_ops ranks the
two plans by how much RANK each adds, and the one that DELETES the axis wins
(rule 8b: unit-axis copies are the number that must reach zero; rule 14: the
passes below have fewer choices than this one and a phantom axis is a constraint
they cannot answer).

WHAT WOULD BREAK IT: reading the ranks instead of the extents in `_bijection`;
restating the descriptor without dropping the entry for the axis that left;
ranking the two plans by reach alone, which stages `memref<8x1xf32>` instead and
hands fit_layout_to_lanes an axis the program does not have.

torch equivalent: (x.reshape(N, 1).reshape(N) + w)

Pins the exact branch answering for a rank change made only of extent-1 axes. Fails on the tree before with `absorb_layout_ops left 1 op it declared would not survive it: tensor.collapse_shape x1`.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 8
D = 2           # the DRAM stride of the [N, 1] window. D > 1 is what stops the
                # backward plan being a MERGE that happens to succeed, so the
                # only reading left is the exact one this kernel pins.


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_absorb_unit_axis_kernel(x_ptr, w_ptr, o_ptr, N: tl.constexpr,
                                    D: tl.constexpr):
        r = tl.arange(0, N)
        # [N, 1] -- the unit axis is real in the load, so the tile is staged
        # with it and the reshape below is the only thing that removes it.
        x = tl.load(x_ptr + r[:, None] * D)
        y = tl.reshape(x, (N,))
        tl.store(o_ptr + r, y + tl.load(w_ptr + r))

    return ops_absorb_unit_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N * D, generator=g, dtype=torch.float32),
            "w": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": (inputs["x"][::D] + inputs["w"]).contiguous()}


SPEC = KernelSpec(
    name="ops_absorb_unit_axis_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "N": "constexpr", "D": "constexpr"},
    constexprs={"N": N, "D": D},
    args=[Arg("x", IN, "float32", (N * D,)),
          Arg("w", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=0, atol=0,
    extra={"forbids": {"psto-absorb-layout-ops": True}},
)
