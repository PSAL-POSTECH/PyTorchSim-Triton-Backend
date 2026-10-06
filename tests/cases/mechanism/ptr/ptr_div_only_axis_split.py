"""`(x // K)` with no `%` partner is an axis split whose inner half has STRIDE
ZERO, and triton-shared performs it.

WHAT IT IS. Nearest-neighbour upsampling indexes `x[oy // 2, ox // 2]`, which
is one source element read K times along each axis -- the exact tile is
[OH/K, K, OW/K, K] with DRAM strides [W, 0, 1, 0]. `make_tptr` can carry a
stride of 0, so the box is expressible, and `visitOperandDiv` writes it: one
tensor axis becomes two state axes, the outer one counting the runs.

WHAT IT COST BEFORE. With no arm for `arith.divsi` the axis was unstructured.
ONE such axis is free -- `make_gather_scatter_tptr` carries it and strides
everything else -- but this kernel divides on BOTH axes and that op takes only
one, so the access fell to a per-element `tts.gather`: one index per element,
every DRAM stride 0, no burst.

WHY THE TORCHSIM-COMPILE REWRITE NEVER REACHED IT. `c01._retile_div_rem` (deleted 2026-09-06,
the split having made it dead on all 450 kernels) modelled `(x // K)*A +
(x % K)*B`, and this shape missed it in five places -- no `addi` root, a `//`
with no `%` partner, a `tt.expand_dims` between the `divsi` and its scale, ONE
CSE'd value feeding BOTH axes, and two splits that have to compose on an
already-split tile. THE LAST TWO ARE WHY IT BELONGED BELOW: a state that knows
which axes it has already cut answers them, and a syntactic rewrite cannot.

MEASURED on tests/cases/workloads/upsample.py: one of its five gathers is this
shape. The other four compute their index through `fptosi` of a float and are
unstructured for real.

A partnerless `//` splits the axis: the exact tile is [16, 2, 16, 2] with a stride of 0 on each inner half, which is what "read this source element K times" is.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

H = W = 16
OH = OW = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_div_only_axis_split_kernel(x_ptr, o_ptr, W: tl.constexpr,
                                       OH: tl.constexpr, OW: tl.constexpr):
        oy = tl.arange(0, OH)
        ox = tl.arange(0, OW)
        sy = oy // 2
        sx = ox // 2
        v = tl.load(x_ptr + sy[:, None] * W + sx[None, :])
        tl.store(o_ptr + oy[:, None] * OW + ox[None, :], v)

    return ptr_div_only_axis_split_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(H * W, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].view(H, W)
    return {"o": x.repeat_interleave(2, 0).repeat_interleave(2, 1).reshape(-1)}


SPEC = KernelSpec(
    name="ptr_div_only_axis_split_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "W": "constexpr",
               "OH": "constexpr", "OW": "constexpr"},
    constexprs={"W": W, "OH": OH, "OW": OW},
    args=[Arg("x", IN, "float32", (H * W,)),
          Arg("o", OUT, "float32", (OH * OW,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # NOT `expect="open"`: the ANSWER is right, so nothing fails today. What is
    # recorded is the SHAPE it is computed in, and expect_ir is what carries a
    # performance fact -- the day a partnerless `//` splits, this count moves
    # and whoever moved it reads the docstring above.
    # READS split-stepped-axes ON, which is how pipeline.py ships. The split is
    # built now: sizes [16, 2, 16, 2], strides [%c16, 0, 1, 0], and the load
    # folds back to the value with one collapse_shape. With the flag off this
    # was {"tts.gather": 1, "tts.make_tptr": 1} -- one address per element.
    expect_ir={"03-ttshared.mlir": {"tts.gather": 0, "tts.make_tptr": 2,
                                    "tensor.collapse_shape": 1}},
)
