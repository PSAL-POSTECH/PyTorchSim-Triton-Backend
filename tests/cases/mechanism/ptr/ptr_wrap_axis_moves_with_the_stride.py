"""A wrapped tile that absorb_layout TRANSPOSES: the cut has to move with it.

    row = (R0 + i) % R        i = 0..N-1,  R0 = 12, R = N = 16
    rows 12..15 then 0..11    one seam -> two pieces on tile axis 0
    then a ROW-WISE reduction, which absorb_layout answers by transposing the
    staged tile -- and a transpose is a rename of the axis the cut names

WHAT IT PINS: that `pytorchsim.cut` is restated with `dram_stride` and not left
naming the axis it named before (rule 18). A transfer states several things per
axis and a transpose has to move all of them; moving only the stride leaves the
cut on the WRONG axis, and the descriptor then skips rows nobody owns.

WHAT PROVES IT: the tile is 8x8 and the cut is 4+4, so a cut left on the other
axis still writes half the tile -- the failure is a wrong number, not a refusal.
Measured on the tree before the restatement: 0.509115 against the reference.

A wrap plus a row-wise reduction: the cut and the stride are two statements about one axis and absorb_layout has to restate both.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R = 16          # the modulus: how many rows the source has
R0 = 12         # first row read, so 12..15 then 0..11 -- one seam
N = 16          # rows read: every row, each exactly once
M = 8           # columns, the axis the reduction runs along


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_axis_moves_with_the_stride_kernel(
            x_ptr, o_ptr, R: tl.constexpr, R0: tl.constexpr,
            N: tl.constexpr, M: tl.constexpr):
        i = tl.arange(0, N)
        j = tl.arange(0, M)
        # The wrap is on the ROW axis, and the reduction runs along the other
        # one -- so the layout answer moves the rows, cut and all.
        row = (R0 + i) % R
        idx = row[:, None] * M + j[None, :]
        v = tl.load(x_ptr + idx)
        tl.store(o_ptr + idx, v - tl.sum(v, axis=1)[:, None])

    return ptr_wrap_axis_moves_with_the_stride_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R, M, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    return {"o": x - x.sum(dim=1, keepdim=True)}


SPEC = KernelSpec(
    name="ptr_wrap_axis_moves_with_the_stride_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr",
               "R0": "constexpr", "N": "constexpr", "M": "constexpr"},
    constexprs={"R": R, "R0": R0, "N": N, "M": M},
    args=[Arg("x", IN, "float32", (R, M)), Arg("o", OUT, "float32", (R, M))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
