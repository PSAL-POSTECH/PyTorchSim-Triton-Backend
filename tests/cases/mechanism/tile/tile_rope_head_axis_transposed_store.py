"""Interleaved rotary read row-major and stored head-major, with 4 heads.

    load   x[row][head][dim]     row stride HEADS*DIM, head stride DIM
    store  out[head][row][dim]   head stride ROWS*DIM, row stride DIM
    angle  cos/sin indexed by ROW, which is the axis the store demotes

WHAT IT COVERS. The load and the store disagree about axis order, and the angle
belongs to the axis the store demotes:

    tl.load (in_ptr0 + (448 + x1 + 512*x2 + 2048*y0))   row-major
    tl.store(out_ptr0 + (x1 + 512*y0 + 16384*x2))       head-major

At HEADS = 1 that transpose is the identity, so four heads is what makes the
disagreement real.

WHERE IT CAME FROM, and it PASSES -- which is the point. DeepSeek-V4-Flash-Base
failed this op at four heads and passed it at one, so the axis order was the
obvious suspect and this kernel was written to convict it. It did not. The cause
was one layer down: a constant base dropped off a split axis in the linalg lowering's
rebuildAsGatherScatter, pinned by ../ptr/ptr_const_offset_with_split_axis.py.
The head count was a coincidence of which buffers carried a non-zero constant --
which is why "it fails at 4 and passes at 1" felt like evidence and was not.

    tests/cases/mechanism/tile/tile_rope_interleaved_pair_index.py is the same
    rotary WITHOUT the head axis or the transpose, and also passes.

It stays because a load and a store on different axis orders, with the angle on
the demoted axis, is a shape worth holding down.

WHAT BREAKS IT AGAIN: HEADS = 1, which makes the store's permutation vanish, or
indexing cos/sin by the head instead of the row, which would make the two axis
orders agree by accident.

Rotary read row-major, stored head-major, angle indexed by the row. DeepSeek-V4 fails this shape at 4 heads and passes it at 1.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS, HEADS, DIM = 32, 4, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_rope_head_axis_transposed_store_kernel(
            x_ptr, cos_ptr, sin_ptr, out_ptr,
            ROWS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr):
        # One program, three axes flattened the way Inductor flattens them:
        # y is the row, and x carries (head, dim).
        y = tl.arange(0, ROWS)[:, None]
        xh = tl.arange(0, HEADS * DIM)[None, :]
        head = xh // DIM
        col = xh % DIM

        base = col + DIM * head + (HEADS * DIM) * y          # row-major read
        even = tl.load(x_ptr + (2 * (col // 2) + DIM * head + (HEADS * DIM) * y))
        odd = tl.load(x_ptr + (1 + 2 * (col // 2) + DIM * head + (HEADS * DIM) * y))

        # The angle belongs to the ROW. The store below puts the head first, so
        # this is the index that has to survive an axis order it does not share.
        c = tl.load(cos_ptr + (col // 2 + (DIM // 2) * y))
        s = tl.load(sin_ptr + (col // 2 + (DIM // 2) * y))

        rotated = tl.where((col % 2) == 0, even * c - odd * s, odd * c + even * s)
        tl.store(out_ptr + (col + DIM * y + (ROWS * DIM) * head), rotated)

    return tile_rope_head_axis_transposed_store_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(ROWS, HEADS, DIM, generator=g, dtype=torch.float32),
            "cos": torch.randn(ROWS, DIM // 2, generator=g, dtype=torch.float32),
            "sin": torch.randn(ROWS, DIM // 2, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x, c, s = inputs["x"], inputs["cos"], inputs["sin"]
    even = x[..., 0::2]                       # [ROWS, HEADS, DIM/2]
    odd = x[..., 1::2]
    cc = c[:, None, :]                        # angle is the row's, shared by heads
    ss = s[:, None, :]
    rot = torch.empty_like(x)
    rot[..., 0::2] = even * cc - odd * ss
    rot[..., 1::2] = odd * cc + even * ss
    return {"out": rot.permute(1, 0, 2).contiguous()}   # head-major, as stored


SPEC = KernelSpec(
    name="tile_rope_head_axis_transposed_store_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "cos_ptr": "*fp32", "sin_ptr": "*fp32",
               "out_ptr": "*fp32", "ROWS": "constexpr", "HEADS": "constexpr",
               "DIM": "constexpr"},
    constexprs={"ROWS": ROWS, "HEADS": HEADS, "DIM": DIM},
    args=[Arg("x", IN, "float32", (ROWS, HEADS, DIM)),
          Arg("cos", IN, "float32", (ROWS, DIM // 2)),
          Arg("sin", IN, "float32", (ROWS, DIM // 2)),
          Arg("out", OUT, "float32", (HEADS, ROWS, DIM))],
    grid=(1,),
    reference=reference, make_inputs=make_inputs,
)
