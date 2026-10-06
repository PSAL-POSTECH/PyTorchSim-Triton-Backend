"""The COMPUTED half of tile_rank3_row_inlane.py, and nothing can stage it.

Same shape, same refusal, one difference: the plane broadcast along the rank-3
tile's middle axis is a REDUCTION RESULT rather than a load.

    x  = tl.load(...)             [M, K, N]   the tile, banked on axis 0
    s  = tl.sum(x, axis=1)        [M, N]      computed on chip
    y  = x * s[:, None, :]                    s read as (d0, 0, d2)

EXPECTED TO PASS AND IT DOES NOT, WHICH IS THE POINT. Not `blocked`: nothing
about spike or LLVM is in the way, the gap is ours (see below), and `blocked`
is reserved for a fact about somebody else's tree. Not `refuse` either: this is
not a decision, it is a form vectorize has not been given. So it stands as
a red line in the sweep until the form lands, which is what a reproducer is.

WHY IT FAILS. Nothing about this is impossible on the
machine: with the lanes on axis 0, lane L holds a K x N slab of the tile and the
1 x N plane it needs is built from that same slab. Not one element crosses a
bank. What is missing is that vectorize has ONE broadcast form -- read one
element per lane and repeat it down that lane's vector (`vload`'s splat path) --
and this operand is constant down axis 1 while varying down axis 2, so the splat
would repeat the wrong element:

    NotVectorisable: operand [8, 1, 16] broadcasts along dim 0 into a tile
    [8, 4, 16], and it was staged one element per lane

fit_layout_to_lanes's staging menu cannot answer it either, and the reason is in
the menu rather than in this kernel:

    1  widen the load        there is no load. `s` is written by a
                             linalg.reduce, so there is no DRAM address to
                             re-read with a zero stride.
    2  rebuild it            _rebuild_computed_row needs a producer that reads
                             NOTHING (a `tl.arange` body). This one reads the
                             tile.
    3  replicating relayout  a round trip buys nothing here -- rule 8b -- since
                             nothing has to cross a lane, and
                             _replicate_computed_row has no rank-3 form anyway.

    4  the one that is missing: materialise the operand at tile shape IN LANE.
       It is a pure in-lane copy, no DMA and no bank crossing -- but written as
       a linalg.generic it is an op with the SAME broadcast vectorize
       cannot do, so it has to be vectorize that learns the form, not
       fit_layout_to_lanes that emits another op.

WHAT LIFTS THIS. A second broadcast form in vectorize: an operand constant
on some in-lane axes and varying on others is an in-lane strided read, which the
vector unit can do -- it is the splat's addressing that is hardcoded, not the
hardware. When that lands, this kernel starts passing and `blocked` comes off.

WHERE IT WAS FOUND. dl/cascade_merge_state, dl/selective_state_update and
sort/sort_radix all die on exactly this, and all three are large enough that the
message could be blamed on anything else in them. This is the same failure with
nothing else in it.

SISTER KERNEL: tile_rank3_row_inlane.py, identical but with `b` LOADED, which
passes -- staged by a second MVIN with `dram_stride = [16, 0, 1]`. The pair is
what shows the gap is about where the values come from and not about the shape.

The computed half of tile_rank3_row_inlane.py. Same shape, same refusal, and no menu entry answers it because the plane has no DRAM address and no input-free producer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 8, 4, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_rank3_computed_row_inlane_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                              K: tl.constexpr,
                                              N: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        off = (rm[:, None, None] * (K * N) + rk[None, :, None] * N
               + rn[None, None, :])
        x = tl.load(x_ptr + off)
        s = tl.sum(x, axis=1)                 # [M, N], computed on chip
        tl.store(o_ptr + off, x * s[:, None, :])

    return tile_rank3_computed_row_inlane_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * K * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(M, K, N)
    s = x.sum(dim=1, keepdim=True)
    return {"o": (x * s).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="tile_rank3_computed_row_inlane_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr",
               "K": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "K": K, "N": N},
    args=[Arg("x", IN, "float32", (M * K * N,)),
          Arg("o", OUT, "float32", (M * K * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
