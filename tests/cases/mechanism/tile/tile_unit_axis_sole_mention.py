"""A per-row mask over a row-gathered tile: the unit axis is one operand's ONLY
mention of its iteration dim.

WHAT IT PINS. `absorb_layout_ops` moves a layout into the axes of what reads it, and
a move with gaps -- `[16, 1, 512] -> [16, 512]` -- drops an axis. When the
dropped axis carried an operand's only bare mention of an iteration dim,
restating that operand's map leaves the dim named by nothing:

    ins  (d0, d1, d2) -> (d0, d1, d2)      becomes  (d0, d1, d2) -> (d0, d2)
    outs (d0, d1, d2) -> (d0, d2)          unchanged
                    ^^ named by nothing now

linalg cannot read a loop bound it cannot see, so the pass refused the whole
move. But a dim loses its last mention two ways and only one is fatal:

    the axis SPLIT     the loop is still there and its bound is unstated
    the axis was UNIT  the loop had one iteration and the move removed it,
                       so there is no bound to state

The second wants the dim deleted from the iteration space, which the pass knew
how to do one statement later (`_unit_iterations` -> `_without_dims`) and never
reached. It could not have reached it by inference either: once the dim is
unnamed, `_dim_extents` has no operand left to read its extent from. The MOVE is
what says it -- `of_old[d] is None` and `move.old[d] == 1` -- which is why
`_dims_the_move_retires` asks the move (rule 5).

WHAT BREAKS THIS. Reverting that. Measured on this file, stage 4:

    without   _Fatal: indexing map cannot take the move   (p06_absorb_layout_ops)
    with      all stages ok, max_abs_err 0

WHERE IT CAME FROM, AND WHY IT IS SHAPED LIKE THIS. transformers 5.x fuses the
MoE gate into one kernel and moonlight, kimi-k2, kimi-k25 and DeepSeek-V3 all
stopped on that line. FIVE ATTEMPTS TO BUILD IT UP FROM SCRATCH FAILED -- a
reshape leaves every map rank 3 and pins nothing, a sum over the unit axis
crashes vectorize instead (tile_reduce_over_unit_axis.py), and a plain masked RMSNorm
does not reach it. What worked was shrinking the real kernel until it stopped
reproducing, and the three pieces below are what survived that:

    a per-row index READ FROM MEMORY, used to gather a per-row scalar
    a 2-D gather whose ROW index is that same loaded value
    a `where` whose condition is per-ROW ([X, 1]) over a per-element value

The last is the one that mints the unit axis: a [16, 1] condition broadcast
against a [16, 512] value is where the `1` comes from, and dropping it is what
made every hand-built attempt pass either way. The RMSNorm tail is kept because
it is what reads the result at rank 2, which is what makes the ins map the SOLE
mention.

WHAT IS NOT LOAD-BEARING, each checked by removing it and watching the refusal
survive: the sigmoid/top-k scaling chain, the second index chain, the `>= 2`
comparison being a comparison rather than any per-row predicate, and two of the
three stacked input loads.

ARITHMETIC. RMSNorm over 416 of 512 columns, f32. Exact bar the reciprocal
square root, which is why the tolerance is the suite's default rather than 0.

a per-row predicate over a row-gathered tile, whose unit axis is one operand's only mention of its iteration dim. POSITIVE.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

X = 16          # rows
R = 416         # real columns
RB = 512        # the padded block, so the mask is live
EPS = 1e-5


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_unit_axis_sole_mention_kernel(o_ptr, v_ptr, e_ptr, c_ptr, g_ptr,
                                           w_ptr, xnumel, r0_numel,
                                           XBLOCK: tl.constexpr,
                                           R0_BLOCK: tl.constexpr):
        xindex = tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0 = tl.arange(0, R0_BLOCK)[None, :]
        r0_mask = r0 < r0_numel
        x0 = xindex

        base = tl.load(v_ptr + (r0 + 416 * x0), r0_mask & xmask, other=0.0)

        # A PER-ROW INDEX READ FROM MEMORY, clamped the way the gate clamps it.
        e = tl.load(e_ptr + x0, xmask, eviction_policy='evict_last')
        e = tl.where(e < 0, e + XBLOCK, e)

        # A PER-ROW SCALAR GATHERED BY IT, and the predicate built from that.
        # [XBLOCK, 1] -- this is the operand whose unit axis is the whole point.
        cnt = tl.load(c_ptr + e, xmask, eviction_policy='evict_last')
        drop = cnt >= 2

        # A 2-D GATHER WHOSE ROW INDEX IS THAT SAME LOADED VALUE.
        rows = tl.load(g_ptr + (r0 + 416 * e), r0_mask & xmask, other=0.0)

        # THE `where` IS WHAT MINTS THE UNIT AXIS: a [XBLOCK, 1] condition
        # against a [XBLOCK, R0_BLOCK] value. Every attempt without it passed
        # with and without the fix, which is to say it pinned nothing.
        v = base + tl.where(drop, 0.0, rows)

        # The RMSNorm tail reads that result at rank 2, which is what leaves the
        # ins map above as the SOLE mention of the dim the move retires.
        sq = tl.where(r0_mask & xmask, tl.broadcast_to(v * v, [XBLOCK, R0_BLOCK]), 0.0)
        rs = 1.0 / tl.sqrt(tl.sum(sq, 1)[:, None] / 416.0 + 1e-05)
        w = tl.load(w_ptr + r0, r0_mask, eviction_policy='evict_last', other=0.0)
        tl.store(o_ptr + (r0 + 416 * x0), w * (v * rs), r0_mask & xmask)

    return tile_unit_axis_sole_mention_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"v": torch.randn(X * R, generator=g, dtype=torch.float32),
            "e": torch.randint(0, X, (X,), generator=g, dtype=torch.int32),
            "c": torch.randint(0, 4, (X,), generator=g, dtype=torch.int32),
            "gg": torch.randn(X * R, generator=g, dtype=torch.float32),
            "w": torch.randn(R, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    e = inputs["e"].long()
    base = inputs["v"].reshape(X, R)
    rows = inputs["gg"].reshape(X, R)[e]
    drop = (inputs["c"][e] >= 2).reshape(X, 1)
    v = base + torch.where(drop, torch.zeros_like(rows), rows)
    rs = torch.rsqrt((v * v).sum(1, keepdim=True) / 416.0 + EPS)
    return {"o": inputs["w"].reshape(1, R) * (v * rs)}


SPEC = KernelSpec(
    name="tile_unit_axis_sole_mention_kernel",
    kernel=kernel,
    signature={"o_ptr": "*fp32", "v_ptr": "*fp32", "e_ptr": "*i32",
               "c_ptr": "*i32", "g_ptr": "*fp32", "w_ptr": "*fp32",
               "xnumel": "i32", "r0_numel": "i32",
               "XBLOCK": "constexpr", "R0_BLOCK": "constexpr"},
    constexprs={"XBLOCK": X, "R0_BLOCK": RB},
    args=[Arg("o", OUT, "float32", (X, R)),
          Arg("v", IN, "float32", (X * R,)), Arg("e", IN, "int32", (X,)),
          Arg("c", IN, "int32", (X,)), Arg("gg", IN, "float32", (X * R,)),
          Arg("w", IN, "float32", (R,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    extra={"scalar_args": [("xnumel", "int32_t"), ("r0_numel", "int32_t")],
           "scalar_values": {"xnumel": X, "r0_numel": R}},
)
