"""A loop-carried accumulator that loses its unit axis INSIDE the body and
keeps it on the `scf.for`, so the two no longer agree on rank.

    init      tensor<8x1x32xf32>       the accumulator as the loop is entered
    iter_arg  tensor<8x32xf32>         the same value, one pass later, inside

    error: 'scf.for' op 0-th init and 0-th region iter_arg have different type:
           'tensor<8x1x32xf32>' != 'tensor<8x32xf32>'

WHAT THE FOLD IS AND WHY IT IS RIGHT TO WANT IT. The middle axis is a unit one:
`8x1x32` carries exactly the data of `8x32` and the extra dim costs an iterator
that no operand varies along. Dropping it is free, and it is how a rank-3 tile
that came out of `[:, None]` indexing gets back to a rank the lanes can take.

WHAT IS MISSING IS THE REST OF THE EDIT. A loop-carried value has three
occurrences that have to agree -- the `init` operand, the region's `iter_arg`,
and the `scf.yield` -- and the body is only one of them. Folding inside the
region and leaving the boundary alone produces an `scf.for` that no longer
verifies, which bufferize is simply the first pass to notice; the IR was already
wrong when it was written.

WHY IT NEEDS THE ROW. Remove `p_ptr`'s per-row load and this compiles. That load
is the only rank-2 operand -- `(d0,d1,d2) -> (d0,d1)` -- among rank-3 ones, and
it is what makes the pass want to line the ranks up at all. With every operand
the same rank there is nothing to reconcile and the unit axis is left alone.

WHY THE ACCUMULATOR HAS TO BE MASKED. `tl.where(r0_mask & xmask, acc + v, acc)`
is what keeps the accumulator a tensor with its own guarded update rather than
something the reduction can absorb. Written as a plain `acc + v` the loop is a
reduction the earlier passes rewrite, and the carried value never reaches this
shape.

THE BOUNDS ARE RUNTIME ON PURPOSE. With `xnumel` and `r0_numel` constexpr both
compares fold to true, the `tl.where` folds with them, and the carried tensor
goes away before any of this.

WHERE IT CAME FROM. Reducing Llama 3.2 Vision's cross-attention decoder layer
(PyTorchSim tests/models/Llama/test_mllama.py --part text). That kernel stops
EARLIER, in p06_absorb_layout_ops's `_map_restated`, and for a different reason than
this one -- measured by instrumenting the two `return None`s in
`_restated_maps`:

    PROBE: dim lost its bare mention; named=[0, 2] n_dims=3
           maps=[(d0,d1,d2) -> (d0,d2), (d0,d1,d2) -> (d0,d2)]

Its op is a single-input elementwise generic, `ins(%14) outs(%13)`, both maps
the identity over `tensor<32x1x1024xf32>`. The move drops the unit d1 from the
slot's map, and since there is no THIRD operand, d1 loses its last bare mention
and `_restated_maps` refuses. That is the case its own docstring names: "Every
generic in the suite whose unit axis is dropped has another operand naming the
same dim -- causal_conv1d's (d0,d1,d2)->(d0,d1) weight beside the
(d0,d1,d2)->(d0,d1,d2) tile -- and the one that does not is the one this
refuses."

So the two are NOT the same defect, and the absorb_layout_ops one has a name already:

    Mllama, absorb_layout_ops     a unit dim loses its last bare mention, and the refusal is
                    about linalg inferring a loop bound -- which for a UNIT dim
                    is 1 and not in question. `_restate_names` was taught
                    exactly this in "A unit axis that goes away takes no name
                    with it" (efaf26c): "a unit axis is not an axis ... it holds
                    one element, so it steps nothing". `_restated_maps` was not.
    here,   bufferize     the fold DOES go through and misses the loop boundary.

A rank-3 unit-middle tile with a two-axis mask and a body reading `linalg.index`
compiles clean on its own, and so does that plus a scalar `[1, 1]` broadcast and
a masked accumulator; only the per-row load fails, and it fails here. A
reproducer for the absorb_layout_ops refusal wants the opposite shape -- an op with NO second
operand naming the unit dim -- and is not in this file.

GREEN SINCE 2026-09-05, err 0, AND THE PARAGRAPH THIS REPLACES NAMED IT. A loop
carries four slots of one type -- init, iter_arg, yield operand, result -- and a
walk over VALUES reaches only the two outside the region. absorb_layout_ops was
widening those two and leaving the body at the old rank, which is exactly the
verifier error at the top of this file.

It refuses that plan now rather than half-taking it: a move it cannot carry
through the region is a move the layout op keeps, and a standing layout op is a
refusal this pass is allowed to make. Carrying it -- retyping the iter_arg and
the yield with the init and the result -- is the larger answer and is not built;
nothing in the suite asks for it yet.

torch equivalent: o[i] = sum_j (x[i,j] + p[i]*y[i,j])**2 over the masked box.

Found reducing Llama 3.2 Vision's cross-attention layer. NOT the absorb_layout_ops '_map_restated: indexing map cannot take the move' that stops that model -- this is what the reduction hit on the way, one pass later, and the absorb_layout_ops reproducer is still unfound: unit middle axis + two-axis mask + a body reading linalg.index compiles clean on its own, and so does that plus a scalar [1,1] broadcast. The per-row load is what makes either fail.
"""

#: WHAT THIS PINS, CLOSED 2026-09-23. It refused because THE OP WAS NOT IN THE
#: GRAPH AT ALL: `lib_hlo_graph.Build` took an operand that is a view node
#: (`memref.expand_shape`) as a read and stopped, so the slot the op WRITES was
#: never recorded, an op that defines nothing was skipped, and nothing ever
#: constrained its operands -- two of them on dims [0, 2] of the same `8x1x32`.
#:
#: The write through a view is now a new VERSION of the view's own buffer, which
#: is what upstream's kDynamicUpdateSlice defines. An earlier attempt that made
#: the op define the SOURCE's buffer instead cost 449/1 -> 440/8: the versions
#: then carried the view's rank against the source's, and the rungs below got a
#: map of rank 2 over a rank-3 operand.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

X, R = 8, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def unit_axis_carried_rank(x_ptr, p_ptr, y_ptr, o_ptr, xnumel, r0_numel,
                               XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        x0 = xindex
        # THE RANK-2 OPERAND. One value per row, `(d0,d1,d2) -> (d0,d1)` beside
        # the rank-3 tiles. Drop this load and the whole kernel compiles.
        tmp4 = tl.load(p_ptr + (x0), xmask, eviction_policy="evict_last")
        _acc = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            r0_1 = r0_index
            tmp0 = tl.load(x_ptr + (r0_1 + R0_BLOCK * x0), r0_mask & xmask,
                           other=0.0)
            tmp5 = tl.load(y_ptr + (r0_1 + R0_BLOCK * x0), r0_mask & xmask,
                           other=0.0)
            tmp8 = tmp0 + (tmp4 * tmp5)
            tmp9 = tmp8 * tmp8
            tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
            tmp12 = _acc + tmp10
            # THE GUARDED UPDATE. Keeps the accumulator a carried tensor instead
            # of a reduction the earlier passes fold away.
            _acc = tl.where(r0_mask & xmask, tmp12, _acc)
        acc = tl.sum(_acc, 1)[:, None]
        tl.store(o_ptr + x0, acc, xmask)

    return unit_axis_carried_rank


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(X * R, generator=g, dtype=torch.float32),
            "p": torch.randn(X, generator=g, dtype=torch.float32),
            "y": torch.randn(X * R, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(X, R)
    p = inputs["p"].reshape(X, 1)
    y = inputs["y"].reshape(X, R)
    v = x + (p * y)
    return {"o": (v * v).sum(dim=1).contiguous()}


SPEC = KernelSpec(
    name="unit_axis_carried_rank", kernel=kernel,
    signature={"x_ptr": "*fp32", "p_ptr": "*fp32", "y_ptr": "*fp32",
               "o_ptr": "*fp32", "xnumel": "i32", "r0_numel": "i32",
               "XBLOCK": "constexpr", "R0_BLOCK": "constexpr"},
    constexprs={"XBLOCK": X, "R0_BLOCK": R},
    args=[Arg("x", IN, "float32", (X * R,)),
          Arg("p", IN, "float32", (X,)),
          Arg("y", IN, "float32", (X * R,)),
          Arg("o", OUT, "float32", (X,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("xnumel", "int32_t"), ("r0_numel", "int32_t")],
           "scalar_values": {"xnumel": X, "r0_numel": R}},
    rtol=1e-4, atol=1e-4,
)
