"""A unit axis collapsed on the LAST operand that still named its loop.

    tile   tensor<32x1x512xf32>   the expand_shape triton-shared writes for a
                                  masked row reduce -- d1 is the unit axis
    row    tensor<32x1xf32>       the reduced value, broadcast back over d2

WHAT IT PINS. `absorb_layout_ops` collapses `32x1x512 -> 32x512` across the group,
one operand at a time. Each collapse restates that operand's indexing map, and
`_restated_maps` refuses a restatement that leaves an iteration dim with no bare
mention anywhere -- linalg infers a loop's bound from the operand shapes through
the maps, so a loop nobody names has no bound and the verifier says

    expected the shape-to-loops map to be non-null

That check is right about the maps and wrong about THIS move. The dim losing its
last namer is d1, and d1 is a PARALLEL loop of extent 1 that the body does not
name -- droppable, and the same function already drops exactly that
(`_unit_iterations` -> `_without_dims` -> `_renumber_indices`, the lines right
after the refusal). It never gets there:

    before   FAIL  _Fatal: indexing map cannot take the move
                   maps ['(d0,d1,d2)->(d0,d1,d2)', '(d0,d1,d2)->(d0,d2)']
                   restating slot 0 leaves both at (d0,d2), d1 unnamed
    after    PASS  the loop goes with the axis: two maps at (d0,d1)->(d0,d1),
                   two iterators, linalg.index dim 2 renumbered to 1

WHY THE ORDER MATTERS, and it is the whole of the case. Both operands of that
generic carry d1 when the walk starts, so collapsing the FIRST one is fine --
the other still names the loop. The refusal fires on the LAST one, when the
group has already taken the move everywhere else. A kernel whose unit axis is
named by an operand that does NOT take the move (causal_conv1d's
`(d0,d1,d2)->(d0,d1)` weight) never reaches this and is why the suite passed
without it.

WHERE IT CAME FROM. DeepSeek-V3 on the Triton route under transformers 5.15.0,
whose MoE gate is a sort fused into the RMSNorm reduce; measured the same on
DeepSeek-V2-Lite and V3.2-Exp, all three stopping on this one op. The shape
here is that kernel's, reduced: 432 of 512 columns live, so the mask survives
into the body and keeps the `linalg.index` reads that make the drop conditional
rather than free.

WHAT BREAKS IT AGAIN: reading the collapsed axis's extent off the maps AFTER
the move (it is unknowable there -- nothing names the dim, `_dim_extents`
returns None for it). The extent has to come from the move, which states it:
`of_old[d] is None` means that old axis had extent 1, and `move.old[d]` is that
1 written down.

A unit axis collapsed on the last operand that named its loop. absorb_layout_ops refuses the restatement instead of dropping the loop the axis took with it. DeepSeek-V2/V3/V3.2 all stop here under transformers 5.15.0.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

X, R = 32, 432                   # 432 live columns in a 512 block: the mask stays
XBLOCK, R0_BLOCK = 32, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_unit_axis_collapse_last_namer_kernel(
            in_ptr0, in_ptr1, in_ptr2, out_ptr0,
            xnumel: tl.constexpr, r0_numel: tl.constexpr,
            XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
        xindex = tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_mask = r0_index < r0_numel
        x0 = xindex
        r0_1 = r0_index

        tmp0 = tl.load(in_ptr0 + (r0_1 + r0_numel * x0), r0_mask & xmask,
                       other=0.0)
        # The row scalar. It is what gives the group a `(d0,d1)` operand beside
        # the `(d0,d1,d2)` tile, so the unit axis has a second namer for part of
        # the walk and the refusal lands on the LAST collapse rather than the
        # first.
        tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tl.where(r0_mask & xmask, tmp3, 0.0)
        tmp5 = tl.sum(tmp4, 1)[:, None]
        tmp6 = tmp5 / r0_numel
        tmp7 = tl.rsqrt(tmp6 + 1e-6)
        tmp8 = tmp2 * tmp7
        # The per-COLUMN weight: a (1, R0_BLOCK) operand, which is the map that
        # skips the unit axis.
        tmp9 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last',
                       other=0.0)
        tmp8 = tmp9 * tmp8
        tl.store(out_ptr0 + (r0_1 + r0_numel * x0), tmp8, r0_mask & xmask)

    return tile_unit_axis_collapse_last_namer_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(X, R, generator=g, dtype=torch.float32),
            "row": torch.randn(X, generator=g, dtype=torch.float32),
            "col": torch.randn(R, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    t = inputs["x"] + inputs["row"][:, None]
    rms = torch.rsqrt((t * t).mean(dim=1, keepdim=True) + 1e-6)
    return {"out": inputs["col"][None, :] * (t * rms)}


SPEC = KernelSpec(
    name="tile_unit_axis_collapse_last_namer_kernel", kernel=kernel,
    signature={"in_ptr0": "*fp32", "in_ptr1": "*fp32", "in_ptr2": "*fp32", "out_ptr0": "*fp32",
               "xnumel": "constexpr", "r0_numel": "constexpr",
               "XBLOCK": "constexpr", "R0_BLOCK": "constexpr"},
    constexprs={"xnumel": X, "r0_numel": R, "XBLOCK": XBLOCK,
                "R0_BLOCK": R0_BLOCK},
    args=[Arg("x", IN, "float32", (X, R)), Arg("row", IN, "float32", (X,)), Arg("col", IN, "float32", (R,)),
          Arg("out", OUT, "float32", (X, R))],
    grid=(1, 1),
    reference=reference, make_inputs=make_inputs,
)
