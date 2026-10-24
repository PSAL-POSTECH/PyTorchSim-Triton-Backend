"""A clamped store of a per-row scalar -- the clamp names an axis of extent 1.

WHAT THIS PINS. `mask_and_fill` names EVERY axis of `static_mask_dims` in
`masked_axes`, including one whose static extent already covers the tile. That
clamp can never fire, but the NAME is real to every reader of the transfer, and
`absorb_layout_ops` is the one that cannot live with it: a per-row scalar leaves the
reduction as a [XBLOCK, 1] tile, the layout op that collapses the unit axis has
no name to give it, and `_restate_names` refuses -- correctly, since an axis
that goes away takes its name with it.

    pytorchsim_triton_compiler.lowering.passes.p06_absorb_layout_ops._Fatal: transfer's masked_axes cannot take
    the move: "pytorchsim.transfer"(...) {direction = "store", dram_stride = [1],
    masked_axes = [0, 1], masked_fill = 0}

The refusal is right and the name was wrong. `mask_and_fill` already knows the
concept one line above -- "an extent that already covers the tile clamps to a
no-op" -- and used it only to decide the FILL, all axes at once. Per axis, a
covering extent means: do not name it, and do not emit its clamp pair.

THE SHAPE THAT SHOWS IT. Inductor writes every reduction statistic this way, so
this is the first kernel of `tests/models/test_transformer.py` that does not
compile: layernorm stores `mean` and `rsqrt` as [XBLOCK, 1] under `xmask` while
the normalised rows go out as [XBLOCK, R0_BLOCK] under the same mask. The rank-2
store is what makes the unit axis exist at all -- `tl.store(p + rows, s, mask)`
with rank-1 operands never grows one.

  XNUMEL(6) < XBLOCK(8)   so axis 0's clamp is REAL and has to survive
  the store's second axis is 1 wide with mask dim 1, so its clamp is the no-op

WHAT WOULD BREAK THIS. Naming a covering axis again in `masked_axes`; or making
`absorb_layout_ops` drop a name it cannot move, which would silently discard a clamp
that does fire.

A clamped [XBLOCK, 1] store: the mask names an axis of extent 1 that the collapse then removes. Covers mask_and_fill's per-axis no-op clamp and absorb_layout_ops's refusal to move a name that went away.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

XNUMEL = 6      # real rows
XBLOCK = 8      # rows staged -- the mask is XNUMEL < XBLOCK
RBLOCK = 8      # columns, reduced


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def row_scalar_store_unit_axis(in_ptr, sum_ptr, out_ptr,
                                   xnumel: tl.constexpr,
                                   XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
        xindex = tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r = tl.arange(0, RBLOCK)[None, :]
        x = tl.load(in_ptr + (r + RBLOCK * xindex), xmask, other=0.0)
        s = tl.sum(x, 1)[:, None]
        # THE UNIT AXIS. Both operands are [XBLOCK, 1], so this store's tile has
        # a second axis of extent 1 and its mask dim is 1 -- the clamp that
        # cannot fire and whose name absorb_layout_ops then cannot move.
        tl.store(sum_ptr + xindex, s, xmask)
        tl.store(out_ptr + (r + RBLOCK * xindex), x - s, xmask)

    return row_scalar_store_unit_axis


def make_inputs(torch, seed=0):
    # A distinct constant per real row, so a row sum that lost terms and a row
    # that was never written (0.0) are both visible in the output.
    x = torch.zeros(XNUMEL * RBLOCK, dtype=torch.float32)
    for i in range(XNUMEL):
        x[i * RBLOCK:(i + 1) * RBLOCK] = float(i + 1)
    return {"x": x}


def reference(inputs):
    x = inputs["x"].reshape(XNUMEL, RBLOCK)
    return {"s": x.sum(dim=1).contiguous(),
            "o": (x - x.sum(dim=1, keepdim=True)).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="row_scalar_store_unit_axis", kernel=kernel,
    signature={"in_ptr": "*fp32", "sum_ptr": "*fp32", "out_ptr": "*fp32",
               "xnumel": "constexpr", "XBLOCK": "constexpr",
               "RBLOCK": "constexpr"},
    constexprs={"xnumel": XNUMEL, "XBLOCK": XBLOCK, "RBLOCK": RBLOCK},
    args=[Arg("x", IN, "float32", (XNUMEL * RBLOCK,)),
          Arg("s", OUT, "float32", (XNUMEL,)),
          Arg("o", OUT, "float32", (XNUMEL * RBLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=0, atol=0,
)
