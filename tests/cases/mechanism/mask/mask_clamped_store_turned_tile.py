"""A masked store whose tile is TURNED under it -- the clamp box has to turn too.

WHAT THIS PINS. `absorb_layout_ops` sinks a permutation into a `torchsim.transfer` by
permuting its `dram_stride`, and its own docstring used to call that the whole
rewrite: "the strides are all this has to say". They are not. A transfer states
FOUR things per axis -- `dram_stride`, `subtile_size`, `masked_axes` and
`dyn_stride_axes` -- and every one of them indexes the axes being renamed.
`_reshaped_strides` already knew this and refuses a RANK change on a transfer
carrying any of the four; the permutation arm next to it renamed one and left
the other three, which is the half-rewrite rule 4 is about.

THE SHAPE THAT SHOWS IT. XBLOCK rows are staged but only `xnumel` of them are
real, so both the masked load and the masked store carry a clamp of (0, xnumel)
on axis 0. Axis 1's extent is RBLOCK and its mask dim is RBLOCK, so that clamp
cannot fire and `mask_and_fill` does not name it -- `masked_axes = [0]` here,
`[1]` on the turned MVIN. It used to name both, which is what made a per-row
scalar's [XBLOCK, 1] store unabsorbable (see mask_row_scalar_store_unit_axis).
The name that has to move is the one that clamps. `tl.sum(x, 1)`
arrives from triton-shared as `linalg.transpose` + `linalg.reduce dimensions =
[0]`, and absorb_layout_ops sinks that transpose BACKWARDS into the MVIN that staged
the tile -- so the tile the DMA fills is the transposed one and its axis 0 is
now the RBLOCK axis.

MEASURED before the fix, on this file at a6271ac -- and NOTHING reports it:

    "torchsim.transfer"(...) {dma_kind = "MVIN", dram_stride = [1, 8],
                            masked_axes = [0, 1]}       <- [1, 0] is correct

    o          FAIL  shape=(48,) dtype=float32
      max_abs_err = 12   max_rel_err = 0.285714   (rtol=0, atol=0)
      worst at flat index 40: got=np.float32(-30.0)  want=np.float32(-42.0)
      48/48 elements outside tolerance

The descriptor is self-consistent, verifies, lowers to a legal DMA and moves the
wrong bytes: `dram_stride` turned to [1, 8] while `masked_axes` still clamped the
tile's axis 0 to `xnumel = 6`, and after the turn that axis is the RBLOCK axis of
extent 8. Two of every row's eight columns were never loaded, so every row sum
was short by two terms -- row 5 summed to 36 instead of 48 and the answer came
back -30 where -42 was correct. Only stage 8 saw it (rule 12).

WHY THE MASK MUST BE ROW-CLAMPING AND NOT COLUMN-CLAMPING. A clamp box whose two
axes carry the SAME extent cannot tell a turned descriptor from an unturned one,
and this is exactly the file that has to. XBLOCK(8) != xnumel(6) and RBLOCK(8) is
full, so the two axes disagree and the transposition is visible in the values.

WHAT WOULD BREAK THIS. Permuting `dram_stride` on its own again; permuting the
`masked_axes` ENTRIES' positions instead of renaming them (the operand pairs are
positional and would then be attached to the wrong axis); or absorbing a
permutation into a transfer that carries `subtile_size` or `dyn_stride_axes`
without renaming those too.

A clamped transfer whose tile absorb_layout_ops turns. The clamp box has to be renamed by the same permutation as the DRAM strides; renaming only the strides moves 6 of 8 columns and reports success.
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
    def clamped_store_turned_tile(in_ptr, out_ptr, xnumel: tl.constexpr,
                                  XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
        xindex = tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r = tl.arange(0, RBLOCK)[None, :]
        x = tl.load(in_ptr + (r + RBLOCK * xindex), xmask, other=0.0)
        # THE TURN. triton-shared spells a reduce over the last axis as a
        # transpose feeding `dimensions = [0]`, and absorb_layout_ops sinks that
        # transpose all the way to the MVOUT below.
        s = tl.sum(x, 1)[:, None]
        tl.store(out_ptr + (r + RBLOCK * xindex), x - s, xmask)

    return clamped_store_turned_tile


def make_inputs(torch, seed=0):
    # Each real row is a distinct constant, so `x - rowsum` is a distinct
    # non-zero constant per row and a column that was never written (0.0) can
    # be told from one that was.
    x = torch.zeros(XNUMEL * RBLOCK, dtype=torch.float32)
    for i in range(XNUMEL):
        x[i * RBLOCK:(i + 1) * RBLOCK] = float(i + 1)
    return {"x": x}


def reference(inputs):
    x = inputs["x"].reshape(XNUMEL, RBLOCK)
    return {"o": (x - x.sum(dim=1, keepdim=True)).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="clamped_store_turned_tile", kernel=kernel,
    signature={"in_ptr": "*fp32", "out_ptr": "*fp32", "xnumel": "constexpr",
               "XBLOCK": "constexpr", "RBLOCK": "constexpr"},
    constexprs={"xnumel": XNUMEL, "XBLOCK": XBLOCK, "RBLOCK": RBLOCK},
    args=[Arg("x", IN, "float32", (XNUMEL * RBLOCK,)),
          Arg("o", OUT, "float32", (XNUMEL * RBLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=0, atol=0,
)
