"""A MASKED load and a masked store: the box, its two ends, and the fill.

Split out of the old tests/cases/ptr_structured.py section D. The mask survived
MaskAnalysis as a pair of dims (see mask_1d_cmp_dims.py), so by the time p06
sees it there is no i1 tensor left -- only a box. The DMA takes that box and the
`other=` value as two attributes and does the rest in hardware.

THE COMPARISON THIS FILE USED TO MAKE IS GONE, and that is the point of keeping
it. StructuredToMemref needed an alloc, an scf.if-guarded linalg.fill, two
subviews and a memref.copy for this; the pass is deleted (rule 21) and the same
three accesses now come out as three transfers with no scf.if and no
linalg.fill anywhere in 08-transformed.mlir.

COVERS (4)   p01_convert_accesses_to_transfer.py, mask_and_fill()

  mask-dim-dynamic     the extent is an SSA Value -> `extent = next(mit)`,
      passed as a clamp operand. Dim 0 of the mask, which depends on pid.
  mask-dim-static      the extent is an attribute -> self.const(lo + m). Dim 1,
      the literal `< 8`. One load covers both arms because the two dims of its
      mask are of different kinds -- that is why the mask is written this way.
          "torchsim.transfer"(..., %c0, %10, %c0, %c8)
                                  ^^^^^^^^  ^^^^^^^^ (lo, hi) per masked axis
  mask-noop-dropped    an extent that already covers the tile clamps to
      nothing, so it never reaches masked_axes and drags in no descriptor
      store. The masked STORE has one axis for `r < 16` and no second entry.
  masked-store         a masked tts.store is an MVOUT with the same box; there
      is no read-modify-write, because the DMA drops the out-of-box addresses.

AND ONE CONTRAST THAT NO LONGER EXISTS
--------------------------------------
Loads 1 and 2 are the same address and mask with and without `other=0.0`, and
they now produce the SAME transfer. mask_and_fill leaves `fill` at 0 when
`other` is absent, so `masked_fill = 0` either way -- an absent other is
UNDEFINED in Triton and filling it with zero is a legal refinement. The old
path could tell them apart (fill vs no fill at all, leaving the alloc dirty);
ours cannot, and there is nothing to fix: the answer is the same.

WHAT PROVES IT (08-transformed.mlir, verbatim -- three transfers, zero scf.if)

  load 1   {dma_kind = "MVIN",  dram_stride = [1, 32], masked_axes = [1, 0],
            masked_fill = 0}
  load 2   {dma_kind = "MVIN",  dram_stride = [32, 1], masked_axes = [0, 1],
            masked_fill = 0}
      -- the two differ only because absorb_layout_ops absorbed a transpose into one of them
         and renumbered its per-axis attributes with it (rule 18).
  store    {dma_kind = "MVOUT", dram_stride = [1],     masked_axes = [0],
            masked_fill = 0}

AND IT WAS ALSO HOLDING DOWN A WRONG NUMBER, which is not what it was written
for and is worth saying because a census kernel is easy to read as "coverage
only". `tl.sum(..., axis=1)` makes absorb_layout_ops turn the tile under ONE of the
two loads, and the turn permuted that MVIN's `dram_stride` [32, 1] -> [1, 32]
while leaving `masked_axes = [0, 1]` naming the old axes -- so the clamp that
meant "the first %7 ROWS" was applied to the columns. Measured:
`verify: numerical mismatch 7.91137`, got 7.9113655 where 15.822732 was correct
on 16 of 32 elements, i.e. exactly one of the two loads contributing. The two
loads are byte-for-byte identical apart from `other=`, which is what made the
factor of two readable. See tests/cases/mechanism/mask/mask_clamped_store_turned_tile
.py, which pins the descriptor rewrite on its own.

Masked load and store: the box's two ends (one runtime, one literal) and the fill. Numerically runnable -- load 2's tail is masked off again before the store either way.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BS = 32
NY = BS * BS
KEEP = 16       # the static store bound
COLS = 8        # the static mask column bound


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_memref_masked_fill_kernel(y_ptr, o_ptr, BS: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, BS)
        p2d = y_ptr + r[:, None] * BS + r[None, :]
        # dim 0 of the mask is dynamic (pid), dim 1 is a literal
        m2 = (r[:, None] < (pid + 1) * BS) & (r[None, :] < 8)

        a = tl.load(p2d, mask=m2, other=0.0)   # 1 fill + subview copy
        b = tl.load(p2d, mask=m2)              # 2 subview copy only
        # collapse to 1-D so the masked STORE is a plain one-dimensional case
        v = tl.sum(a + b, axis=1)
        tl.store(o_ptr + r, v, mask=r < 16)    #   masked store

    return ptr_memref_masked_fill_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"y": torch.randn(NY, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    y = inputs["y"].reshape(BS, BS)
    keep = torch.zeros(BS, BS, dtype=torch.float32)
    keep[:, :COLS] = y[:, :COLS]           # pid == 0 -> the row mask is full
    v = (keep + keep).sum(dim=1)
    o = torch.zeros(BS, dtype=torch.float32)
    o[:KEEP] = v[:KEEP]
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_memref_masked_fill_kernel", kernel=kernel,
    signature={"y_ptr": "*fp32", "o_ptr": "*fp32", "BS": "constexpr"},
    constexprs={"BS": BS},
    args=[Arg("y", IN, "float32", (NY,)), Arg("o", OUT, "float32", (BS,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
