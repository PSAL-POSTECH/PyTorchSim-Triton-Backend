"""The same one-entry index, with the tile banked on the OTHER axis.

WHY THIS IS A SEPARATE FILE. The index buffer holds one entry per gathered
position and the DMA finds each entry by that position's own coordinate --
never by the element's scratchpad slot, and never by the lane it is being
written into. So the tile's lane axis is free: the load below is the same 16x8
row gather as gather_index_one_entry_per_row, but the sum over the gathered
axis makes fit_layout_to_lanes bank the TILE on axis 1 instead, and the index is
unchanged by that.

WHAT DECIDES THE INDEX'S OWN BANK. The index is its own rank-1 tile, banked the
way any rank-1 tile is: entry p lives in lane `p % lanes` at offset
`p / lanes`. The descriptor gives spike `indirect_dim`, spike recomputes that
bank from the gathered coordinate, and the DMA is free to read any lane's bank
because it is the engine that drives the lanes, not one of them.

WHAT PROVES IT (04-passes/07-torchsim-bufferize.mlir)
      %alloc_4 = memref.alloc() {torchsim.lane_axis = 1} : memref<16x8xf32>
      %alloc_3 = memref.alloc() {torchsim.lane_axis = 0} : memref<16xi32>
      "torchsim.transfer"(... %alloc_3) {indirect, indirect_dim = 0, ...}
  The tile is on axis 1 and the index is still sixteen entries, one per row.
  An earlier design banked the index alongside the tile and could only drop the
  replica when the two axes agreed; the count below is what says that
  dependency is gone.

THE SAME TWO AXES UNDER THE BANK VIEW, where the lane axis is dim 1 of the type
rather than an attribute: the tile reads `memref<1x8x16xf32>` -- vlane 8, which
is axis 1 of the 16x8 tile -- and the index reads `memref<1x16xi32>`, sixteen
entries over sixteen lanes. Still two different axes, still one entry per row,
and `indirect_dim` moves with the permutation (0 on the tile view, 2 on the bank
view) because it names the tile axis the coordinate comes from. The declaration
names both spellings of each type and counts them together.

torch equivalent: o = x.view(-1, 8)[i].sum(0)

The tile banked on the other axis, and the index still one entry per gathered row: the descriptor reads it by coordinate, not lane.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NX = 2048
BM = 16
BN = 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_index_survives_a_foreign_lane_axis_kernel(
            x_ptr, i_ptr, o_ptr, BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        i = tl.load(i_ptr + rm)
        a = tl.load(x_ptr + i[:, None] * BN + rn[None, :])
        tl.store(o_ptr + rn, tl.sum(a, axis=0))

    return gather_index_survives_a_foreign_lane_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "i": torch.randint(0, NX // BN, (BM,), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    x, i = inputs["x"].view(-1, BN), inputs["i"].long()
    return {"o": x[i].sum(0).contiguous()}


SPEC = KernelSpec(
    name="gather_index_survives_a_foreign_lane_axis_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int32", (BM,)),
          Arg("o", OUT, "float32", (BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # FIVE AND NOT EIGHT since 2026-09-06 -- see gather_index_one_entry_per_row:
    # the snapshot copy is gone, and it named the type twice.
    # TWO SPELLINGS OF EACH TYPE: the bank view puts the lane axis in the type,
    # so the index is `memref<1x16xi32>` there and a replica would arrive as
    # `memref<1x8x16xi32>`. `indirect_dim` is counted as a token, not a value --
    # that the descriptor names an axis at all is the claim.
    expect_ir={"04-passes/06-torchsim-bufferize.mlir": {
        ("memref<16xi32>", "memref<1x16xi32>"): 2,
        ("memref<16x8xi32>", "memref<1x8x16xi32>"): 0,
        "indirect_dim": 1}},
)
