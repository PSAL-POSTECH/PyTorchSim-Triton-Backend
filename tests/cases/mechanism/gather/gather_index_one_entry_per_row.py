"""The gather index is one entry per gathered row, not one per tile element.

WHAT THE DMA USED TO NEED. The indirect descriptor read its index at the
element's own scratchpad slot -- `indirect_base + s_idx * esize + lane_bank`,
where `s_idx` counts ELEMENTS inside one lane's bank. So an index buffer had to
be the same shape as the tile, and convert_accesses_to_transfer replicated a row index
across it: for the 16x8 tile below, 16 values became 128 slots, plus a DMA that
re-read the same eight DRAM words with a zero stride to fill them.

WHAT SAYS IT INSTEAD. The descriptor carries `indirect_dim` at byte +132 -- 0 is
the old per-element walk, k+1 names the axis whose COORDINATE addresses the
index. spike already has `n`/`c`/`h`/`w` in the DMA loop, so one entry per
position on the gathered axis is enough and the replica goes. The widening in
convert_accesses_to_transfer went with it.

WHERE THE ENTRY LIVES. The index is its own rank-1 tile, banked the way any
rank-1 tile is -- entry p in lane `p % lanes` at offset `p / lanes` -- and spike
recomputes that bank from the coordinate. It does not read the bank of the lane
it is writing into, which is why the TILE's lane axis has no say here; see
gather_index_survives_a_foreign_lane_axis.py, the same gather with the tile
banked on the other axis and the same index.

WHAT PROVES IT (08-passes/07-torchsim-bufferize.mlir)
      %alloc_3 = memref.alloc() {torchsim.lane_axis = 0} : memref<16xi32>
      "torchsim.transfer"(... %alloc_3) {gather ..., indirect_dim = 0, ...}
        : (..., memref<16x8xf32>, ..., memref<16xi32>) -> ()
  Sixteen entries against a 16x8 tile. The two absences carry as much: no
  `memref<16x8xi32>` anywhere, and no `linalg.broadcast` -- the two shapes the
  replica used to arrive in.

WHERE THE LANE AXIS IS WRITTEN DOWN. Under the bank view it is dim 1 of the type
and not an attribute, so those same two lines read `memref<1x16xi32>` for the
index and `memref<1x8x16xf32>` for the tile. Nothing about this gather changes:
sixteen entries, one per row, and no replica. The declaration therefore names
both spellings of each type and counts them together -- the claim is about the
buffer, and which of the two ways it is spelt is the layout path's business.

torch equivalent: o.view(16, 8)[j, :] = x.view(-1, 8)[i[j], :]

A row index needs one entry per row, not one per element: the descriptor's indirect_dim names the axis and the replica goes.
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
    def gather_index_one_entry_per_row_kernel(x_ptr, i_ptr, o_ptr,
                                              BM: tl.constexpr,
                                              BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        i = tl.load(i_ptr + rm)
        a = tl.load(x_ptr + i[:, None] * BN + rn[None, :])
        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], a)

    return gather_index_one_entry_per_row_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "i": torch.randint(0, NX // BN, (BM,), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    import torch
    x, i = inputs["x"].view(-1, BN), inputs["i"].long()
    return {"o": x[i].reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="gather_index_one_entry_per_row_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int32", (BM,)),
          Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # FIVE AND NOT EIGHT since 2026-09-06: the index tile used to be built in
    # one buffer and snapshotted into the gather's own, and bufferize's copy
    # named the type twice more. bufferize._write_where_the_copy_lands lands the
    # build in the gather's buffer, so the count doubles as that pin.
    # TWO SPELLINGS OF EACH TYPE: the bank view puts the lane axis in the type,
    # so the index is `memref<1x16xi32>` there and a replica would arrive as
    # `memref<1x8x16xi32>`. Counted together, both paths say five and zero.
    expect_ir={"08-passes/06-torchsim-bufferize.mlir": {
        ("memref<16xi32>", "memref<1x16xi32>"): 2,
        ("memref<16x8xi32>", "memref<1x8x16xi32>"): 0,
        "linalg.broadcast": 0}},
)
