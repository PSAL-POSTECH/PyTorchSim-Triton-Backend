"""torchsim-fit-transfers-to-descriptor on transfers no Triton kernel reaches: a clamp on an extent-1
axis (Triton drops it), a nonzero sram_idx (p01 builds 0) and a one-bank tile of rank > 4.
    python3 tests/passes/fit_transfers_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.lowering import passes  # noqa: E402
from pytorchsim_triton_compiler.dialects.torchsim import TransferOp, operand_number  # noqa: E402

_MOD = """
module {{
  memref.global "private" @spad : memref<{shape}xf32, 1>
  func.func @f(%x: memref<65536xf32>, %lo: index, %hi: index) {{
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %s = memref.get_global @spad : memref<{shape}xf32, 1>
    "torchsim.transfer"(%x, %c0, %s, {sidx}, %c0{clamp}) <{{dma_kind = "MVIN", dram_arg = 0 : i64,
        dram_stride = {dstride}{mask}, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, {nclamp}, 0, 0, 0>}}>
        : (memref<65536xf32>, index, memref<{shape}xf32, 1>, index, index{clamp_ty}) -> ()
    return
  }}
}}
"""


def _module(shape, dstride, masked=None, sidx="%c0"):
    m = masked is not None
    return _MOD.format(shape="x".join(map(str, shape)), dstride=list(dstride), sidx=sidx,
                       clamp=", %lo, %hi" if m else "", clamp_ty=", index, index" if m else "",
                       nclamp=2 if m else 0,
                       mask=f", masked_axes = [{masked}], masked_fill = 0 : i64" if m else "")


def _fit(text):
    from mlir.ir import Module
    module = Module.parse(text)
    before = [o for o in passes.lib_ir.walk(module.operation) if o.name == TransferOp.OPERATION_NAME]
    parent_sidx = before[0].operands[operand_number(before[0], "sram_idx")]
    passes.p10a_fit_transfers_to_descriptor.run(module)
    assert module.operation.verify()
    after = [o for o in passes.lib_ir.walk(module.operation) if o.name == TransferOp.OPERATION_NAME]
    return module, after, parent_sidx


def _shape(tx):
    from mlir.ir import MemRefType
    return list(MemRefType(tx.operands[operand_number(tx, "sram")].type).shape)


def _masked(tx):
    from mlir.ir import ArrayAttr, IntegerAttr
    return [IntegerAttr(a).value for a in ArrayAttr(tx.attributes["masked_axes"])]


def peel_keeps_a_clamped_unit_axis():
    module, after, _ = _fit(_module([4, 1, 2, 2, 2, 8], [4096, 2048, 512, 128, 32, 1], masked=1))
    assert len(after) == 1, after
    shape, masked = _shape(after[0]), _masked(after[0])
    assert len(shape) <= 4 and all(shape[a] == 1 for a in masked) and masked, (shape, masked)


def fold_keeps_a_clamped_unit_axis_apart():
    module, after, _ = _fit(_module([4, 1, 2, 4, 8], [2048, 1024, 256, 32, 2], masked=1))
    shape, masked = _shape(after[0]), _masked(after[0])
    assert len(shape) <= 4 and all(shape[a] == 1 for a in masked) and masked, (shape, masked)


def fold_keeps_a_clamped_unit_axis_after_an_unnamed_one():
    module, after, _ = _fit(_module([4, 1, 1, 2, 4, 8], [256, 64, 64, 32, 8, 1], masked=2))
    assert len(after) == 1, after
    shape, masked = _shape(after[0]), _masked(after[0])
    assert len(shape) <= 4 and all(shape[a] == 1 for a in masked) and masked, (shape, masked)


def peel_offsets_a_nonzero_sram_idx():
    module, after, parent = _fit(_module([4, 2, 2, 2, 8], [4096, 512, 128, 32, 1], sidx="%c3"))
    sidx = after[0].operands[operand_number(after[0], "sram_idx")]
    apply = sidx.owner
    assert apply.name == "affine.apply" and apply.operands[0] == parent, apply


def one_bank_fold_keeps_its_lane_mark():
    module, after, _ = _fit(_module([1, 1, 2, 4, 8, 16], [0, 0, 512, 128, 16, 1]))
    shape = _shape(after[0])
    assert len(shape) <= 4 and shape[0] == 1, shape


def one_bank_peel_is_refused():
    try:
        _fit(_module([1, 2, 2, 2, 2, 8], [0, 1024, 256, 64, 16, 1]))
    except NotImplementedError as e:
        assert "one-bank tile" in str(e), e
        return
    raise AssertionError("a one-bank tile was peeled")


CASES = [peel_keeps_a_clamped_unit_axis, fold_keeps_a_clamped_unit_axis_apart,
         fold_keeps_a_clamped_unit_axis_after_an_unnamed_one,
         peel_offsets_a_nonzero_sram_idx, one_bank_fold_keeps_its_lane_mark,
         one_bank_peel_is_refused]


def main():
    from mlir.ir import Context, Location
    bad = 0
    for case in CASES:
        ctx = Context()
        ctx.allow_unregistered_dialects = True
        with ctx, Location.unknown():
            try:
                case()
                print(f"ok    {case.__name__}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
