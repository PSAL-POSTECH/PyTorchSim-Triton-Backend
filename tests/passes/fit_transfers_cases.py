"""pytorchsim-fit-transfers-to-descriptor on transfers no Triton kernel reaches: a clamp on an extent-1
axis, a nonzero sram_idx, a one-bank tile or a cut of rank > 4, a cut axis clamped at run time; and a
folded gather's index tile, one entry per position, folded with its data tile.
    python3 tests/passes/fit_transfers_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family import lower_dma  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lib_ir, p11_fit_transfers_to_descriptor  # noqa: E402
from pytorchsim_triton_compiler.dialects.torchsim import TransferOp, operand_number  # noqa: E402

_MOD = """
module {{
  memref.global "private" @spad : memref<{shape}xf32, 1>
  func.func @f(%x: memref<65536xf32>, %lo: index, %hi: index) {{
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %s = memref.get_global @spad : memref<{shape}xf32, 1>
    "torchsim.transfer"(%x, %c0, %s, {sidx}, %c0{clamp}) <{{dma_kind = "{kind}", dram_arg = 0 : i64,
        dram_stride = {dstride}{mask}{cut}, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, {nclamp}, 0, 0, 0>}}>
        : (memref<65536xf32>, index, memref<{shape}xf32, 1>, index, index{clamp_ty}) -> ()
    return
  }}
}}
"""


def _module(shape, dstride, masked=None, sidx="%c0", cut=None, kind="MVIN", bounds="%lo, %hi"):
    m = masked is not None
    entries = ", ".join("array<i64" + (": " + ", ".join(map(str, e)) if e else "") + ">"
                        for e in (cut or []))
    return _MOD.format(shape="x".join(map(str, shape)), dstride=list(dstride), sidx=sidx,
                       cut=f", cut = [{entries}]" if cut else "",
                       kind=kind, clamp=", " + bounds if m else "", clamp_ty=", index, index" if m else "",
                       nclamp=2 if m else 0,
                       mask=f", masked_axes = [{masked}], masked_fill = 0 : i64" if m else "")


def _fit(text):
    from mlir.ir import Module
    module = Module.parse(text)
    before = [o for o in lib_ir.walk(module.operation) if o.name == TransferOp.OPERATION_NAME]
    parent_sidx = before[0].operands[operand_number(before[0], "sram_idx")]
    p11_fit_transfers_to_descriptor.run(module, descriptor_dims=lower_dma.DESCRIPTOR_DIMS)
    assert module.operation.verify()
    after = [o for o in lib_ir.walk(module.operation) if o.name == TransferOp.OPERATION_NAME]
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
        assert "uniform tile" in str(e), e
        return
    raise AssertionError("a uniform tile was peeled")


def _boxes(after):
    from mlir.ir import ArrayAttr
    return [[lib_ir.int_array(a) for a in ArrayAttr(t.attributes["piece_box"])] for t in after]


def cut_fold_carries_the_digits():
    from mlir.ir import AffineAddExpr, AffineConstantExpr, AffineMapAttr
    cut = [[], [0, 32, 64, 32], [], [], []]
    module, after, parent = _fit(_module([4, 2, 2, 2, 8], [64, 32, 16, 8, 1], cut=cut))
    assert [_shape(t) for t in after] == [[4, 64]] * 2, [_shape(t) for t in after]
    assert _boxes(after) == [[[], [0, 32]], [[], [32, 64]]], _boxes(after)
    second = after[1].operands[operand_number(after[1], "dram_idx")].owner
    assert second.name == "affine.apply" and second.operands[0] == parent, second
    expr = AffineAddExpr(AffineMapAttr(second.attributes["map"]).value.results[0])
    assert AffineConstantExpr(expr.rhs).value == -32, expr


def cut_axis_is_never_peeled():
    from mlir.ir import Module
    cut = [[], [0, 2048, 4096, 2048], [], [], []]
    module = Module.parse(_module([4, 2, 2, 2, 8], [8192, 2048, 512, 64, 1], cut=cut))
    op = next(o for o in lib_ir.walk(module.operation) if o.name == TransferOp.OPERATION_NAME)
    tx = p11_fit_transfers_to_descriptor.TransferFacts(op)
    fit = p11_fit_transfers_to_descriptor.choose_fit(tx, lower_dma.DESCRIPTOR_DIMS)
    assert 1 not in fit.peeled and len(fit.peeled) == 1 and fit.dmas == 4, fit
    module, after, _ = _fit(_module([4, 2, 2, 2, 8], [8192, 2048, 512, 64, 1], cut=cut))
    assert _boxes(after) == [[[], [0, 1], [], []], [[], [1, 2], [], []]], _boxes(after)


def _masked_or_none(tx):
    return _masked(tx) if "masked_axes" in tx.attributes else None


def cut_clamp_meets_each_box():
    cut = [[0, 1, 6, 2]]
    module, after, _ = _fit(_module([8], [1], masked=0, cut=cut, bounds="%c0, %c3"))
    assert _boxes(after) == [[[0, 3]], [[3, 8]]], _boxes(after)
    assert [_masked_or_none(t) for t in after] == [None, [0]], [_masked_or_none(t) for t in after]
    module, after, _ = _fit(_module([8], [1], masked=0, cut=cut, bounds="%c0, %c3", kind="MVOUT"))
    assert _boxes(after) == [[[0, 3]]] and _masked_or_none(after[0]) is None, _boxes(after)


def cut_clamp_at_run_time_is_refused():
    try:
        _fit(_module([8], [1], masked=0, cut=[[0, 1, 6, 2]]))
    except NotImplementedError as e:
        assert "runtime bound" in str(e), e
        return
    raise AssertionError("a cut axis clamped at run time was split into static boxes")


_PIECE = """
module {{
  memref.global "private" @spad : memref<8xf32, 1>
  func.func @f(%x: memref<64xf32>) {{
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %s = memref.get_global @spad : memref<8xf32, 1>
    "torchsim.transfer"(%x, %c0, %s, %c0, %c0, %c0, %c3) <{{dma_kind = "MVIN", dram_arg = 0 : i64,
        dram_stride = [1], masked_axes = [0], masked_fill = 0 : i64, piece_box = [array<i64: {box}>],
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}}>
        : (memref<64xf32>, index, memref<8xf32, 1>, index, index, index, index) -> ()
    return
  }}
}}
"""


def verifier_takes_a_boxed_clamp_only_when_it_misses_the_box():
    from mlir.ir import Module
    Module.parse(_PIECE.format(box="3, 8"))
    try:
        Module.parse(_PIECE.format(box="2, 8"))
    except Exception as e:  # noqa: BLE001
        assert "boxed and clamped" in str(e), e
        return
    raise AssertionError("the verifier took a piece whose clamp meets its box")


_GATHER = """
module {
  memref.global "private" @spad : memref<8x2x8x2x8xf32, 1>
  memref.global "private" @idx : memref<8x2x8x2x8xi64, 1>
  func.func @f(%x: memref<9264xf32>) {
    %c0 = arith.constant 0 : index
    %s = memref.get_global @spad : memref<8x2x8x2x8xf32, 1>
    %i = memref.get_global @idx : memref<8x2x8x2x8xi64, 1>
    "torchsim.transfer"(%x, %c0, %s, %c0, %c0, %i) <{dma_kind = "MVIN", dram_arg = 0 : i64,
        dram_stride = [0, 0, 0, 4640, 0], offset_stride = 1 : i64,
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}>
        : (memref<9264xf32>, index, memref<8x2x8x2x8xf32, 1>, index, index, memref<8x2x8x2x8xi64, 1>) -> ()
    return
  }
}
"""


def gather_fold_folds_its_index_tile():
    """A fold of a gather whose index tile holds one entry per position folds the index tile alike, as
    the runtime-attribute pass reads it beside the data tile (depthwise conv2d's stencil under grad)."""
    from mlir.ir import MemRefType
    module, after, _ = _fit(_GATHER)
    assert len(after) == 1, after
    index = MemRefType(after[0].operands[operand_number(after[0], "index")].type)
    assert len(_shape(after[0])) <= 4 and list(index.shape) == _shape(after[0]), (index, _shape(after[0]))


CASES = [peel_keeps_a_clamped_unit_axis, fold_keeps_a_clamped_unit_axis_apart,
         fold_keeps_a_clamped_unit_axis_after_an_unnamed_one,
         peel_offsets_a_nonzero_sram_idx, one_bank_fold_keeps_its_lane_mark,
         one_bank_peel_is_refused, cut_fold_carries_the_digits, cut_axis_is_never_peeled,
         cut_clamp_meets_each_box, cut_clamp_at_run_time_is_refused,
         verifier_takes_a_boxed_clamp_only_when_it_misses_the_box, gather_fold_folds_its_index_tile]


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
