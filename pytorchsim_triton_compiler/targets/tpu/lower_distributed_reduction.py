"""lower_distributed_reduction: every `pytorchsim.distributed_reduction` onto the cross-lane unit
(tpu::Xlu): each lane pushes its run of the source with xlu_push_reduce, and the pops hand every lane
each column's lane reduce."""

from .pytorchsim_tpu import XluPopOp, XluPushReduceOp
from ...passes.utils.bank_layout import (
    DISTRIBUTED_REDUCTION_OP, REDUCTION_BY, bank_depth, bank_view, view_lanes)
from ...passes.utils.ir_utils import const_index, transfer_read, transfer_write, walk_ops, zero_of
from ...passes.utils.vcix import bitcast, i64 as a64
from .units import lane_number
from .xlu import XLU_FOLDS, XLU_FOLDS_DROPPING_NAN, xlu_fold_type
from .xlu_move import refuse

import mlir.ir as ir  # noqa: E402
from ...passes import pass_name  # noqa: E402

NAME = pass_name(__name__)

FORBIDS = (DISTRIBUTED_REDUCTION_OP,)

MARKERS = FORBIDS


def run(module, *, subtarget, **_):
    """Every lane reduce lowered where it stands."""
    for reduction in _lane_reduces_in(module):
        _xlu_lane_reduce(reduction, subtarget.vector_lanes, subtarget.vlen_bits)
    return module


def _lane_reduces_in(module):
    return [o.operation for region in module.operation.regions for b in region.blocks
            for o in walk_ops(b) if o.operation.name == DISTRIBUTED_REDUCTION_OP]


def _xlu_lane_reduce(move, lanes, vlen):
    """A lane reduce in one tile: each lane pushes its whole run of the source with xlu_push_reduce, a
    register group a push, the pops give every lane each column's lane reduce, and the destination, in one
    bank, takes its combine with what it held. A lane the source does not ride pushes the combiner's
    identity, so it adds nothing. A push takes one register group of f32, or of i64 when the lane's
    number masks it; the pushes before the first pop are one tile, whatever their number."""
    from mlir.dialects import arith
    by = ir.StringAttr(move.attributes[REDUCTION_BY]).value
    src, dst = move.operands[0], move.operands[1]
    elt = ir.MemRefType(src.type).element_type
    used = view_lanes(src)
    if by not in XLU_FOLDS or elt != xlu_fold_type():
        return refuse(DISTRIBUTED_REDUCTION_OP, move,
                      f"the unit lane-reduces {xlu_fold_type()} by {sorted(XLU_FOLDS)}, not {elt} by {by}")
    if used > lanes:
        return refuse(DISTRIBUTED_REDUCTION_OP, move,
                      f"the source rides {used} lanes, more than the {lanes} the machine has: "
                      "materialize_layouts reduces the laps in each lane first")
    depth = bank_depth(src, lanes)
    if depth is None or bank_depth(dst, lanes) != depth:
        return refuse(DISTRIBUTED_REDUCTION_OP, move, "a lane's run of the source is not the destination's")
    i64, i32 = ir.IntegerType.get_signless(64), ir.IntegerType.get_signless(32)
    ident = {"max": float("-inf"), "sum": 0.0, "min": float("inf")}[XLU_FOLDS[by]]
    combine = {"arith.maximumf": arith.MaximumFOp, "arith.maxnumf": arith.MaxNumFOp,
               "arith.addf": arith.AddFOp, "arith.minimumf": arith.MinimumFOp,
               "arith.minnumf": arith.MinNumFOp}[by]
    step = min(depth, 8 * vlen // (32 if used == lanes else 64))
    chunks = [(at, min(step, depth - at)) for at in range(0, depth, step)]

    def const(t, v):
        return arith.ConstantOp(t, v).result

    with ir.InsertionPoint(move):
        src_bank, dst_bank = bank_view(src, lanes), bank_view(dst, lanes)
        for at, w in chunks:
            ty, words = ir.VectorType.get([w], elt), ir.VectorType.get([w], i32)
            rvl = const(i64, a64(w))
            v = transfer_read(ty, src_bank, [const_index(0), const_index(at)], zero_of(elt), True)
            if used < lanes:
                wide = ir.VectorType.get([w], i64)
                mine = arith.CmpIOp(arith.CmpIPredicate.slt, lane_number(rvl, w),
                                    const(wide, ir.DenseElementsAttr.get_splat(wide, a64(used)))).result
                v = arith.SelectOp(mine, v, const(ty, ir.DenseElementsAttr.get_splat(
                    ty, ir.FloatAttr.get(elt, ident)))).result
            if by in XLU_FOLDS_DROPPING_NAN:
                nan = arith.CmpFOp(arith.CmpFPredicate.UNO, v, v).result
                v = arith.SelectOp(nan, const(ty, ir.DenseElementsAttr.get_splat(
                    ty, ir.FloatAttr.get(elt, ident))), v).result
            XluPushReduceOp(bitcast(v, words), rvl, XLU_FOLDS[by])
        for at, w in chunks:
            ty, words = ir.VectorType.get([w], elt), ir.VectorType.get([w], i32)
            got = XluPopOp(words, const(i64, a64(w)), "reduce").result
            there = transfer_read(ty, dst_bank, [const_index(0), const_index(at)], zero_of(elt), True)
            transfer_write(combine(bitcast(got, ty), there).result, dst_bank,
                           [const_index(0), const_index(at)], True)
    move.erase()
    return True
