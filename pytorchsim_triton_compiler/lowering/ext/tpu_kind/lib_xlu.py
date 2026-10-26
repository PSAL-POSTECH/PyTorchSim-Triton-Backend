"""lib_xlu: the cross-lane unit (tpu::Xlu) -- the push/pop rows its 32-bit queue moves for what lib_xlu_plan
plans (each XLU op and the rounds it runs to carry a correspondence); no SLOT, so not a pass."""

from .lib_xlu_plan import (  # noqa: F401 -- what lib_xlu emits, planned there
    MAPPED, RPU_ARBITRARY, RPU_BYPASS, RPU_NONE, RPU_REPLICATE, XLU_CROSSING,
    XLU_FOLDS, XLU_FOLDS_DROPPING_NAN, XLU_GATHER, XLU_OPS, XLU_PERMUTE, _get_lane_runs, _laps, as_signed_imm,
    choose_xlu_op, crossing_passes, crossings_of, emitter_of, fields, gather_rounds,
    is_a_load, permute_groups, permute_runs, plan_of_xlu_op, rounds_of_pairs, rounds_of_xlu_op,
    rows_per_vector, simm5)
from ...passes.lib_ir import build_trip_loop, const_index, elem_bits, transfer_read, transfer_write
from ...passes.lib_vcix import bitcast, i64 as a64, vcix, word_vector

import mlir.ir as ir  # noqa: E402


XLU_PUSH_OPCODE, XLU_POP_OPCODE = 3, 1

# xlu_set_pattern (sf.vc.ivv 3): each lane's source lanes for an ARBITRARY shuffle before the crossing, from vs2, which
# every push after it takes as it issues, as the v6e's vset.pattern.permute.
XLU_SET_PATTERN_OPCODE = 3

# xlu_push_reduce (sf.vc.ivv 2): a tile folded across the lanes, its field the fold (0 max, 1 sum, 2 min).
XLU_REDUCE_OPCODE = 2


SF_VC_IVV_SE = "llvm.riscv.sf.vc.ivv.se"


def round_rows(r):
    """(source row, destination row) of a permute round of either form."""
    return (r[1], r[2]) if r[0] == MAPPED else (r[0], r[3])


def lane_runs(src_of):
    """The runs of lanes `(lo, hi, step, mul, add, div)` a MAPPED round's lanes fall into, each lane
    `l` reading lane `(l * mul + add) / div`."""
    return _get_lane_runs({lane: s for lane, s in enumerate(src_of) if s >= 0})


def reaches_every_lane(pairs, lanes):
    """Whether these (from, to) pairs carry each destination row from one source element into all
    `lanes` lanes: EVERY_LANE, read off the correspondence a move carries, not its buffer."""
    rows = {}
    for f, (lane, deep) in pairs or ():
        srcs, got = rows.setdefault(deep, (set(), set()))
        srcs.add(f)
        got.add(lane)
    return bool(rows) and all(len(s) == 1 and g == set(range(lanes)) for s, g in rows.values())


def element_from_words(word, h, bits, elem_ty, parked=None):
    from mlir.dialects import arith
    if bits < 32:
        return arith.TruncIOp(elem_ty, word).result
    if bits == 32:
        return word
    wide = arith.ExtUIOp(elem_ty, word).result
    if h == 0:
        return wide
    shift = arith.ConstantOp(
        elem_ty, ir.DenseElementsAttr.get_splat(
            elem_ty, ir.IntegerAttr.get(
                ir.VectorType(elem_ty).element_type, 32))).result
    return arith.OrIOp(parked, arith.ShLIOp(wide, shift).result).result


def crossing_words(v, bits):
    from mlir.dialects import arith
    if bits == 32:
        return [v]
    vt = ir.VectorType(v.type)
    word = ir.VectorType.get(list(vt.shape), ir.IntegerType.get_signless(32))
    if bits < 32:
        return [arith.ExtUIOp(word, v).result]
    shift = arith.ConstantOp(
        vt, ir.DenseElementsAttr.get_splat(
            vt, ir.IntegerAttr.get(vt.element_type, 32))).result
    return [arith.TruncIOp(word, v).result,
            arith.TruncIOp(word, arith.ShRUIOp(v, shift).result).result]


def row_at(base, i, step):
    from mlir.dialects import arith
    if i is None:
        return const_index(base) if isinstance(base, int) else base
    off = i if step == 1 else arith.MulIOp(i, const_index(step)).result
    if isinstance(base, int):
        return off if not base else arith.AddIOp(off, const_index(base)).result
    return arith.AddIOp(base, off).result


def build_row_loop(n, body):
    if n == 1:
        body(None)
        return
    iv, inside = build_trip_loop(ir.InsertionPoint.current, n)
    with inside:
        body(iv)


def crossing_bits(elt):
    return max(elem_bits(elt), 32)


def rows_fit(bank, at, step, rows, nr):
    """Do the `rows // nr` accesses of `nr` rows from row `at`, `nr * step` apart, stay inside the
    lane's `[1, depth]` bank? Only a static start and depth prove it; else the access keeps its
    bounds check."""
    depth = ir.MemRefType(bank.type).shape[1]
    if not isinstance(at, int) or ir.ShapedType.is_dynamic_size(depth):
        return False
    return 0 <= at and at + (rows // nr - 1) * nr * step + nr <= depth


def build_push_rows(bank, at, step, rows, in_ty, rvl, h, bits, opcode, imm, pad):
    """`rows` rows of `bank` from row `at`, `step` apart, pushed `in_ty` at a time: word `h`."""
    nr = ir.VectorType(in_ty).shape[0]
    fits = rows_fit(bank, at, step, rows, nr)

    def push(i):
        v = transfer_read(in_ty, bank, [const_index(0), row_at(at, i, nr * step)], pad, fits)
        vcix("vcix.iv", [crossing_words(as_type(v, word_vector(in_ty)), bits)[h], rvl], [],
             {"opcode": a64(opcode), "imm": a64(imm), "rd": a64(0)})
    build_row_loop(rows // nr, push)


def build_pop_rows(bank, at, step, rows, out_ty, rvl, h, bits, crossings, opcode, imm, pad):
    """`rows` rows popped into `bank` from row `at`, `step` apart, `out_ty` at a time: word `h`,
    joined to the words before it."""
    nr = ir.VectorType(out_ty).shape[0]
    cross_ty = ir.VectorType.get([nr], ir.IntegerType.get_signless(32))
    fits = rows_fit(bank, at, step, rows, nr)

    def pop(i):
        got = vcix("vcix.v.i", [rvl], [cross_ty],
                   {"opcode": a64(opcode), "imm": a64(imm), "rs2": a64(0)}).results[0]
        row = row_at(at, i, nr * step)
        v = element_from_words(
            got, h, bits, word_vector(out_ty),
            None if crossings == 1 or not h else as_type(
                transfer_read(out_ty, bank, [const_index(0), row], pad, fits), word_vector(out_ty)))
        transfer_write(as_type(v, out_ty), bank, [const_index(0), row], fits)
    build_row_loop(rows // nr, pop)


def as_type(v, ty):
    return v if v.type == ty else bitcast(v, ty)


def call_intrinsic(name, operands, result_tys=()):
    """Emit one `llvm.call_intrinsic`, for an intrinsic THE VCIX DIALECT DOES NOT NAME."""
    from mlir.dialects import llvm
    tys = list(result_tys)
    return llvm.CallIntrinsicOp(tys[0] if tys else None, ir.StringAttr.get(name),
                                list(operands), [], [])
