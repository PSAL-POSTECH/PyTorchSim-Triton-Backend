"""tritonpytorchsim-lower-cross-lane-tpuv6e: every marked `torchsim.crosslane_move` onto vcix_tpuv6e's
cross-lane unit (tpu::Xlu, libtpuv6e.so) -- its pushes and pops, as the rounds written on it."""

from ...passes.lib_bank_layout import bank_depth, bank_view, is_uniform, lane_extent, view_lanes
from ...passes.lib_cross_lane import (CROSS_LANE_ATTR, CROSS_LANE_KIND, CROSS_LANE_OP,
                                      CROSS_LANE_ROUNDS, FOLD, FOLD_BY)
from ...passes.lib_ir import build_trip_loop, const_index, elem_bits, transfer_read, transfer_write
from ...passes.lib_ir import walk_ops, zero_of
from ...passes.lib_vcix import bitcast, i64 as a64, vcix, word_vector
from ...passes.lower import LOWER_CROSS_LANE
from .lib_units import lane_number
from .lib_xlu import (XLU_FOLDS, XLU_FOLDS_DROPPING_NAN, XLU_GATHER, XLU_OPS, XLU_PERMUTE, XLU_POP_OPCODE, XLU_PUSH_OPCODE,
                      XLU_REDUCE_OPCODE,
                      as_signed_imm, build_pop_rows, build_push_rows, build_row_loop,
                      crossing_bits, crossing_words, element_from_words, emitter_of, is_a_load,
                      crossings_of, permute_runs, row_at, rows_per_vector)

import mlir.ir as ir  # noqa: E402

NAME = "tritonpytorchsim-lower-cross-lane-tpuv6e"

REQUIRES = ("pytorchsim_tpu_family",)

SLOT = LOWER_CROSS_LANE.name

FORBIDS = LOWER_CROSS_LANE.lowers

MARKERS = FORBIDS

_SF_VC_IVV_SE = "llvm.riscv.sf.vc.ivv.se"

QUEUE_KEY = "tpu_xlu_queue_entries"

# why a permute's pops each wait on their own push, on the function that holds it
XLU_PIPELINE = "torchsim.xlu_pipeline"


def run(module, *, target, **_):
    """Every marked cross-lane move lowered where it stands."""
    queue = int(target.vcix[QUEUE_KEY]) if QUEUE_KEY in target.vcix else None
    _lower_every_xlu_move(module, target.vector_lanes, target.vlen_bits, queue)
    return module


def _lower_every_xlu_move(module, lanes, vlen, queue=None):
    folds = sum(_xlu_fold(move, lanes, vlen) for move in _folds_in(module))
    return folds + sum(_lower_one_xlu_move(move, name, lanes, vlen, queue)
                       for move, name in _xlu_moves_in(module))


def _folds_in(module):
    return [o.operation for region in module.operation.regions for b in region.blocks
            for o in walk_ops(b)
            if o.operation.name == CROSS_LANE_OP and CROSS_LANE_KIND in o.operation.attributes
            and ir.StringAttr(o.operation.attributes[CROSS_LANE_KIND]).value == FOLD]


def _xlu_fold(move, lanes, vlen):
    """A fold across the lanes in one tile: each lane pushes its whole run of the source with
    xlu_push_reduce, a register group a push, the pops give every lane each column's fold over the
    lanes, and the destination, in one bank, takes its combine with what it held. A lane the source does not ride pushes the
    fold's identity, so it adds nothing."""
    from mlir.dialects import arith
    by = ir.StringAttr(move.attributes[FOLD_BY]).value
    src, dst = _xlu_move_ends(move)
    elt = ir.MemRefType(src.type).element_type
    used = view_lanes(src)
    if by not in XLU_FOLDS or str(elt) != "f32":
        return _refuse_xlu_move(FOLD, move, f"the unit folds f32 by {sorted(XLU_FOLDS)}, not {elt} by {by}")
    if used > lanes:
        return _refuse_xlu_move(FOLD, move, f"the source rides {used} lanes, more than the {lanes} the "
                                "machine has: materialize_layouts reduces the laps in each lane first")
    depth = bank_depth(src, lanes)
    if depth is None or bank_depth(dst, lanes) != depth:
        return _refuse_xlu_move(FOLD, move, "a lane's run of the source is not the destination's")
    i64, i32 = ir.IntegerType.get_signless(64), ir.IntegerType.get_signless(32)
    ident = {0: float("-inf"), 1: 0.0, 2: float("inf")}[XLU_FOLDS[by]]
    combine = {"arith.maximumf": arith.MaximumFOp, "arith.maxnumf": arith.MaxNumFOp,
               "arith.addf": arith.AddFOp, "arith.minimumf": arith.MinimumFOp,
               "arith.minnumf": arith.MinNumFOp}[by]
    # a push takes one register group of f32, or of i64 when the lane's number masks it; the pushes
    # before the first pop are one tile, whatever their number
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
            call_intrinsic(_SF_VC_IVV_SE,
                           [const(i64, a64(XLU_REDUCE_OPCODE)),
                            const(words, ir.DenseElementsAttr.get_splat(words, ir.IntegerAttr.get(i32, 0))),
                            bitcast(v, words), const(i64, a64(XLU_FOLDS[by])), rvl])
        for at, w in chunks:
            ty, words = ir.VectorType.get([w], elt), ir.VectorType.get([w], i32)
            got = vcix("vcix.v.i", [const(i64, a64(w))], [words],
                       {"opcode": a64(XLU_POP_OPCODE), "imm": a64(0), "rs2": a64(0)}).results[0]
            there = transfer_read(ty, dst_bank, [const_index(0), const_index(at)], zero_of(elt), True)
            transfer_write(combine(bitcast(got, ty), there).result, dst_bank,
                           [const_index(0), const_index(at)], True)
    move.erase()
    return True


def _lower_one_xlu_move(move, name, lanes, vlen, queue=None):
    kind = emitter_of(name)
    if kind == XLU_GATHER:
        return _xlu_all_gather(move, lanes, vlen)
    if kind == XLU_PERMUTE:
        return _xlu_permute(move, lanes, vlen, queue)
    src, dst = _xlu_move_ends(move)
    if not _xlu_end_is_contiguous(src, lanes) or not _xlu_end_is_contiguous(
            dst, lanes):
        return _refuse_xlu_move(
            name, move, "a lane's rows of one end of this move are not evenly "
            "spaced, so the bank has no single step to walk")
    written = _get_written_rounds(move)
    if written is None:
        return _refuse_xlu_move(name, move, "no passes are written on it")
    return _xlu_row_passes(move, lanes, vlen, name, written)


def _xlu_all_gather(move, lanes, vlen):
    xlu_op = ir.IntegerAttr(move.attributes[CROSS_LANE_ATTR]).value
    from mlir.dialects import arith
    src, dst = _xlu_move_ends(move)
    bits = elem_bits(ir.MemRefType(src.type).element_type)
    crossings, why = crossings_of(bits)
    if not crossings:
        return _refuse_xlu_move(xlu_op, move, why)
    got = _xlu_all_gather_shape(src, dst, lanes, vlen)
    if got is None:
        return _refuse_xlu_move(xlu_op, move, "the shapes are not the unit's")
    n, per, chunk, elt = got
    plan = _get_written_rounds(move)
    if plan is None:
        return _refuse_xlu_move(
            xlu_op, move, "a destination lane does not spell the source in "
            "the order one pass per value it holds delivers")
    i64 = ir.IntegerType.get_signless(64)


    with ir.InsertionPoint(move):
        src_bank = bank_view(src, lanes)
        dst_bank = bank_view(dst, lanes)
    rounds = ir.MemRefType(src_bank.type).shape[1]
    trips = len(plan)
    step = plan[0][2]
    at = per if step == 1 else 1
    if plan != [(i % rounds, i // rounds * rounds * per + i % rounds * at, step)
                for i in range(trips)]:
        return _refuse_xlu_move(
            xlu_op, move, f"the written passes are not {rounds} values per lane "
            f"gathered {per} at a time, lap after lap")
    if step != 1:
        chunk = 1
    lap = trips * per
    laps, rest = divmod(ir.MemRefType(dst_bank.type).shape[1], lap)
    if rest or not laps:
        return _refuse_xlu_move(
            xlu_op, move, "a lane would not end up with the whole tile")

    g_push, g_pop, g_imm = _xlu_encoding(xlu_op, move)
    # every pass reads depth j < rounds of the source and lands per // chunk chunks from its
    # offset, `chunk * step` apart: in the destination when the last one ends inside it
    depth = ir.MemRefType(dst_bank.type).shape[1]
    lands_inside = all(off + (per // chunk - 1) * chunk * st + chunk <= depth
                       for _, off, st in plan)
    one_ty = ir.VectorType.get([1], elt)
    chunk_ty = ir.VectorType.get([chunk], elt)
    one_word, chunk_word = word_vector(one_ty), word_vector(chunk_ty)

    def cast(v, ty):
        return v if v.type == ty else bitcast(v, ty)

    with ir.InsertionPoint(move):
        rvl1 = arith.ConstantOp(i64, a64(1)).result
        rvlc = arith.ConstantOp(i64, a64(chunk)).result
        cross_chunk = ir.VectorType.get([chunk],
                                        ir.IntegerType.get_signless(32))

        def landed(got, h, at):
            """One crossing's answer into the destination."""
            v = element_from_words(
                got, h, bits, chunk_word,
                None if crossings == 1 or not h else cast(transfer_read(
                    chunk_ty, dst_bank, [const_index(0), at], zero_of(elt), lands_inside),
                    chunk_word))
            transfer_write(cast(v, chunk_ty), dst_bank, [const_index(0), at], lands_inside)

        def at_of(base, k):
            """The destination depth `k` past `base`, where None is the start."""
            if base is None:
                return const_index(k)
            return base if not k else arith.AddIOp(
                base, const_index(k)).result

        def one_pass(j, base):
            """One pass: every lane pushes the value it holds at depth `j` and takes the whole
            tile's row back."""
            v = cast(transfer_read(one_ty, src_bank, [const_index(0), j],
                                    zero_of(elt), True), one_word)
            for h, part in enumerate(crossing_words(v, bits)):
                vcix("vcix.iv", [part, rvl1], [],
                      {"opcode": a64(g_push), "imm": a64(g_imm), "rd": a64(0)})

                def pop(i, h=h):
                    got = vcix("vcix.v.i", [rvlc], [cross_chunk],
                                {"opcode": a64(g_pop), "imm": a64(g_imm),
                                 "rs2": a64(0)}).results[0]
                    landed(got, h, row_at(0 if base is None else base,
                                               i, chunk * step))
                build_row_loop(per // chunk, pop)

        if trips == 1:
            one_pass(const_index(0), None)
        else:
            iv, inside = build_trip_loop(ir.InsertionPoint(move), trips)
            with inside:
                j = (const_index(0) if rounds == 1 else iv if trips == rounds else
                     arith.RemUIOp(iv, const_index(rounds)).result)
                if step == 1:
                    base = arith.MulIOp(iv, const_index(per)).result
                elif trips == rounds:
                    base = j
                else:
                    base = arith.AddIOp(arith.MulIOp(
                        arith.DivUIOp(iv, const_index(rounds)).result,
                        const_index(rounds * per)).result, j).result
                one_pass(j, base)
        if laps > 1:
            _copy_first_lap(dst_bank, lap, laps, elt, vlen)
    move.erase()
    return True


def _copy_first_lap(bank, lap, laps, elt, vlen):
    """Every lane-full of a gathered destination past the first is the first again: each lane
    copies its own first `lap` rows `laps - 1` times deeper, a register group at a time; `w`
    divides `lap`, so no access leaves the bank."""
    import math
    from ....machine import ISA_MAX_LMUL
    w = math.gcd(lap, ISA_MAX_LMUL * vlen // crossing_bits(elt))
    ty = ir.VectorType.get([w], elt)

    def rows(i):
        v = transfer_read(ty, bank, [const_index(0), row_at(0, i, w)], zero_of(elt), True)
        for k in range(1, laps):
            transfer_write(v, bank, [const_index(0), row_at(k * lap, i, w)], True)
    build_row_loop(lap // w, rows)


def _xlu_move_ends(move):
    return move.operands[0], move.operands[1]


def _refuse_xlu_move(name, op, why):
    raise NotImplementedError(
        f"{CROSS_LANE_ATTR} = '{name}' cannot be lowered here ({why}); "
        f"materialize_layouts must not mint one this cannot take: {op}")


def _xlu_all_gather_shape(src, dst, lanes, vlen):
    elt = ir.MemRefType(src.type).element_type
    n = None if is_uniform(src) else view_lanes(src)
    if n is None or (n > lanes and n % lanes):
        return None
    if lane_extent(dst) is None:
        return None
    per = min(n, lanes)
    chunk = min(per, vlen // 32)
    if per % chunk:
        return None
    return n, per, chunk, elt


def _get_written_rounds(op):
    if CROSS_LANE_ROUNDS not in op.attributes:
        return None
    return [tuple(ir.DenseI64ArrayAttr(r))
            for r in ir.ArrayAttr(op.attributes[CROSS_LANE_ROUNDS])]


def _xlu_encoding(xlu_op, op):
    if is_a_load(xlu_op):
        raise NotImplementedError(
            f"{CROSS_LANE_ATTR} = {xlu_op} has pre = 3, which names no RPU mode -- "
            f"it is a pattern load and not a pass the unit can run: {op}")
    if xlu_op not in XLU_OPS:
        raise NotImplementedError(
            f"{CROSS_LANE_ATTR} = {xlu_op} is not a pass this unit answers. SIMM5 0 "
            f"asks it for no crossing at all, which is a trap: {op}")
    return XLU_PUSH_OPCODE, XLU_POP_OPCODE, as_signed_imm(xlu_op)


def _xlu_permute(move, lanes, vlen, queue=None):
    """The move's rounds, each a push of rows and the pop that answers it, every push before the
    first pop: the unit's input queue drains into its stages with no pop, and it answers in order.
    With no queue stated each pop follows its push, and the function says why on XLU_PIPELINE."""
    xlu_op = ir.IntegerAttr(move.attributes[CROSS_LANE_ATTR]).value
    from mlir.dialects import arith
    push_op, pop_op, simm5 = _xlu_encoding(xlu_op, move)
    src, dst = _xlu_move_ends(move)
    if not _xlu_end_is_contiguous(src, lanes) or not _xlu_end_is_contiguous(
            dst, lanes):
        return _refuse_xlu_move(
            xlu_op, move, "a lane's rows of one end are not evenly spaced")
    elt = ir.MemRefType(src.type).element_type
    bits = elem_bits(elt)
    crossings, why = crossings_of(bits)
    if not crossings:
        return _refuse_xlu_move(xlu_op, move, why)
    i64 = ir.IntegerType.get_signless(64)


    with ir.InsertionPoint(move):
        src_bank = bank_view(src, lanes)
        dst_bank = bank_view(dst, lanes)
    depth = ir.MemRefType(dst_bank.type).shape[1]
    row = min(view_lanes(dst) or lanes, lanes)
    rounds = _get_written_rounds(move)
    if rounds is None or any(not 0 <= r[3] < depth for r in rounds):
        return _refuse_xlu_move(
            xlu_op, move, "no rounds, or a round writes a destination row the bank "
            "does not have")
    src_depth = ir.MemRefType(src_bank.type).shape[1]
    if any(not 0 <= r[0] < src_depth for r in rounds):
        return _refuse_xlu_move(
            xlu_op, move, "a round reads a source row the bank does not have")

    i32_ty = ir.IntegerType.get_signless(32)

    def cast(v, ty):
        return v if v.type == ty else bitcast(v, ty)

    def rvl_of(w):
        return arith.ConstantOp(i64, a64(w)).result

    def splat(k, w):
        ty = ir.VectorType.get([w], i64)
        return arith.ConstantOp(
            ty, ir.DenseElementsAttr.get_splat(ty, a64(k))).result

    from mlir.dialects import vector
    with ir.InsertionPoint(move):
        rvl1 = rvl_of(1)
        pad = zero_of(elt)
        family, code = rvl_of(push_op), rvl_of(simm5)
        at_width = {}

        def _get_lane_number(w):
            if w not in at_width:
                at_width[w] = lane_number(rvl1 if w == 1 else rvl_of(w), w)
            return at_width[w]

        def vec(k, w):
            """`k` in each of `w` rows: a constant, or an i64 a sweep made."""
            if isinstance(k, int):
                return splat(k, w)
            return vector.BroadcastOp(ir.VectorType.get([w], i64), k).result

        def owns(lo, hi, every, w):
            """The lanes this round writes -- lo, lo + every, ... -- in each of `w` rows."""
            lane = at_width[w]
            keep = arith.CmpIOp(arith.CmpIPredicate.slt, lane, vec(hi, w)).result
            if not (isinstance(lo, int) and not lo):
                keep = arith.AndIOp(keep, arith.CmpIOp(
                    arith.CmpIPredicate.sge, lane, vec(lo, w)).result).result
            if every == 1:
                return keep
            off = (lane if isinstance(lo, int) and not lo
                   else arith.SubIOp(lane, vec(lo, w)).result)
            return arith.AndIOp(keep, arith.CmpIOp(
                arith.CmpIPredicate.eq, arith.RemSIOp(off, splat(every, w)).result,
                splat(0, w)).result).result

        def one_round(mul, add, div, lo, hi, every, part, si, di, w, h):
            """`w` rows from source row `si` and the pattern they share, in ONE push, and the pop
            that answers them into row `di` -- the push, the pop or both, as `stage` says."""
            if stage[0] != "pop":
                push_round(mul, add, div, w, si, h)
            if stage[0] != "push":
                pop_round(lo, hi, every, part, di, w, h)

        def push_round(mul, add, div, w, si, h):
            idx = at_width[w]
            if mul != 1:
                idx = arith.MulIOp(idx, splat(mul, w)).result
            if not isinstance(add, int) or add:
                idx = arith.AddIOp(idx, vec(add, w)).result
            if div != 1:
                idx = arith.DivSIOp(idx, splat(div, w)).result
            wide = ir.VectorType.get([w], elt)
            wide_w = word_vector(wide)
            wide_x = ir.VectorType.get([w], i32_ty)
            rvl = rvl1 if w == 1 else rvl_of(w)
            v = transfer_read(wide, src_bank, [const_index(0), si], pad, True)
            call_intrinsic(_SF_VC_IVV_SE,
                           [family, arith.TruncIOp(wide_x, idx).result,
                            crossing_words(cast(v, wide_w), bits)[h],
                            code, rvl])

        def pop_round(lo, hi, every, part, di, w, h):
            wide = ir.VectorType.get([w], elt)
            wide_w = word_vector(wide)
            wide_x = ir.VectorType.get([w], i32_ty)
            rvl = rvl1 if w == 1 else rvl_of(w)
            got = vcix("vcix.v.i", [rvl], [wide_x],
                       {"opcode": a64(pop_op), "imm": a64(simm5),
                        "rs2": a64(0)}).results[0]
            there = (cast(transfer_read(wide, dst_bank, [const_index(0), di],
                                        pad, True), wide_w) if h else None)
            got = element_from_words(got, h, bits, wide_w, there)
            transfer_write(cast(got, wide), dst_bank, [const_index(0), di], True,
                           owns(lo, hi, every, w) if part else None)

        per_reg = vlen // crossing_bits(elt)
        plan = [((mul, div, every, ss, ds, n, w, part), (add, lo, hi, s0, d0))
                for (mul, add, lo, hi, div, every), (s0, ss, d0, ds, n), w, part
                in permute_runs(rounds, row, per_reg)]
        for (*_, n, w, _part), _start in plan:
            for x in {w, n % w} - {0}:
                _get_lane_number(x)
        def sweep_at(g, base, k):
            """`base + g * k` for sweep `g` -- the int itself outside a sweep, so a round that is
            not one emits exactly what it did."""
            if g is None:
                return base
            if not k:
                return const_index(base) if isinstance(base, int) else base
            return row_at(base, g, k)

        def i64_at(g, base, k):
            """`base + g * k` for sweep `g` as an i64 made inside the loop -- the pattern's `add`
            and the round's lanes."""
            if g is None or not k:
                return base
            got = arith.IndexCastOp(i64, g).result
            if k != 1:
                got = arith.MulIOp(got, rvl_of(k)).result
            if not isinstance(base, int):
                return arith.AddIOp(got, base).result
            return arith.AddIOp(got, rvl_of(base)).result if base else got

        def past(base, k):
            """Row `base + k`, `base` an int or a value."""
            if isinstance(base, int):
                return const_index(base + k)
            return base if not k else arith.AddIOp(base, const_index(k)).result

        def emit_rows(shape, now, h):
            """One entry's rows at the starts `now`, a vector of rows per round where they run on,
            and the rows left over."""
            mul, div, every, ss, ds, n, w, part = shape
            add, lo, hi, si, di = now

            def rows(i):
                one_round(mul, add, div, lo, hi, every, part, row_at(si, i, w * ss),
                          row_at(di, i, w * ds), w, h)
            build_row_loop(n // w, rows)
            if n % w:
                k = n - n % w
                one_round(mul, add, div, lo, hi, every, part, past(si, k * ss),
                          past(di, k * ds), n % w, h)

        def nest(shape, now, dims, h):
            """The loops of a sweep, outer first, each moving the starts by its steps."""
            if not dims:
                return emit_rows(shape, now, h)
            (m, steps), inner = dims[0], dims[1:]

            def sweep(g):
                nest(shape, [i64_at(g, x, k) for x, k in zip(now[:3], steps[:3])]
                     + [sweep_at(g, x, k) for x, k in zip(now[3:], steps[3:])], inner, h)
            build_row_loop(m, sweep)

        sweeps = _get_sweeps(plan)
        why = f"the target states no {QUEUE_KEY}" if queue is None else None
        stages = ("both",) if why else ("push", "pop")
        if why:
            func = move.operation.parent
            while func is not None and func.name != "func.func":
                func = func.parent
            if func is not None:
                func.attributes[XLU_PIPELINE] = ir.StringAttr.get(why)
        stage = [None]
        for st in stages:
            stage[0] = st
            for h in range(crossings):
                for shape, start, dims in sweeps:
                    nest(shape, list(start), dims or ((1, (0,) * len(start)),), h)

    move.erase()
    return True


def _xlu_end_is_contiguous(v, lanes):
    return bank_depth(v, lanes) is not None


def _get_sweeps(plan):
    entries = [(shape, tuple(start), ()) for shape, start in plan]
    while True:
        groups = []
        for shape, start, dims in entries:
            if groups and groups[-1][0] == shape and groups[-1][2] == dims:
                g = groups[-1]
                steps = g[4] if g[3] > 1 else tuple(b - a for a, b in zip(g[1], start))
                if start == tuple(a + k * g[3] for a, k in zip(g[1], steps)):
                    g[3], g[4] = g[3] + 1, steps
                    continue
            groups.append([shape, start, dims, 1, None])
        folded = [(sh, st, d if m == 1 else ((m, k),) + d) for sh, st, d, m, k in groups]
        if len(folded) == len(entries):
            return folded
        entries = folded


def _xlu_row_passes(move, lanes, vlen, xlu_op, passes):
    from mlir.dialects import arith
    src, dst = _xlu_move_ends(move)
    elt = ir.MemRefType(src.type).element_type
    bits = elem_bits(elt)
    crossings, why = crossings_of(bits)
    if not crossings:
        return _refuse_xlu_move(xlu_op, move, why)
    push_op, pop_op, simm5 = _xlu_encoding(xlu_op, move)
    i64 = ir.IntegerType.get_signless(64)
    per_reg = vlen // crossing_bits(elt)

    with ir.InsertionPoint(move):
        src_bank = bank_view(src, lanes)
        dst_bank = bank_view(dst, lanes)
        pad = None
        for s_at, s_step, d_at, d_step, rows, wide in passes:
            nr_in = rows_per_vector(s_step, rows, per_reg)
            nr_out = rows_per_vector(d_step, wide, per_reg)
            if rows % nr_in or wide % nr_out:
                return _refuse_xlu_move(
                    xlu_op, move, f"{rows} in and {wide} out per lane are not a "
                    f"whole number of {nr_in}- and {nr_out}-wide registers")
            in_ty = ir.VectorType.get([nr_in], elt)
            out_ty = ir.VectorType.get([nr_out], elt)
            rvl_in = arith.ConstantOp(i64, a64(nr_in)).result
            rvl = (rvl_in if nr_out == nr_in
                   else arith.ConstantOp(i64, a64(nr_out)).result)
            pad = zero_of(elt) if pad is None else pad
            for h in range(crossings):
                build_push_rows(src_bank, s_at, s_step, rows, in_ty, rvl_in, h, bits,
                                 push_op, simm5, pad)
                build_pop_rows(dst_bank, d_at, d_step, wide, out_ty, rvl, h, bits, crossings,
                                pop_op, simm5, pad)
    move.erase()
    return True


def _xlu_moves_in(module):
    return [(o.operation, ir.IntegerAttr(o.operation.attributes[CROSS_LANE_ATTR]).value)
            for region in module.operation.regions for b in region.blocks
            for o in walk_ops(b)
            if o.operation.name == CROSS_LANE_OP and CROSS_LANE_ATTR in o.operation.attributes]


def call_intrinsic(name, operands, result_tys=()):
    """Emit one `llvm.call_intrinsic`, for an intrinsic THE VCIX DIALECT DOES NOT NAME."""
    from mlir.dialects import llvm
    tys = list(result_tys)
    return llvm.CallIntrinsicOp(tys[0] if tys else None, ir.StringAttr.get(name),
                                list(operands), [], [])
