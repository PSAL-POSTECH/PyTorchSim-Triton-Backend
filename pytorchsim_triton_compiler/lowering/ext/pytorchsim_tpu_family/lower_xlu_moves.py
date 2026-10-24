"""lower_xlu_moves: every `pytorchsim_tpu.xlu_move` -- the pack's own op, minted by lower_copies -- onto
the cross-lane unit (tpu::Xlu): its pushes and pops, as the rounds written on it."""

from ...passes.lib_bank_layout import bank_depth, bank_view, is_uniform, lane_extent, view_lanes
from ...passes.lib_ir import build_trip_loop, const_index, elem_bits, transfer_read, transfer_write
from ...passes.lib_ir import walk_ops, zero_of
from ...passes.lib_vcix import bitcast, i64 as a64, vcix, word_vector
from ...passes.lower import LOWER_TARGET_OPS
from .lib_units import lane_number
from .lib_xlu import (SF_VC_IVV_SE, XLU_GATHER, XLU_OPS, XLU_PERMUTE, XLU_POP_OPCODE, XLU_PUSH_OPCODE,
                      XLU_SET_PATTERN_OPCODE, as_signed_imm, build_pop_rows, build_push_rows, build_row_loop,
                      call_intrinsic, crossing_bits, crossing_words, element_from_words, emitter_of, is_a_load,
                      MAPPED, crossings_of, lane_runs, permute_runs, round_rows, row_at)
from .lib_xlu_move import QUEUE_KEY, ROUNDS_ATTR, XLU_MOVE_OP, XLU_OP_ATTR, refuse

import mlir.ir as ir  # noqa: E402

NAME = "pytorchsim-lower-xlu-moves"

SLOT = LOWER_TARGET_OPS.name

FORBIDS = (XLU_MOVE_OP,)

MARKERS = FORBIDS

# why a permute's pops each wait on their own push, on the function that holds it
XLU_PIPELINE = "pytorchsim.xlu_pipeline"


def run(module, *, target, lmul=None, **_):
    """Every xlu_move lowered where it stands, a push or a pop a group of `lmul` registers at most:
    the kernel's LMUL, which a spill's retry narrows."""
    from ....machine import ISA_MAX_LMUL
    queue = int(target.vcix[QUEUE_KEY]) if QUEUE_KEY in target.vcix else None
    _lower_every_xlu_move(module, target.vector_lanes, target.vlen_bits, queue,
                          ISA_MAX_LMUL if lmul is None else lmul)
    return module


def _lower_every_xlu_move(module, lanes, vlen, queue=None, lmul=1):
    return sum(_lower_one_xlu_move(move, name, lanes, vlen, queue, lmul)
               for move, name in _xlu_moves_in(module))


def _xlu_move_ends(move):
    return move.operands[0], move.operands[1]


def _lower_one_xlu_move(move, name, lanes, vlen, queue=None, lmul=1):
    kind = emitter_of(name)
    if kind == XLU_GATHER:
        return _xlu_all_gather(move, lanes, vlen, lmul)
    if kind == XLU_PERMUTE:
        return _xlu_permute(move, lanes, vlen, queue, lmul)
    src, dst = _xlu_move_ends(move)
    if not _xlu_end_is_contiguous(src, lanes) or not _xlu_end_is_contiguous(
            dst, lanes):
        return refuse(
            name, move, "a lane's rows of one end of this move are not evenly "
            "spaced, so the bank has no single step to walk")
    written = _get_written_rounds(move)
    if written is None:
        return refuse(name, move, "no passes are written on it")
    return _xlu_row_passes(move, lanes, vlen, name, written, lmul)


def _xlu_all_gather(move, lanes, vlen, lmul=1):
    xlu_op = ir.IntegerAttr(move.attributes[XLU_OP_ATTR]).value
    from mlir.dialects import arith
    src, dst = _xlu_move_ends(move)
    bits = elem_bits(ir.MemRefType(src.type).element_type)
    crossings, why = crossings_of(bits)
    if not crossings:
        return refuse(xlu_op, move, why)
    got = _xlu_all_gather_shape(src, dst, lanes, vlen * lmul // max(1, bits // 32))
    if got is None:
        return refuse(xlu_op, move, "the shapes are not the unit's")
    n, per, chunk, elt = got
    plan = _get_written_rounds(move)
    if plan is None:
        return refuse(
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
        return refuse(
            xlu_op, move, f"the written passes are not {rounds} values per lane "
            f"gathered {per} at a time, lap after lap")
    if step != 1:
        chunk = 1
    lap = trips * per
    laps, rest = divmod(ir.MemRefType(dst_bank.type).shape[1], lap)
    if rest or not laps:
        return refuse(
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


def _xlu_all_gather_shape(src, dst, lanes, vlen):
    elt = ir.MemRefType(src.type).element_type
    n = None if is_uniform(src) else view_lanes(src)
    if n is None or (n > lanes and n % lanes):
        return None
    if lane_extent(dst) is None:
        return None
    import math
    per = min(n, lanes)
    chunk = math.gcd(per, vlen // 32)
    return n, per, chunk, elt


def _get_written_rounds(op):
    if ROUNDS_ATTR not in op.attributes:
        return None
    return [tuple(ir.DenseI64ArrayAttr(r))
            for r in ir.ArrayAttr(op.attributes[ROUNDS_ATTR])]


def _xlu_encoding(xlu_op, op):
    if is_a_load(xlu_op):
        raise NotImplementedError(
            f"{XLU_OP_ATTR} = {xlu_op} has pre = 3, which names no RPU mode -- "
            f"it is a pattern load and not a pass the unit can run: {op}")
    if xlu_op not in XLU_OPS:
        raise NotImplementedError(
            f"{XLU_OP_ATTR} = {xlu_op} is not a pass this unit answers. SIMM5 0 "
            f"asks it for no crossing at all, which is a trap: {op}")
    return XLU_PUSH_OPCODE, XLU_POP_OPCODE, as_signed_imm(xlu_op)


def _xlu_permute(move, lanes, vlen, queue=None, lmul=1):
    """The move's rounds, each a push of rows and the pop that answers it, every push before the
    first pop: the unit's input queue drains into its stages with no pop, and it answers in order.
    Each run of rounds that shares a pattern sets it once (xlu_set_pattern) before its pushes, which
    take it as they issue. A push takes a register group of the element; lane numbers (the pattern,
    the lanes a round writes) are i64, so a group of them is as wide as `lane_reg`. With no queue
    stated each pop follows its push, and the function says why on XLU_PIPELINE."""
    xlu_op = ir.IntegerAttr(move.attributes[XLU_OP_ATTR]).value
    from mlir.dialects import arith
    push_op, pop_op, simm5 = _xlu_encoding(xlu_op, move)
    src, dst = _xlu_move_ends(move)
    if not _xlu_end_is_contiguous(src, lanes) or not _xlu_end_is_contiguous(
            dst, lanes):
        return refuse(
            xlu_op, move, "a lane's rows of one end are not evenly spaced")
    elt = ir.MemRefType(src.type).element_type
    bits = elem_bits(elt)
    crossings, why = crossings_of(bits)
    if not crossings:
        return refuse(xlu_op, move, why)
    i64 = ir.IntegerType.get_signless(64)


    with ir.InsertionPoint(move):
        src_bank = bank_view(src, lanes)
        dst_bank = bank_view(dst, lanes)
    depth = ir.MemRefType(dst_bank.type).shape[1]
    row = min(view_lanes(dst) or lanes, lanes)
    rounds = _get_written_rounds(move)
    if rounds is None or any(not 0 <= round_rows(r)[1] < depth for r in rounds):
        return refuse(
            xlu_op, move, "no rounds, or a round writes a destination row the bank "
            "does not have")
    src_depth = ir.MemRefType(src_bank.type).shape[1]
    if any(not 0 <= round_rows(r)[0] < src_depth for r in rounds):
        return refuse(
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
        set_family = rvl_of(XLU_SET_PATTERN_OPCODE)
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
                push_round(w, si, h)
            if stage[0] != "push":
                pop_round(owns(lo, hi, every, w) if part else None, di, w, h)

        def set_pattern(mul, add, div, w):
            """Each lane's source lane, `(lane * mul + add) / div`, in `w` entries: the pattern
            every push after it takes, entry i mod `w` for element i, so any width serves."""
            idx = at_width[w]
            if mul != 1:
                idx = arith.MulIOp(idx, splat(mul, w)).result
            if not isinstance(add, int) or add:
                idx = arith.AddIOp(idx, vec(add, w)).result
            if div != 1:
                idx = arith.DivSIOp(idx, splat(div, w)).result
            set_lanes(idx, w)

        def set_lanes(idx, w):
            lanes_of = arith.TruncIOp(ir.VectorType.get([w], i32_ty), idx).result
            call_intrinsic(SF_VC_IVV_SE, [set_family, lanes_of, lanes_of, rvl_of(0),
                                           rvl1 if w == 1 else rvl_of(w)])

        def set_mapped(src_of, w):
            """A MAPPED round's pattern: `lane ^ x` when that is every written lane's source,
            else each run's `(lane * mul + add) / div` on its lanes, a lane past the machine's
            (which reads nothing) on the others."""
            lane = at_width[w]
            written = [k for k, s in enumerate(src_of) if s >= 0]
            x = src_of[written[0]] ^ written[0]
            if all(src_of[k] == k ^ x for k in written):
                return set_lanes(arith.XOrIOp(lane, splat(x, w)).result if x else lane, w)
            idx = splat(lanes, w)
            for lo, hi, every, mul, add, div in lane_runs(src_of):
                v = lane if mul == 1 else arith.MulIOp(lane, splat(mul, w)).result
                v = arith.AddIOp(v, splat(add, w)).result if add else v
                v = arith.DivSIOp(v, splat(div, w)).result if div != 1 else v
                idx = arith.SelectOp(owns(lo, hi, every, w), v, idx).result
            set_lanes(idx, w)

        def written_mask(src_of, w):
            """The lanes a MAPPED round writes, or None when it writes every lane of the row."""
            if all(s >= 0 for s in src_of[:row]):
                return None
            mask = None
            for lo, hi, every, *_ in lane_runs([k if s >= 0 else -1 for k, s in enumerate(src_of)]):
                m = owns(lo, hi, every, w)
                mask = m if mask is None else arith.OrIOp(mask, m).result
            return mask

        def emit_mapped(src_of, runs, hs):
            """A MAPPED group: its pattern set once for every word `hs` of the element, then each
            run's rows a register group a push and a pop."""
            if stage[0] != "pop":
                set_mapped(src_of, min(max(w for *_, w in runs), lane_reg))

            def one(si, di, w, h):
                if stage[0] != "pop":
                    push_round(w, si, h)
                if stage[0] != "push":
                    pop_round(written_mask(src_of, w), di, w, h)
            for h in hs:
                for s0, ss, d0, ds, n, w in runs:
                    build_row_loop(n // w, lambda i, s0=s0, ss=ss, d0=d0, ds=ds, w=w, h=h:
                                   one(row_at(s0, i, w * ss), row_at(d0, i, w * ds), w, h))
                    if n % w:
                        k = n - n % w
                        one(past(s0, k * ss), past(d0, k * ds), n % w, h)

        def push_round(w, si, h):
            wide = ir.VectorType.get([w], elt)
            wide_w = word_vector(wide)
            rvl = rvl1 if w == 1 else rvl_of(w)
            v = transfer_read(wide, src_bank, [const_index(0), si], pad, True)
            vcix("vcix.iv", [crossing_words(cast(v, wide_w), bits)[h], rvl], [],
                 {"opcode": a64(push_op), "imm": a64(simm5), "rd": a64(0)})

        def pop_round(mask, di, w, h):
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
            transfer_write(cast(got, wide), dst_bank, [const_index(0), di], True, mask)

        per_reg = lmul * vlen // max(crossing_bits(elt), elem_bits(elt))
        lane_reg = lmul * vlen // 64
        plan, by_map = [], {}
        for pattern, (s0, ss, d0, ds, n), w, part in permute_runs(rounds, row, per_reg, lane_reg):
            if pattern[0] == MAPPED:
                by_map.setdefault(pattern[1], []).append((s0, ss, d0, ds, n, w))
                continue
            mul, add, lo, hi, div, every = pattern
            plan.append(((mul, div, every, ss, ds, n, w, part), (add, lo, hi, s0, d0)))
        mapped = list(by_map.items())
        for (*_, n, w, _part), _start in plan:
            for x in {min(w, lane_reg), min(n % w, lane_reg)} - {0}:
                _get_lane_number(x)
        for _src_of, runs in mapped:
            for *_, n, w in runs:
                for x in {min(w, lane_reg), min(n % w, lane_reg)} - {0}:
                    _get_lane_number(x)
            _get_lane_number(min(max(w for *_, w in runs), lane_reg))
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

        def emit_rows(shape, now, hs):
            """One entry's rows at the starts `now`, its pattern set once for every word `hs` of the
            element: a vector of rows per round where they run on, and the rows left over."""
            mul, div, every, ss, ds, n, w, part = shape
            add, lo, hi, si, di = now
            if stage[0] != "pop":
                set_pattern(mul, add, div, min(w, lane_reg))
            for h in hs:
                build_row_loop(n // w, lambda i, h=h: one_round(
                    mul, add, div, lo, hi, every, part, row_at(si, i, w * ss), row_at(di, i, w * ds),
                    w, h))
                if n % w:
                    k = n - n % w
                    one_round(mul, add, div, lo, hi, every, part, past(si, k * ss),
                              past(di, k * ds), n % w, h)

        def nest(shape, now, dims, hs):
            """The loops of a sweep, outer first, each moving the starts by its steps."""
            if not dims:
                return emit_rows(shape, now, hs)
            (m, steps), inner = dims[0], dims[1:]

            def sweep(g):
                nest(shape, [i64_at(g, x, k) for x, k in zip(now[:3], steps[:3])]
                     + [sweep_at(g, x, k) for x, k in zip(now[3:], steps[3:])], inner, hs)
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
            for shape, start, dims in sweeps:
                nest(shape, list(start), dims or ((1, (0,) * len(start)),), range(crossings))
            for src_of, runs in mapped:
                emit_mapped(src_of, runs, range(crossings))

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


def _xlu_row_passes(move, lanes, vlen, xlu_op, passes, lmul=1):
    from mlir.dialects import arith
    src, dst = _xlu_move_ends(move)
    elt = ir.MemRefType(src.type).element_type
    bits = elem_bits(elt)
    crossings, why = crossings_of(bits)
    if not crossings:
        return refuse(xlu_op, move, why)
    push_op, pop_op, simm5 = _xlu_encoding(xlu_op, move)
    i64 = ir.IntegerType.get_signless(64)
    per_reg = lmul * vlen // max(crossing_bits(elt), elem_bits(elt))

    with ir.InsertionPoint(move):
        src_bank = bank_view(src, lanes)
        dst_bank = bank_view(dst, lanes)
        pad = None
        import math
        for s_at, s_step, d_at, d_step, rows, wide in passes:
            nr_in = math.gcd(rows, per_reg) if s_step == 1 else 1
            nr_out = math.gcd(wide, per_reg) if d_step == 1 else 1
            if rows % nr_in or wide % nr_out:
                return refuse(
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
    return [(o.operation, ir.IntegerAttr(o.operation.attributes[XLU_OP_ATTR]).value)
            for region in module.operation.regions for b in region.blocks
            for o in walk_ops(b)
            if o.operation.name == XLU_MOVE_OP and XLU_OP_ATTR in o.operation.attributes]
