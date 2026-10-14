"""torchsim-lower-to-xlu: every marked `torchsim.crosslane_move` -> the cross-lane unit's pushes and
pops, as the rounds written on it; a kernel's torchsim.vcix -> its encoding issued over its tiles'
rows, as those pushes and pops read and write them."""

import sys

from ... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from .lib_bank_layout import bank_depth, bank_view, index_bank_casts, lanes_of  # noqa: E402
from .lib_cross_lane import (CROSS_LANE_ATTR, CROSS_LANE_OP, XLU_GATHER, XLU_OPS,  # noqa: E402
                             XLU_PERMUTE, XLU_POP_OPCODE, XLU_PUSH_OPCODE, as_signed_imm,
                             emitter_of, is_a_load, permute_groups)
from .lib_ir import (build_trip_loop, const_index, elem_bits, operand_segments,  # noqa: E402
                     transfer_read, transfer_write, walk, walk_ops, zero_of)
from ...contract.vcix_form import FORMS  # noqa: E402
from .lib_vcix import bitcast, i64 as a64, vcix, word_vector  # noqa: E402
from ..ext.vcix_tpuv6.lib_units import lane_number  # noqa: E402

KERNEL_VCIX = "torchsim.vcix"

FORBIDS = (CROSS_LANE_OP, KERNEL_VCIX)

MARKERS = FORBIDS

_SF_VC_IVV_SE = "llvm.riscv.sf.vc.ivv.se"

def run(module, *, target, **_):
    """Every marked cross-lane move and every kernel's vcix lowered where it stands."""
    vectorlane, vlen = target.vector_lanes, target.vlen_bits
    _lower_every_xlu_move(module, vectorlane, vlen)
    for op in [o for o in walk(module.operation) if o.name == KERNEL_VCIX]:
        _lower_kernel_vcix(op, vectorlane, vlen)
    return module


def _lower_kernel_vcix(op, lanes, vlen):
    """`vcix.<form>` with the kernel's opcode and imm over its tiles' rows in order -- read from
    its operand, written into its dst, or both -- as a crossing reads and writes them."""
    from mlir.dialects import arith
    form = ir.StringAttr(op.attributes["form"]).value
    opcode, imm = (ir.IntegerAttr(op.attributes[k]).value for k in ("opcode", "imm"))
    n_in, n_dst = operand_segments(op)
    if FORMS.get(form) != (n_in, n_dst) or len(op.results):
        raise NotImplementedError(f"a kernel's vcix.{form} on buffers with {n_in} operand(s) and "
                                  f"{n_dst} dst: the form takes {FORMS.get(form)}: {op}")
    tiles = list(op.operands)
    elt = ir.MemRefType(tiles[0].type).element_type
    bits = elem_bits(elt)
    if form != "v.iv" and bits > 32:
        raise NotImplementedError(
            f"a kernel's vcix.{form} on {bits}-bit elements: the unit carries 32 bits at a time, "
            f"and a kernel's push and pop are separate instructions, so the halves cannot "
            f"alternate as a crossing's do: {op}")
    for t in tiles:
        if lanes_of(t) != lanes:
            raise NotImplementedError(
                f"a kernel's vcix.{form} on {t.type}: the tile does not hold a row in each of the "
                f"machine's {lanes} lanes, so the instruction reads or writes lanes it does not "
                f"cover: {op}")
    i64 = ir.IntegerType.get_signless(64)
    with ir.InsertionPoint(op):
        banks = [bank_view(t, lanes) for t in tiles]
        depth = ir.MemRefType(banks[0].type).shape[1]
        nr = _get_rows_per_vector(1, depth, vlen // _get_crossing_bits(elt))
        if depth % nr or any(ir.MemRefType(b.type).shape[1] != depth for b in banks):
            raise NotImplementedError(f"{depth} rows per lane are not a whole number of "
                                      f"{nr}-wide registers, or its tiles differ in depth: {op}")
        vec = ir.VectorType.get([nr], elt)
        if form == "v.iv":
            _build_viv_rows(banks[0], banks[1], depth, vec, opcode, imm, zero_of(elt))
        elif n_dst:
            rvl = arith.ConstantOp(i64, a64(nr)).result
            _build_pop_rows(banks[0], 0, 1, depth, vec, rvl, 0, bits, 1, opcode, imm, None)
        else:
            rvl = arith.ConstantOp(i64, a64(nr)).result
            _build_push_rows(banks[0], 0, 1, depth, vec, rvl, 0, bits, opcode, imm, zero_of(elt))
    op.erase()


def _build_viv_rows(src, dst, rows, vec, opcode, imm, pad):
    """`rows` rows of `src`, `vec` at a time, each through one vcix.v.iv into the same rows of
    `dst`."""
    nr = ir.VectorType(vec).shape[0]

    def one(i):
        at = _get_row_at(0, i, nr)
        v = transfer_read(vec, src, [const_index(0), at], pad)
        got = vcix("vcix.v.iv", [v], [vec], {"opcode": a64(opcode), "imm": a64(imm)}).results[0]
        transfer_write(got, dst, [const_index(0), at])
    _build_row_loop(rows // nr, one)


def _lower_every_xlu_move(module, lanes, vlen):
    index_bank_casts(module)
    return sum(_lower_one_xlu_move(move, name, lanes, vlen)
               for move, name in _xlu_moves_in(module))


def _lower_one_xlu_move(move, name, lanes, vlen):
    kind = emitter_of(name)
    if kind == XLU_GATHER:
        return _xlu_all_gather(move, lanes, vlen)
    if kind == XLU_PERMUTE:
        return _xlu_permute(move, lanes, vlen)
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
    crossings, why = _get_crossings(bits)
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
    if ir.MemRefType(dst_bank.type).shape[1] != trips * per:
        return _refuse_xlu_move(
            xlu_op, move, "a lane would not end up with the whole tile")

    g_push, g_pop, g_imm = _xlu_encoding(xlu_op, move)
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
            v = _build_element_from_words(
                got, h, bits, chunk_word,
                None if crossings == 1 or not h else cast(transfer_read(
                    chunk_ty, dst_bank, [const_index(0), at], zero_of(elt)),
                    chunk_word))
            transfer_write(cast(v, chunk_ty), dst_bank, [const_index(0), at])

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
                                    zero_of(elt)), one_word)
            for h, part in enumerate(_build_crossing_words(v, bits)):
                vcix("vcix.iv", [part, rvl1], [],
                      {"opcode": a64(g_push), "imm": a64(g_imm), "rd": a64(0)})

                def pop(i, h=h):
                    got = vcix("vcix.v.i", [rvlc], [cross_chunk],
                                {"opcode": a64(g_pop), "imm": a64(g_imm),
                                 "rs2": a64(0)}).results[0]
                    landed(got, h, _get_row_at(0 if base is None else base,
                                               i, chunk * step))
                _build_row_loop(per // chunk, pop)

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
    move.erase()
    return True


def _xlu_move_ends(move):
    return move.operands[0], move.operands[1]


def _get_crossings(bits):
    if bits <= 32:
        return 1, None
    if bits != 64:
        return 0, f"its element is {bits} bits and the unit carries 32 at a time"
    return 2, None


def _refuse_xlu_move(name, op, why):
    raise NotImplementedError(
        f"{CROSS_LANE_ATTR} = '{name}' cannot be lowered here ({why}); "
        f"materialize_layouts must not mint one this cannot take: {op}")


def _xlu_all_gather_shape(src, dst, lanes, vlen):
    elt = ir.MemRefType(src.type).element_type
    n = _lane_extent(src)
    if n is None or (n > lanes and n % lanes):
        return None
    if _lane_extent(dst) is None:
        return None
    per = min(n, lanes)
    chunk = min(per, vlen // 32)
    if per % chunk:
        return None
    return n, per, chunk, elt


def _lane_extent(v):
    n = lanes_of(v)
    return None if n == 1 else n


def _get_written_rounds(op):
    from .lib_cross_lane import CROSS_LANE_ROUNDS
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


def _build_element_from_words(word, h, bits, elem_ty, parked=None):
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


def _build_crossing_words(v, bits):
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


def _get_row_at(base, i, step):
    from mlir.dialects import arith
    if i is None:
        return const_index(base) if isinstance(base, int) else base
    off = i if step == 1 else arith.MulIOp(i, const_index(step)).result
    if isinstance(base, int):
        return off if not base else arith.AddIOp(off, const_index(base)).result
    return arith.AddIOp(base, off).result


def _build_row_loop(n, body):
    if n == 1:
        body(None)
        return
    iv, inside = build_trip_loop(ir.InsertionPoint.current, n)
    with inside:
        body(iv)


def _xlu_permute(move, lanes, vlen):
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
    crossings, why = _get_crossings(bits)
    if not crossings:
        return _refuse_xlu_move(xlu_op, move, why)
    i64 = ir.IntegerType.get_signless(64)


    with ir.InsertionPoint(move):
        src_bank = bank_view(src, lanes)
        dst_bank = bank_view(dst, lanes)
    depth = ir.MemRefType(dst_bank.type).shape[1]
    row = min(lanes_of(dst) or lanes, lanes)
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

        lane = _get_lane_number(1)

        def vec(k, w):
            """`k` in each of `w` rows: a constant, or an i64 a sweep made."""
            if isinstance(k, int):
                return splat(k, w)
            return vector.BroadcastOp(ir.VectorType.get([w], i64), k).result

        def owns(lo, hi, every):
            """The lanes this round is the one to write -- lo, lo + every, ..."""
            keep = arith.CmpIOp(arith.CmpIPredicate.slt, lane, vec(hi, 1)).result
            if not (isinstance(lo, int) and not lo):
                keep = arith.AndIOp(keep, arith.CmpIOp(
                    arith.CmpIPredicate.sge, lane, vec(lo, 1)).result).result
            if every == 1:
                return keep
            off = (lane if isinstance(lo, int) and not lo
                   else arith.SubIOp(lane, vec(lo, 1)).result)
            return arith.AndIOp(keep, arith.CmpIOp(
                arith.CmpIPredicate.eq, arith.RemSIOp(off, splat(every, 1)).result,
                splat(0, 1)).result).result

        def one_round(mul, add, div, lo, hi, every, part, si, di, w, h):
            """`w` rows from source row `si` and the pattern they share, in ONE push, and the pop
            that answers them into row `di`."""
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
            v = transfer_read(wide, src_bank, [const_index(0), si], pad)
            call_intrinsic(_SF_VC_IVV_SE,
                           [family, arith.TruncIOp(wide_x, idx).result,
                            _build_crossing_words(cast(v, wide_w), bits)[h],
                            code, rvl])
            got = vcix("vcix.v.i", [rvl], [wide_x],
                       {"opcode": a64(pop_op), "imm": a64(simm5),
                        "rs2": a64(0)}).results[0]
            there = (cast(transfer_read(wide, dst_bank, [const_index(0), di],
                                        pad), wide_w) if part or h else None)
            got = _build_element_from_words(got, h, bits, wide_w,
                                            there if h else None)
            if part:
                got = arith.SelectOp(owns(lo, hi, every), got, there).result
            transfer_write(cast(got, wide), dst_bank, [const_index(0), di])

        per_reg = vlen // _get_crossing_bits(elt)
        plan = []
        for mul, add, lo, hi, div, every, runs in permute_groups(rounds):
            part = bool(lo) or hi != row or every != 1
            for s0, ss, d0, ds, n in runs:
                w = (1 if part else min(_get_rows_per_vector(ss, n, per_reg),
                                        _get_rows_per_vector(ds, n, per_reg)))
                plan.append(((mul, div, every, ss, ds, n, w, part), (add, lo, hi, s0, d0)))
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
            return _get_row_at(base, g, k)

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
                one_round(mul, add, div, lo, hi, every, part, _get_row_at(si, i, w * ss),
                          _get_row_at(di, i, w * ds), w, h)
            _build_row_loop(n // w, rows)
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
            _build_row_loop(m, sweep)

        for h in range(crossings):
            for shape, start, dims in _get_sweeps(plan):
                nest(shape, list(start), dims or ((1, (0,) * len(start)),), h)

    move.erase()
    return True


def _xlu_end_is_contiguous(v, lanes):
    return bank_depth(v, lanes) is not None


def _get_crossing_bits(elt):
    return max(elem_bits(elt), 32)


def _get_rows_per_vector(step, n, per_reg):
    return min(per_reg, n) if step == 1 else 1


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
    crossings, why = _get_crossings(bits)
    if not crossings:
        return _refuse_xlu_move(xlu_op, move, why)
    push_op, pop_op, simm5 = _xlu_encoding(xlu_op, move)
    i64 = ir.IntegerType.get_signless(64)
    per_reg = vlen // _get_crossing_bits(elt)

    with ir.InsertionPoint(move):
        src_bank = bank_view(src, lanes)
        dst_bank = bank_view(dst, lanes)
        pad = None
        for s_at, s_step, d_at, d_step, rows, wide in passes:
            nr_in = _get_rows_per_vector(s_step, rows, per_reg)
            nr_out = _get_rows_per_vector(d_step, wide, per_reg)
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
                _build_push_rows(src_bank, s_at, s_step, rows, in_ty, rvl_in, h, bits,
                                 push_op, simm5, pad)
                _build_pop_rows(dst_bank, d_at, d_step, wide, out_ty, rvl, h, bits, crossings,
                                pop_op, simm5, pad)
    move.erase()
    return True


def _build_push_rows(bank, at, step, rows, in_ty, rvl, h, bits, opcode, imm, pad):
    """`rows` rows of `bank` from row `at`, `step` apart, pushed `in_ty` at a time: word `h`."""
    nr = ir.VectorType(in_ty).shape[0]

    def push(i):
        v = transfer_read(in_ty, bank, [const_index(0), _get_row_at(at, i, nr * step)], pad)
        vcix("vcix.iv", [_build_crossing_words(_cast(v, word_vector(in_ty)), bits)[h], rvl], [],
             {"opcode": a64(opcode), "imm": a64(imm), "rd": a64(0)})
    _build_row_loop(rows // nr, push)


def _build_pop_rows(bank, at, step, rows, out_ty, rvl, h, bits, crossings, opcode, imm, pad):
    """`rows` rows popped into `bank` from row `at`, `step` apart, `out_ty` at a time: word `h`,
    joined to the words before it."""
    nr = ir.VectorType(out_ty).shape[0]
    cross_ty = ir.VectorType.get([nr], ir.IntegerType.get_signless(32))

    def pop(i):
        got = vcix("vcix.v.i", [rvl], [cross_ty],
                   {"opcode": a64(opcode), "imm": a64(imm), "rs2": a64(0)}).results[0]
        row = _get_row_at(at, i, nr * step)
        v = _build_element_from_words(
            got, h, bits, word_vector(out_ty),
            None if crossings == 1 or not h else _cast(
                transfer_read(out_ty, bank, [const_index(0), row], pad), word_vector(out_ty)))
        transfer_write(_cast(v, out_ty), bank, [const_index(0), row])
    _build_row_loop(rows // nr, pop)


def _cast(v, ty):
    return v if v.type == ty else bitcast(v, ty)


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
