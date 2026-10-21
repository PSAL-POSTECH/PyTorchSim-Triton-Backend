"""lib_xlu: vcix_tpuv6e's cross-lane unit (tpu::Xlu) -- what each XLU op, a (pre, XU, post) setting,
is, the rounds it runs to carry a correspondence, and the push/pop rows its 32-bit queue moves; no
SLOT, so not a pass."""

from ...passes.lib_bank_layout import need_per_lane
from ...passes.lib_cross_lane import CROSSING, GATHER, PERMUTE
from ...passes.lib_ir import build_trip_loop, const_index, elem_bits, transfer_read, transfer_write
from ...passes.lib_vcix import bitcast, i64 as a64, vcix, word_vector

import mlir.ir as ir  # noqa: E402

RPU_BYPASS, RPU_REPLICATE, RPU_ARBITRARY, RPU_NONE = 0, 1, 2, 3

XLU_PUSH_OPCODE, XLU_POP_OPCODE = 3, 1

# xlu_push_reduce (sf.vc.ivv 2): a tile folded across the lanes, its field the fold (0 max, 1 sum, 2 min).
XLU_REDUCE_OPCODE = 2

XLU_FOLDS = {"arith.maximumf": 0, "arith.maxnumf": 0, "arith.addf": 1, "arith.minimumf": 2, "arith.minnumf": 2}

# The unit's max and min are arith.maximumf/minimumf: a NaN on either side is the result. These combiners drop
# a NaN instead, so each lane pushes the fold's identity where it holds one.
XLU_FOLDS_DROPPING_NAN = frozenset({"arith.maxnumf", "arith.minnumf"})


def simm5(pre, xu, post):
    """The five bits of a `(pre, XU, post)` answer."""
    return (pre << 3) | (xu << 2) | post


def as_signed_imm(xlu_op):
    """The SIMM5 field as the instruction spells it."""
    return xlu_op - 32 if xlu_op >= 16 else xlu_op


def fields(xlu_op):
    """`(pre, XU, post)` of a SIMM5."""
    return (xlu_op >> 3) & 3, (xlu_op >> 2) & 1, xlu_op & 3


def is_a_load(xlu_op):
    """Is this SIMM5 a pattern load rather than a pass?"""
    return ((xlu_op >> 3) & 3) == RPU_NONE


XLU_OPS = tuple(sorted(simm5(p, x, q)
                       for p in (RPU_BYPASS, RPU_REPLICATE, RPU_ARBITRARY)
                       for x in (0, 1)
                       for q in (RPU_BYPASS, RPU_REPLICATE, RPU_ARBITRARY)
                       if simm5(p, x, q) != 0))


XLU_CROSSING, XLU_GATHER, XLU_PERMUTE = CROSSING, GATHER, PERMUTE


def emitter_of(xlu_op):
    """Which shape of move `xlu_op` is, or None where it is none of them: a replicating pre-stage
    copies one lane's value into every lane, which no move here asks of the unit."""
    pre, xu, post = fields(xlu_op)
    if RPU_ARBITRARY in (pre, post):
        return XLU_PERMUTE if not xu and post == RPU_BYPASS else None
    if post == RPU_REPLICATE:
        return XLU_GATHER if xu and pre == RPU_BYPASS else None
    if pre == RPU_REPLICATE:
        return None
    return XLU_CROSSING if xu and post == RPU_BYPASS else None


def fold_onto_lanes(pairs, s_shape, d_shape, lanes):
    """`pairs` on the lanes the machine HAS: lane-full t of an axis 0 longer than `lanes` sits t
    times the other axes' rows deeper, as the DMA stages it and need_per_lane reserves it; a last,
    partial lane-full (a scan pads to one and a half) on the lanes it reaches, as deep as a whole."""
    def fold(shape):
        if not shape or shape[0] <= lanes:
            return lambda at: at
        rows = 1
        for d in shape:
            rows *= d
        rows //= shape[0]
        return lambda at: (at[0] % lanes, at[0] // lanes * rows + at[1])
    if pairs is None:
        return None
    fs, fd = fold(s_shape), fold(d_shape)
    return [(fs(f), fd(t)) for f, t in pairs]

def crossing_passes(pairs, lanes, deep, n_lane):
    """One crossing each, as `(source row, its step, destination row, its step, rows in, rows out)`,
    or None wherever a pass's rows are not a RUN AT ONE STEP."""
    if pairs is None:
        return None
    where, b = {}, {}
    for (sl, sd), (tl, td) in pairs:
        where[(sl, sd)] = (tl, td)
        if b.setdefault(sd, tl) != tl:
            return None
    rows = 1 + max(b.values())
    if rows > n_lane or deep % rows:
        return None
    by_lane = {}
    for sd, tl in b.items():
        by_lane.setdefault(tl, []).append(sd)
    if any(len(v) != deep // rows for v in by_lane.values()):
        return None
    out = []
    for k in range(deep // rows):
        take = [sorted(by_lane[j])[k] for j in range(rows)]
        step = {take[i + 1] - take[i] for i in range(rows - 1)} or {1}
        if len(step) != 1:
            return None
        at = {}
        for l in range(lanes):
            got = where.get((l, take[0]))
            if got is None:
                return None
            at[l] = got[1]
        d_step = {at[l + 1] - at[l] for l in range(lanes - 1)} or {1}
        if len(d_step) != 1:
            return None
        d_step = d_step.pop()
        for j, sd in enumerate(take):
            for l in range(lanes):
                if where.get((l, sd)) != (j, at[0] + d_step * l):
                    return None
        out.append((take[0], step.pop(), at[0], d_step, rows, lanes))
    return out

def rounds_of_pairs(pairs):
    """The same rounds, asked of pairs the caller already built: one per destination row, source row
    and RUN OF LANES `(at, mul, add, depth, lo, hi, div, step)`."""
    if pairs is None:
        return None
    groups = {}
    for (s_lane, s_deep), (d_lane, d_deep) in pairs:
        groups.setdefault((d_deep, s_deep), {})[d_lane] = s_lane
    return [(at, mul, add, depth, lo, hi, div, step)
            for (depth, at), seen in sorted(groups.items())
            for lo, hi, step, mul, add, div in _get_lane_runs(seen)]


def _get_lane_runs(seen):
    runs = []
    for d in sorted(seen):
        if runs:
            d0, a, n, b = runs[-1]
            if n == 1:
                runs[-1] = [d0, d - d0, 2, seen[d] - seen[d0]]
                continue
            if d == d0 + a * n and seen[d] == seen[d0] + b * n:
                runs[-1][2] += 1
                continue
        runs.append([d, 1, 1, 1])
    out = []
    for d0, a, n, b in runs:
        if b % a == 0:
            mul, add, div = b // a, seen[d0] - b // a * d0, 1
        else:
            mul, add, div = b, seen[d0] * a - b * d0, a
        out.append((d0, d0 + a * (n - 1) + 1, a, mul, add, div))
    return out

def gather_rounds(pairs, src_shape, dst_shape, lanes, src_uniform):
    """Every pass of a gathering move, as `(source depth, destination depth, its step)`: every lane
    takes one source depth's values, landing at `depth * source lanes + lane`, or popped one at a
    time at `lane * rows + depth`; once per lane-full of the destination. A source axis 0 longer
    than `lanes` is read as the DMA stages it, lane-full t that many rows deeper: a pass still
    carries one value per lane, so it is one more pass per lane-full."""
    if pairs is None or not dst_shape or not src_shape or src_uniform:
        return None
    per = src_shape[0]
    if per > lanes:
        if per % lanes:
            return None
        pairs, per = fold_onto_lanes(pairs, src_shape, [], lanes), lanes
    numel = 1
    for n in src_shape:
        numel *= n
    total = 1
    for n in dst_shape:
        total *= n
    if total != numel * dst_shape[0] or len(pairs) != numel * len({t[0] for _f, t in pairs}):
        return None
    rows = numel // per
    if all(t[1] == f[1] * per + f[0] for f, t in pairs):
        at, step = per, 1
    elif all(t[1] == f[0] * rows + f[1] for f, t in pairs):
        at, step = 1, rows
    else:
        return None
    if _laps(dst_shape, lanes) is None:
        return None
    # every lane-full of the destination past the first holds the first again: the passes stop
    # there, and the lowering copies that lap in-lane (lower_cross_lane._copy_first_lap)
    return [(i, i * at, step) for i in range(rows)]

def _laps(shape, lanes):
    if shape[0] <= lanes:
        return 1
    return None if shape[0] % lanes else shape[0] // lanes

def permute_groups(rounds):
    """The rounds that share ONE pattern: `(mul, add, lo, hi, div, step, runs)`, a run being
    `(source row, its step, destination row, its step, how many)`."""
    by, order = {}, []
    for at, mul, add, depth, lo, hi, div, step in rounds:
        key = (mul, add, lo, hi, div, step)
        if key not in by:
            by[key], _ = [], order.append(key)
        by[key].append((at, depth))
    out = []
    for key in order:
        runs = []
        for at, p in by[key]:
            if runs:
                s0, ss, d0, ds, n = runs[-1]
                if n == 1:
                    ss, ds = at - s0, p - d0
                if ss and at == s0 + ss * n and p == d0 + ds * n:
                    runs[-1] = (s0, ss, d0, ds, n + 1)
                    continue
            runs.append((at, 1, p, 1, 1))
        out.append(key + (runs,))
    return out


def plan_of_xlu_op(xlu_op, pairs, s_shape, d_shape, lanes, src_uniform, src_lanes):
    """The rounds `xlu_op` runs for this correspondence on `lanes` lanes, as integer tuples, or None
    where it does not carry it; `src_uniform` is whether the source's buffer is one lane's,
    `src_lanes` how many lanes the source's own elements ride (the caller's view_lanes)."""
    kind = emitter_of(xlu_op)
    if kind == XLU_CROSSING:
        extent = src_lanes
        numel = 1
        for d in s_shape:
            numel *= d
        return crossing_passes(fold_onto_lanes(pairs, s_shape, d_shape, lanes),
                               min(extent, lanes), need_per_lane(numel, extent, lanes), lanes)
    if kind == XLU_PERMUTE:
        return rounds_of_pairs(fold_onto_lanes(pairs, s_shape, d_shape, lanes))
    if kind == XLU_GATHER:
        return gather_rounds(pairs, s_shape, d_shape, lanes, src_uniform)
    return None


def reaches_every_lane(pairs, lanes):
    """Whether these (from, to) pairs carry each destination row from one source element into all
    `lanes` lanes: EVERY_LANE, read off the correspondence a move carries, not its buffer."""
    rows = {}
    for f, (lane, deep) in pairs or ():
        srcs, got = rows.setdefault(deep, (set(), set()))
        srcs.add(f)
        got.add(lane)
    return bool(rows) and all(len(s) == 1 and g == set(range(lanes)) for s, g in rows.values())


def rounds_of_xlu_op(xlu_op, pairs, s_shape, d_shape, lanes, src_uniform, src_lanes):
    """How many rounds `xlu_op` emits for this correspondence, or None where it does not carry it:
    the length of its plan, which the emitter reads."""
    got = plan_of_xlu_op(xlu_op, pairs, s_shape, d_shape, lanes, src_uniform, src_lanes)
    return None if got is None else len(got)


def choose_xlu_op(pairs, s_shape, d_shape, lanes, src_uniform, src_lanes, kinds=None):
    """`(the XLU op of fewest rounds that carries these (from, to) pairs, its plan)`, or None where
    none does."""
    if pairs is None or not lanes:
        return None
    ranked = [(n, c) for c in XLU_OPS if kinds is None or emitter_of(c) in kinds
              for n in [rounds_of_xlu_op(c, pairs, s_shape, d_shape, lanes, src_uniform, src_lanes)]
              if n is not None]
    if not ranked:
        return None
    xlu_op = min(ranked)[1]
    return xlu_op, plan_of_xlu_op(xlu_op, pairs, s_shape, d_shape, lanes, src_uniform, src_lanes)


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


def rows_per_vector(step, n, per_reg):
    return min(per_reg, n) if step == 1 else 1


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
