"""lib_xlu: the cross-lane unit (tpu::Xlu) -- the moves it makes (XLU_MODES), the rounds each runs to carry
a correspondence, and the push/pop rows its 32-bit queue moves; no SLOT, so not a pass."""

from ...passes.lib_bank_layout import need_per_lane
from ...passes.lib_convert import fold_onto_lanes
from ...passes.lib_ir import build_trip_loop, const_index, elem_bits, transfer_read, transfer_write
from ...passes.lib_vcix import bitcast, word_vector

import mlir.ir as ir  # noqa: E402

# xlu.push_reduce: a tile of XLU_FOLD_TYPE folded across the lanes, by the kind each combiner is -- the one
# statement of which reduces the unit folds, which the TritonGPU anchor reads (triton_compile.anchor_options).

XLU_FOLDS = {"arith.maximumf": "max", "arith.maxnumf": "max", "arith.addf": "sum", "arith.minimumf": "min",
             "arith.minnumf": "min"}
XLU_FOLD_TYPE = "f32"

# The unit's max and min are arith.maximumf/minimumf: a NaN on either side is the result. These combiners drop
# a NaN instead, so each lane pushes the fold's identity where it holds one.
XLU_FOLDS_DROPPING_NAN = frozenset({"arith.maxnumf", "arith.minnumf"})


# The moves the unit makes, each the xlu.push / xlu.pop mode of that name, in the order a tie in rounds is
# broken (choose_mode).
XLU_TRANSPOSE, XLU_GATHER, XLU_PERMUTE = "transpose", "gather", "permute"
XLU_MODES = (XLU_TRANSPOSE, XLU_GATHER, XLU_PERMUTE)


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

def rounds_of_pairs(pairs, lanes):
    """The same rounds, asked of pairs the caller already built: one per destination row and source
    row. When one run of lanes carries that row, `(at, mul, add, depth, lo, hi, div, step)`;
    otherwise `(MAPPED, at, depth, the source lane of lane 0, ..., of lane lanes - 1)`, -1 where a
    lane takes nothing: the pattern the unit is set to names any source lane in each lane."""
    if pairs is None:
        return None
    groups = {}
    for (s_lane, s_deep), (d_lane, d_deep) in pairs:
        groups.setdefault((d_deep, s_deep), {})[d_lane] = s_lane
    out = []
    for (depth, at), seen in sorted(groups.items()):
        runs = _get_lane_runs(seen)
        if len(runs) == 1:
            lo, hi, step, mul, add, div = runs[0]
            out.append((at, mul, add, depth, lo, hi, div, step))
        else:
            out.append((MAPPED, at, depth) + tuple(seen.get(lane, -1) for lane in range(lanes)))
    return out


# the first entry of a round whose lanes no one run carries: no row is at -1
MAPPED = -1


def round_rows(r):
    """(source row, destination row) of a permute round of either form."""
    return (r[1], r[2]) if r[0] == MAPPED else (r[0], r[3])


def lane_runs(src_of):
    """The runs of lanes `(lo, hi, step, mul, add, div)` a MAPPED round's lanes fall into, each lane
    `l` reading lane `(l * mul + add) / div`."""
    return _get_lane_runs({lane: s for lane, s in enumerate(src_of) if s >= 0})


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
    """Every pass of a gathering move, `(source depth, destination depth, step)`, one per source row
    of a lane-full; a destination lapping the lanes takes its first lap only, which the lowering
    copies in-lane into the rest (lower_xlu_moves._copy_first_lap)."""
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
    return [(i, i * at, step) for i in range(rows)]

def _laps(shape, lanes):
    if shape[0] <= lanes:
        return 1
    return None if shape[0] % lanes else shape[0] // lanes

def permute_groups(rounds):
    """The rounds that share ONE pattern: `(mul, add, lo, hi, div, step, runs)`, or for MAPPED rounds
    `(MAPPED, the source lane of each lane, runs)`, a run being `(source row, its step, destination
    row, its step, how many)`."""
    by, order = {}, []
    for r in rounds:
        if r[0] == MAPPED:
            key, at, depth = (MAPPED, tuple(r[3:])), r[1], r[2]
        else:
            at, mul, add, depth, lo, hi, div, step = r
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


def permute_runs(rounds, row, per_reg, lane_reg=None):
    """Each run a permute's rounds emit: (its pattern (mul, add, lo, hi, div, every), or (MAPPED, the
    source lane of each lane), the run (s0, ss, d0, ds, n), its rows a push, whether it writes only
    some of the `row` lanes). Such a round's lane numbers are i64, so its push spans `lane_reg`."""
    out = []
    for group in permute_groups(rounds):
        if group[0] == MAPPED:
            pattern, runs = group[:2], group[2]
            part = not all(s >= 0 for s in group[1][:row])
        else:
            pattern, runs = group[:6], group[6]
            _, _, lo, hi, _, every = pattern
            part = bool(lo) or hi != row or every != 1
        cap = min(per_reg, lane_reg) if part and lane_reg else per_reg
        for s0, ss, d0, ds, n in runs:
            w = min(rows_per_vector(ss, n, cap), rows_per_vector(ds, n, cap))
            out.append((pattern, (s0, ss, d0, ds, n), w, part))
    return out


def crossings_of(bits):
    """(the 32-bit words an element of `bits` crosses the unit in, why not when it cannot)."""
    if bits <= 32:
        return 1, None
    if bits != 64:
        return 0, f"its element is {bits} bits and the unit carries 32 at a time"
    return 2, None


def plan_of(mode, pairs, s_shape, d_shape, lanes, src_uniform, src_lanes):
    """The rounds the move `mode` runs for this correspondence on `lanes` lanes, as integer tuples, or None
    where it does not carry it; `src_uniform` is whether the source's buffer is one lane's,
    `src_lanes` how many lanes the source's own elements ride (the caller's view_lanes)."""
    if mode == XLU_TRANSPOSE:
        extent = src_lanes
        numel = 1
        for d in s_shape:
            numel *= d
        return crossing_passes(fold_onto_lanes(pairs, s_shape, d_shape, lanes),
                               min(extent, lanes), need_per_lane(numel, extent, lanes), lanes)
    if mode == XLU_PERMUTE:
        return rounds_of_pairs(fold_onto_lanes(pairs, s_shape, d_shape, lanes), lanes)
    if mode == XLU_GATHER:
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


def choose_mode(pairs, s_shape, d_shape, lanes, src_uniform, src_lanes, modes=XLU_MODES):
    """`(the move among `modes` of fewest rounds that carries these (from, to) pairs, its plan)`, or None
    where none does."""
    if pairs is None or not lanes:
        return None
    ranked = [(len(plan), XLU_MODES.index(m), m, plan) for m in modes
              for plan in [plan_of(m, pairs, s_shape, d_shape, lanes, src_uniform, src_lanes)] if plan is not None]
    return min(ranked)[2:] if ranked else None


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


def build_push_rows(bank, at, step, rows, in_ty, rvl, h, bits, emit, pad):
    """`rows` rows of `bank` from row `at`, `step` apart, pushed `in_ty` at a time: word `h`, each by
    `emit(word, rvl)`."""
    nr = ir.VectorType(in_ty).shape[0]
    fits = rows_fit(bank, at, step, rows, nr)

    def push(i):
        v = transfer_read(in_ty, bank, [const_index(0), row_at(at, i, nr * step)], pad, fits)
        emit(crossing_words(as_type(v, word_vector(in_ty)), bits)[h], rvl)
    build_row_loop(rows // nr, push)


def build_pop_rows(bank, at, step, rows, out_ty, rvl, h, bits, crossings, emit, pad):
    """`rows` rows popped into `bank` from row `at`, `step` apart, `out_ty` at a time: word `h`,
    joined to the words before it, each by `emit(type, rvl)`."""
    nr = ir.VectorType(out_ty).shape[0]
    cross_ty = ir.VectorType.get([nr], ir.IntegerType.get_signless(32))
    fits = rows_fit(bank, at, step, rows, nr)

    def pop(i):
        got = emit(cross_ty, rvl)
        row = row_at(at, i, nr * step)
        v = element_from_words(
            got, h, bits, word_vector(out_ty),
            None if crossings == 1 or not h else as_type(
                transfer_read(out_ty, bank, [const_index(0), row], pad, fits), word_vector(out_ty)))
        transfer_write(as_type(v, out_ty), bank, [const_index(0), row], fits)
    build_row_loop(rows // nr, pop)


def as_type(v, ty):
    return v if v.type == ty else bitcast(v, ty)
