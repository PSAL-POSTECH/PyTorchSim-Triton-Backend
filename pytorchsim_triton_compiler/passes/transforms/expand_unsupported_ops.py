"""pytorchsim-expand-unsupported-ops: no scan, no bodiless extern and no math op that is built from
others (DERIVE, UPSTREAM_EXPANDS) leaves here; every other math op waits for the machine's lowering."""

from collections import namedtuple

from ..utils.linear_layout import (
    compose_reshape, cut_lanes, drop_dims, get_bases, spell, widen, without_idle_registers)
from ..utils.ir_utils import elementwise_generic, funcs, int_attr, is_shaped, lanes_of_module, op_of, walk
from ..utils.math_ops import DERIVE, UPSTREAM_EXPANDS, Native

_SCAN = ("tt.scan", "tt.scan.return", "ttx.cumsum")

FORBIDS = (("tt.extern_elementwise", "tt.gather") + _SCAN
           + tuple("math." + n for n in sorted(set(DERIVE) | set(UPSTREAM_EXPANDS))))

_EXTERN_ALIAS = {
    "rcbrtf": "rcbrt", "erfinvf": "erfinv",
    "erfcinvf": "erfcinv", "exp10f": "exp10",
    "fdimf": "fdim", "fmodf": "fmod", "hypotf": "hypot", "rhypotf": "rhypot",
    "ilogbf": "ilogb", "logbf": "logb",
    "nextafterf": "nextafter", "normcdff": "normcdf",
    "normcdfinvf": "normcdfinv", "remainderf": "remainder",
    "saturatef": "saturate", "scalbnf": "scalbn", "ldexpf": "scalbn",
    "signbitf": "signbit",
    "sinpif": "sinpi", "cospif": "cospi",
    "erfcxf": "erfcx", "lgammaf": "lgamma", "tgammaf": "tgamma",
    "j0f": "j0", "j1f": "j1", "y0f": "y0", "y1f": "y1",
    "cyl_bessel_i0f": "i0", "cyl_bessel_i1f": "i1",
}

_ScanTile = namedtuple("_ScanTile", "shape wide home axis rank tile_tys ones")


class _Fatal(Exception):
    pass


def run(module, **_opts):
    """The externs the linalg lowering left bodiless, derived; upstream's math-expand-ops for the ops it
    builds; then each math op DERIVE builds, and scans, here. Any other math op waits for the machine."""
    from mlir.ir import Location
    from mlir.passmanager import PassManager

    loc = Location.unknown()
    with loc:
        changed = sum(_rewrite_externs(func, loc) for func in funcs(module))
    changed += _get_upstream_expand_count(module)
    PassManager.parse("builtin.module(func.func(math-expand-ops{ops=%s}))"
                      % ",".join(UPSTREAM_EXPANDS)).run(module.operation)
    with loc:
        for func in funcs(module):
            changed += _rewrite_derived_math(func, loc)
            changed += _rewrite_scans(module, func, loc)
            changed += _rewrite_gathers(func)
        if changed:
            PassManager.parse(
                "builtin.module(func.func(canonicalize,cse))").run(
                    module.operation)
    return module


def _rewrite_externs(func, loc):
    changed = 0
    for op in list(walk(func.operation)):
        if op.name != "tt.extern_elementwise":
            continue
        new = _build_derivation(op, _get_canon_of_extern(op), loc)
        op.results[0].replace_all_uses_with(new)
        op_of(op).erase()
        changed += 1
    return changed


def _get_canon_of_extern(op):
    from mlir.ir import StringAttr
    sym = StringAttr(op.attributes["symbol"]).value
    canon = _get_canon_of_symbol(sym)
    if canon is None:
        raise _Fatal(
            f"the linalg lowering left {sym} as a tt.extern_elementwise -- its "
            f"ExternElementwise converters do not know the symbol -- and "
            f"neither does this pass. Nothing downstream can hold a tt.* "
            f"op: add it to _EXTERN_ALIAS: {op}")
    if canon not in DERIVE:
        raise _Fatal(
            f"{sym} spells the canon {canon!r}, which DERIVE does not build: add "
            f"the derivation, or, if the symbol IS a math op, map it in "
            f"the linalg lowering's ExternElementwise table: {op}")
    return canon


def _get_canon_of_symbol(sym):
    base = sym[5:] if sym.startswith("__nv_") else sym
    if base in _EXTERN_ALIAS:
        return _EXTERN_ALIAS[base]
    return _EXTERN_ALIAS.get(base + "f")


def _get_upstream_expand_count(module):
    names = {"math." + n for n in UPSTREAM_EXPANDS}
    return sum(op.name in names for func in funcs(module) for op in walk(func.operation))


def _rewrite_derived_math(func, loc):
    changed = 0
    for op in list(walk(func.operation)):
        name = op.name
        if not name.startswith("math.") or name.split(".", 1)[1] not in DERIVE:
            continue
        new = _build_derivation(op, name.split(".", 1)[1], loc)
        op.results[0].replace_all_uses_with(new)
        op_of(op).erase()
        changed += 1
    return changed


def _rewrite_gathers(func):
    """Each tt.gather, this machine's pytorchsim.gather into a fresh destination: the layouts below decide
    which unit carries it."""
    from mlir.dialects import bufferization
    from mlir.ir import InsertionPoint, IntegerAttr, IntegerType
    from ...dialect import pytorchsim
    gathers = [op for op in walk(func.operation) if op.name == "tt.gather"]
    for op in gathers:
        res = op.results[0]
        with InsertionPoint(op):
            init = bufferization.AllocTensorOp(res.type, []).result
            got = pytorchsim.GatherOp(res.type, op.operands[0], op.operands[1], init,
                                    IntegerAttr.get(IntegerType.get_signless(64), int_attr(op, "axis"))).result
        res.replace_all_uses_with(got)
        op.operation.erase()
    return len(gathers)


def _rewrite_scans(module, func, loc):
    """Each scan, blocked on the lanes the module was made for (read only when it has a scan) when
    its axis is a multiple of them longer than them, else a ladder."""
    scans = [op for op in walk(func.operation) if op.name in ("ttx.cumsum", "tt.scan")]
    base = lanes_of_module(module) if scans else None
    return sum(_rewrite_scan(op, loc, base) for op in scans)


def _rewrite_scan(op, loc, base):
    from mlir.ir import InsertionPoint
    nval = len(op.results)
    rev = "reverse" in op.attributes and _get_bool_attr(op, "reverse")
    with InsertionPoint(op):
        outs = _build_scan(list(op.operands)[:nval], int_attr(op, "axis"), rev,
                           _get_scan_combiner(op), loc, base)
    for res, out in zip(op.results, outs):
        res.replace_all_uses_with(out)
    op.operation.erase()
    return 1


def _get_bool_attr(op, name):
    from mlir.ir import BoolAttr
    return BoolAttr(op.attributes[name]).value


def _build_scan(srcs, axis, rev, combine, loc, base):
    from mlir.ir import ShapedType
    n = ShapedType(srcs[0].type).shape[axis]
    if n <= base or n % base:
        return _build_scan_ladder(srcs, axis, rev, combine, loc)
    return _build_blocked_scan(srcs, axis, rev, combine, loc, base)


def _build_scan_ladder(srcs, axis, rev, combine, loc):
    from mlir.dialects import arith, linalg, tensor
    from mlir.ir import IndexType, IntegerAttr, RankedTensorType, ShapedType

    shape = list(ShapedType(srcs[0].type).shape)
    elems = [ShapedType(v.type).element_type for v in srcs]
    nval, rank = len(elems), len(shape)
    n = shape[axis]
    shifts = _get_hillis_steele_shifts(n)
    if not shifts:
        return srcs

    pad = shifts[-1]
    home = 0 if rev else pad
    enc = _get_encoding_of(srcs[0])
    tile = _ScanTile(shape, shape[:axis] + [pad + n] + shape[axis + 1:], home,
                     axis, rank, [RankedTensorType.get(shape, e, enc) for e in elems],
                     [1] * rank)
    idx = IndexType.get()
    keep_when = arith.CmpIPredicate.ult if rev else arith.CmpIPredicate.uge

    wide_enc = _get_widened_encoding(enc, axis, pad + n)
    bufs = _build_scan_put(tile, srcs, [_build_empty(tile.wide, e, wide_enc) for e in elems])
    for k in shifts:
        kc = arith.ConstantOp(idx, IntegerAttr.get(idx, n - k if rev
                                                   else k)).result
        here = _build_scan_cut(tile, bufs, home)
        back = _build_scan_cut(tile, bufs, home + k if rev else home - k)

        def step(*args, kc=kc):
            """One rung: combine this slice with the shifted one, and keep the combined value only
            where the shift stayed in the tile."""
            a, b = list(args[:nval]), list(args[nval:])
            merged = combine(b, a)
            i = linalg.IndexOp(axis).result
            keep = arith.CmpIOp(keep_when, i, kc).result
            return [arith.SelectOp(keep, m, x).result
                    for m, x in zip(merged, a)]

        res = elementwise_generic(here + back, elems, step, loc, shape)
        bufs = _build_scan_put(tile, res, bufs)
    return _build_scan_cut(tile, bufs, home)


def _get_hillis_steele_shifts(n):
    shifts, d = [], 1
    while d < n:
        shifts.append(d)
        d *= 2
    return shifts


def _get_encoding_of(v):
    from mlir.ir import RankedTensorType
    return RankedTensorType(v.type).encoding


def _get_widened_encoding(enc, axis, extent):
    from mlir.ir import Attribute
    bases = get_bases(enc) if enc is not None else None
    if bases is None:
        return enc
    return Attribute.parse(spell(widen(bases, axis, extent)))


def _build_empty(shape, elem, enc):
    from mlir.ir import Operation, RankedTensorType
    return Operation.create("tensor.empty", results=[RankedTensorType.get(shape, elem, enc)]).results[0]


def _build_scan_put(t, vals, bufs):
    from mlir.dialects import tensor
    return [tensor.InsertSliceOp(v, b, [], [], [],
                                 _get_scan_offsets(t, t.home), t.shape, t.ones).result
            for v, b in zip(vals, bufs)]


def _get_scan_offsets(t, offset):
    return ([0] * t.axis + [offset] + [0] * (t.rank - t.axis - 1))


def _build_scan_cut(t, bufs, offset):
    from mlir.dialects import tensor
    return [tensor.ExtractSliceOp(ty, b, [], [], [],
                                  _get_scan_offsets(t, offset), t.shape, t.ones).result
            for ty, b in zip(t.tile_tys, bufs)]


def _build_blocked_scan(srcs, axis, rev, combine, loc, base):
    from mlir.dialects import arith, linalg, tensor
    from mlir.ir import (
        AffineDimExpr, AffineMap, AffineMapAttr, Attribute, IndexType, IntegerAttr, RankedTensorType,
        ShapedType)
    shape = list(ShapedType(srcs[0].type).shape)
    elems = [ShapedType(v.type).element_type for v in srcs]
    nval, rank, rows = len(elems), len(shape), shape[axis] // base
    folded = shape[:axis] + [rows, base] + shape[axis + 1:]
    groups = ([[d] for d in range(axis)] + [[axis, axis + 1]]
              + [[d + 1] for d in range(axis + 1, rank)])
    bases = get_bases(RankedTensorType(srcs[0].type).encoding)
    if bases is None:
        raise _Fatal("a scan to fold whose layout is no linear one: %s" % srcs[0].type)
    fold_bases = compose_reshape(bases, shape, folded)
    fold = Attribute.parse(spell(fold_bases))
    per_row_enc = Attribute.parse(spell(without_idle_registers(drop_dims(fold_bases, [axis + 1]))))
    tot_enc = Attribute.parse(spell(without_idle_registers(cut_lanes(fold_bases, axis + 1, 1))))
    wide = [tensor.ExpandShapeOp(RankedTensorType.get(folded, e, fold), v, groups, [],
                                 folded).result for v, e in zip(srcs, elems)]
    inner = _build_scan(wide, axis + 1, rev, combine, loc, base)
    col = 0 if rev else base - 1
    tot = folded[:axis + 1] + [1] + folded[axis + 2:]
    totals = [tensor.ExtractSliceOp(
        RankedTensorType.get(tot, e, tot_enc), v, [], [], [],
        [0] * (axis + 1) + [col] + [0] * (rank - axis - 1), tot, [1] * (rank + 1)).result
        for v, e in zip(inner, elems)]
    before = _build_scan_shift(_build_scan(totals, axis, rev, combine, loc, base), axis, rev)
    per_row = shape[:axis] + [rows] + shape[axis + 1:]
    before = [tensor.CollapseShapeOp(RankedTensorType.get(per_row, e, per_row_enc), v, groups).result
              for v, e in zip(before, elems)]

    ident = AffineMapAttr.get(AffineMap.get_identity(rank + 1))
    spread = AffineMapAttr.get(AffineMap.get(rank + 1, 0, [
        AffineDimExpr.get(d) for d in range(rank + 1) if d != axis + 1]))
    idx = IndexType.get()
    first = arith.ConstantOp(idx, IntegerAttr.get(idx, rows - 1 if rev else 0)).result

    def join(*args):
        """Every row but the first takes the total before it on its left."""
        a, b = list(args[:nval]), list(args[nval:])
        merged = combine(b, a)
        at_first = arith.CmpIOp(arith.CmpIPredicate.eq, linalg.IndexOp(axis).result,
                                first).result
        return [arith.SelectOp(at_first, x, m).result for m, x in zip(merged, a)]

    done = elementwise_generic(inner + before, elems, join, loc, folded,
                               maps=[ident] * nval + [spread] * nval + [ident] * nval)
    enc = _get_encoding_of(srcs[0])
    return [tensor.CollapseShapeOp(RankedTensorType.get(shape, e, enc), v, groups).result
            for v, e in zip(done, elems)]


def _build_scan_shift(vals, axis, rev):
    from mlir.dialects import tensor
    from mlir.ir import RankedTensorType, ShapedType
    shape = list(ShapedType(vals[0].type).shape)
    elems = [ShapedType(v.type).element_type for v in vals]
    home = 0 if rev else 1
    enc = _get_encoding_of(vals[0])
    tile = _ScanTile(shape, shape[:axis] + [shape[axis] + 1] + shape[axis + 1:], home,
                     axis, len(shape), [RankedTensorType.get(shape, e, enc) for e in elems],
                     [1] * len(shape))
    wide_enc = _get_widened_encoding(enc, axis, shape[axis] + 1)
    bufs = _build_scan_put(tile, vals, [_build_empty(tile.wide, e, wide_enc) for e in elems])
    return _build_scan_cut(tile, bufs, home + 1 if rev else home - 1)


def _get_scan_combiner(op):
    from mlir.dialects import arith
    from mlir.ir import IntegerType, ShapedType

    if op.name == "ttx.cumsum":
        elem = ShapedType(op.operands[0].type).element_type
        try:
            IntegerType(elem)
        except (TypeError, ValueError):
            return lambda a, b: [arith.AddFOp(a[0], b[0]).result]
        return lambda a, b: [arith.AddIOp(a[0], b[0]).result]

    blk = op.regions[0].blocks[0]
    body = [o.operation for o in blk.operations][:-1]

    def build(a, b):
        """The cloned body with `a` and `b` in its block arguments' places."""
        m = dict(zip(blk.arguments, list(a) + list(b)))
        for inner in body:
            c = inner.clone()
            for i, v in enumerate(c.operands):
                if v in m:
                    c.operands[i] = m[v]
            m.update(zip(inner.results, c.results))
        term = list(blk.operations)[-1]
        return [m[v] for v in term.operands]

    return build


def _build_derivation(op, canon, loc):
    from mlir.ir import InsertionPoint, ShapedType
    arity, rule = DERIVE[canon]
    args, res = list(op.operands), op.results[0]
    if len(args) != arity:
        raise _Fatal(f"{op.name} takes {arity} operands here, got {len(args)}: {op}")
    with InsertionPoint(op_of(op)):
        if not is_shaped(res):
            return _build_in_result_type(rule(Native(args[0].type), *args), res.type, op)
        st = ShapedType(res.type)
        f = Native(ShapedType(args[0].type).element_type)
        return elementwise_generic(
            args, st.element_type,
            lambda *ins: _build_in_result_type(rule(f, *ins), st.element_type, op),
            loc, list(st.shape))


def _build_in_result_type(val, want, op):
    from mlir.ir import IntegerType
    from mlir.dialects import arith
    if val.type == want:
        return val
    try:
        vw = IntegerType(val.type).width
        ww = IntegerType(want).width
    except (TypeError, ValueError):
        raise _Fatal(
            f"derivation produced {val.type} where {want} is required, and "
            f"the two are not both integers -- no widening applies: {op}")
    if vw < ww:
        return arith.ExtUIOp(want, val).result
    if vw > ww:
        return arith.TruncIOp(want, val).result
    return val
