"""tritonpytorchsim-pack-reduction-rows: a reduce loop over a lane's bank rows whose vector is one
whole row takes d contiguous rows per trip, then folds the d rows into one inside the lane. d is
ranked over the divisors whose widest vector fits vlen*lmul; d=1 is a candidate and leaves it."""

import math
import os

from .lib_bank_layout import bank_strides, lanes_of
from .lib_ir import (attrs_of, const_index, elem_bits, funcs, is_memref, op_of,
                     strides_and_offset, walk)
from .p05_vectorize import REDUCE_KIND, VECTORISABLE

NAME = "tritonpytorchsim-pack-reduction-rows"

_LOAD = "affine.vector_load"


class _NoPack(Exception):
    pass


def run(module, *, vlen, lmul, **_opts):
    """Every loop that carries vectors through a combiner is read from its own body; none by name."""
    from mlir.ir import Location
    with Location.unknown():
        for func in funcs(module):
            for lp in [o for o in walk(func.operation) if o.name == "affine.for"]:
                try:
                    shape = _read_loop(lp)
                except _NoPack:
                    continue
                d = _rank_row_counts(shape, vlen, lmul)
                if d > 1:
                    _rewrite_packed(lp, shape, d)
    return module


def _read_loop(lp):
    """What packing needs to know about `lp`, or _NoPack: trip count, row width, the loads that
    walk it row by row, and the combiner each carried vector goes through."""
    from mlir.ir import AffineConstantExpr, AffineMapAttr, IntegerAttr, VectorType
    if len(lp.results) == 0 or IntegerAttr(lp.attributes["step"]).value != 1:
        raise _NoPack
    lo, hi = (AffineMapAttr(lp.attributes[k]).value for k in ("lowerBoundMap", "upperBoundMap"))
    if lo.n_inputs or hi.n_inputs or len(lo.results) != 1 or len(hi.results) != 1:
        raise _NoPack
    if not all(isinstance(e, AffineConstantExpr) for e in (lo.results[0], hi.results[0])):
        raise _NoPack
    lo_v, trips = AffineConstantExpr(lo.results[0]).value, AffineConstantExpr(hi.results[0]).value
    if lo_v != 0:
        raise _NoPack
    blk = lp.regions[0].blocks[0]
    iv, carried = blk.arguments[0], list(blk.arguments)[1:]
    ops = [i.operation for i in blk.operations]
    *body, yld = ops
    width = None
    loads = {}
    for o in body:
        if o.regions and any(r.blocks for r in o.regions):
            raise _NoPack
        for v in list(o.operands) + list(o.results):
            try:
                vt = VectorType(v.type)
            except (TypeError, ValueError):
                continue
            if vt.rank != 1 or (width is not None and vt.shape[0] != width):
                raise _NoPack
            width = vt.shape[0]
        if o.name == _LOAD:
            loads[o] = _read_row_load(o, iv, width)
        elif o.name.split(".")[0] not in VECTORISABLE:
            raise _NoPack
        elif any(a == iv for a in o.operands):
            raise _NoPack
    if width is None or not loads:
        raise _NoPack
    combiners = []
    for i, y in enumerate(yld.operands):
        c = op_of(y.owner)
        if c is None or c.name not in REDUCE_KIND or c.operation.block != blk:
            raise _NoPack
        acc = carried[i]
        if acc not in list(c.operands) or len(list(acc.uses)) != 1:
            raise _NoPack
        combiners.append(c.operation)
    for a in carried:
        for u in a.uses:
            if op_of(u.owner).operation not in combiners:
                raise _NoPack
    return dict(trips=trips, width=width, loads=loads, combiners=combiners, body=body,
                widest=max(elem_bits(VectorType(v.type).element_type)
                           for o in body for v in list(o.results) if _is_vec(v)),
                n_ops=len(body) + 1)


def _is_vec(v):
    from mlir.ir import VectorType
    try:
        VectorType(v.type)
        return True
    except (TypeError, ValueError):
        return False


def _read_row_load(o, iv, width):
    """(axis, collapsed type) for a load whose index on axis `a` is the loop's iv, whose last axis
    is a whole row of `width`, contiguous in the bank after axis `a`."""
    from mlir.ir import AffineDimExpr, AffineMapAttr, MemRefType
    mem = o.operands[0]
    if not is_memref(mem) or lanes_of(mem) is None:
        raise _NoPack
    mt = MemRefType(mem.type)
    shape = list(mt.shape)
    amap = AffineMapAttr(o.attributes["map"]).value
    idx = list(o.operands)[1:]
    uses_iv = [ax for ax, e in enumerate(amap.results)
               if any(idx[p] == iv for p in range(len(idx)) if _mentions(e, p))]
    if len(uses_iv) != 1:
        raise _NoPack
    a = uses_iv[0]
    e = amap.results[a]
    if not (isinstance(e, AffineDimExpr) and idx[AffineDimExpr(e).position] == iv):
        raise _NoPack
    if a + 1 != len(shape) - 1 or shape[-1] != width:
        raise _NoPack
    st = bank_strides(mem)
    if st[-1] != 1 or st[a] != width or not _identity_layout(mt):
        raise _NoPack
    return a


def _mentions(expr, pos):
    from mlir.ir import AffineDimExpr, AffineBinaryExpr
    if isinstance(expr, AffineDimExpr):
        return AffineDimExpr(expr).position == pos
    if isinstance(expr, AffineBinaryExpr):
        b = AffineBinaryExpr(expr)
        return _mentions(b.lhs, pos) or _mentions(b.rhs, pos)
    return False


def _identity_layout(mt):
    so = strides_and_offset(mt)
    sh = list(mt.shape)
    want = [math.prod(sh[i + 1:]) for i in range(len(sh))]
    return so is not None and so[0] == want and so[1] == 0


def _rank_row_counts(s, vlen, lmul):
    """The row count d per trip with the fewest issued instructions: (trips/d) trips of the body,
    plus log2-ish fold steps; d divides the trips and its widest vector fits one register group."""
    forced = os.environ.get("EXP_PACK_ROWS")
    trips, width = s["trips"], s["width"]
    cands = [d for d in range(1, trips + 1)
             if trips % d == 0 and d * width * s["widest"] <= vlen * lmul] or [1]
    if forced:
        want = int(forced)
        return max(d for d in cands if d <= want)

    def issued(d):
        return (trips // d) * s["n_ops"] + 2 * _fold_steps(d) * len(s["combiners"])
    return min(cands, key=lambda d: (issued(d), d))


def _fold_steps(d):
    n = 0
    while d > 1:
        d = d // 2 if d % 2 == 0 else d - 1
        n += 1
    return n


def _rewrite_packed(lp, s, d):
    """Peel trip 0 as the packed seed, run trips 1..trips/d on d rows each, fold the d rows of each
    carried vector, then combine with the original seed in the original combiner."""
    from mlir.ir import (AffineDimExpr, AffineMap, AffineMapAttr, InsertionPoint, MemRefType,
                         Operation, VectorType)
    from mlir.dialects import affine, memref
    width, trips = s["width"], s["trips"]
    blk = lp.regions[0].blocks[0]
    iv, carried = blk.arguments[0], list(blk.arguments)[1:]
    seeds = list(lp.operands)[-len(carried):]
    views = {}
    with InsertionPoint(lp):
        for o, a in s["loads"].items():
            mem = o.operands[0]
            if mem in views:
                continue
            mt = MemRefType(mem.type)
            sh = list(mt.shape)
            new = sh[:a] + [sh[a] * sh[a + 1]]
            groups = [[i] for i in range(a)] + [[a, a + 1]]
            views[mem] = memref.CollapseShapeOp(
                MemRefType.get(new, mt.element_type, memory_space=mt.memory_space),
                mem, groups).result

    def packed_ty(v):
        vt = VectorType(v.type)
        return VectorType.get([width * d], vt.element_type)

    def emit_body(kk, accs):
        env = {iv: kk}
        outs = {}
        for o in s["body"]:
            if o in s["loads"]:
                a = s["loads"][o]
                amap = AffineMapAttr(o.attributes["map"]).value
                idx = list(o.operands)[1:]
                exprs = list(amap.results)
                p = AffineDimExpr(exprs[a]).position
                merged = AffineDimExpr.get(p) * (width * d) + exprs[a + 1]
                nm = AffineMap.get(amap.n_dims, amap.n_symbols, exprs[:a] + [merged])
                ops_in = [env.get(x, x) for x in idx]
                r = affine.AffineVectorLoadOp(packed_ty(o.results[0]), views[o.operands[0]],
                                              ops_in, AffineMapAttr.get(nm)).result
                env[o.results[0]] = r
                continue
            if o in s["combiners"]:
                i = s["combiners"].index(o)
                other = [x for x in o.operands if x != carried[i]][0]
                outs[i] = env.get(other, other)
                if accs is not None:
                    new = Operation.create(o.name, results=[packed_ty(o.results[0])],
                                           operands=[outs[i], accs[i]], attributes=attrs_of(o))
                    outs[i] = new.results[0]
                continue
            new = Operation.create(o.name,
                                   results=[packed_ty(r) if _is_vec(r) else r.type
                                            for r in o.results],
                                   operands=[env.get(x, x) for x in o.operands],
                                   attributes=attrs_of(o))
            env.update(zip(o.results, new.results))
        return [outs[i] for i in range(len(carried))]

    with InsertionPoint(lp):
        first = emit_body(const_index(0), None)
        loop = affine.AffineForOp(1, trips // d, 1, iter_args=first) if trips // d > 1 else None
    if loop is not None:
        with InsertionPoint(loop.body):
            got = emit_body(loop.induction_variable, list(loop.inner_iter_args))
            affine.AffineYieldOp(got)
        packed = list(loop.results)
    else:
        packed = first
    with InsertionPoint(lp):
        finals = []
        for i, (v, c) in enumerate(zip(packed, s["combiners"])):
            v = _fold_rows(v, c, d, width)
            finals.append(Operation.create(c.name, results=[VectorType(v.type)],
                                           operands=[v, seeds[i]],
                                           attributes=attrs_of(c)).results[0])
    for r, f in zip(lp.results, finals):
        r.replace_all_uses_with(f)
    lp.erase()


def _slice(src, off, n, elem):
    from mlir.ir import ArrayAttr, IntegerAttr, IntegerType, Operation, VectorType
    i64 = IntegerType.get_signless(64)
    return Operation.create(
        "vector.extract_strided_slice", results=[VectorType.get([n], elem)], operands=[src],
        attributes={"offsets": ArrayAttr.get([IntegerAttr.get(i64, off)]),
                    "sizes": ArrayAttr.get([IntegerAttr.get(i64, n)]),
                    "strides": ArrayAttr.get([IntegerAttr.get(i64, 1)])}).results[0]


def _fold_rows(v, comb, n, width):
    """`n` rows of `width` in one vector combined row-wise into one row, halving when even and
    folding the last row into the first when odd -- every step stays inside the lane."""
    from mlir.ir import DenseI64ArrayAttr, Operation, VectorType
    elem = VectorType(v.type).element_type

    def combine(x, y, k):
        return Operation.create(comb.name, results=[VectorType.get([k * width], elem)],
                                operands=[x, y], attributes=attrs_of(comb)).results[0]
    while n > 1:
        if n % 2 == 0:
            h = n // 2
            v = combine(_slice(v, 0, h * width, elem), _slice(v, h * width, h * width, elem), h)
        else:
            head = combine(_slice(v, 0, width, elem), _slice(v, (n - 1) * width, width, elem), 1)
            h = n - 1
            if h > 1:
                rest = _slice(v, width, (h - 1) * width, elem)
                v = Operation.create("vector.shuffle", results=[VectorType.get([h * width], elem)],
                                     operands=[head, rest],
                                     attributes={"mask": DenseI64ArrayAttr.get(
                                         list(range(h * width)))}).results[0]
            else:
                v = head
        n = h
    return v
