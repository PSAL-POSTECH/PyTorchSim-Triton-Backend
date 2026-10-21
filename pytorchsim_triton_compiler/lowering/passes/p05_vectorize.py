"""torchsim-vectorize: every elementwise linalg op and reduce -> the banked affine nest."""

import contextlib
import math
from collections import namedtuple

from .lib_bank_layout import bank_strides, in_scratchpad, is_uniform, lane_split, why_no_lane_axis
from .lib_cross_lane import CROSS_LANE_KIND, CROSS_LANE_OP, FOLD, FOLD_BY
from .lib_ir import (VIEW_OPS, attr_is, attrs_of, build_vector_body, const_index, ends_lifetime, funcs,
                     int_array, is_memref, name_of, op_of, walk)

VECTORISABLE = ("arith", "math")

VECTOR_LOOP = "torchsim.vector_loop"

VLANE_LOOP = "torchsim.vlane_loop"

REDUCE_LOOP = "torchsim.reduce_loop"
LANE_SPLAT = "torchsim.lane_splat"

_TARGETS = ("linalg.reduce", "linalg.generic", "linalg.fill")

FORBIDS = ("linalg.*",)

REDUCE_KIND = {
    "arith.addf": "ADD",   "arith.addi": "ADD",
    "arith.mulf": "MUL",   "arith.muli": "MUL",
    "arith.maxnumf": "MAXNUMF", "arith.maximumf": "MAXIMUMF",
    "arith.minnumf": "MINNUMF", "arith.minimumf": "MINIMUMF",
    "arith.maxsi": "MAXSI", "arith.minsi": "MINSI",
    "arith.maxui": "MAXUI", "arith.minui": "MINUI",
    "arith.andi": "AND",   "arith.ori": "OR",  "arith.xori": "XOR",
}

_Plan = namedtuple(
    "_Plan",
    "memrefs n_in body lane_dim depth lanes_used bounds steps vec_len vec_ty pos vec_dim bcast "
    "index_of strided reduce")

_Reduce = namedtuple("_Reduce", "dim extent along kind seeds rows")

_ABSORBED = "torchsim.absorbed_into_reduce"

class NotVectorisable(Exception):
    """The op is not the elementwise-on-one-iteration-space shape this widens."""


def _check_contract(holds, what, op):
    if not holds:
        raise NotVectorisable(f"an upstream pass broke its promise -- {what}: {op}")


def run(module, lanes, **_):
    """Every reduce and elementwise op in program order: its plan, its loops, its body."""
    from mlir.ir import Location
    from mlir.passmanager import PassManager

    with Location.unknown():
        _rewrite_lane_folds(module)
        _rewrite_trailing_units_of_reduces(module)
        seeded = _find_seeding_fills(module)
        for op in [o for o in walk(module.operation) if o.name in _TARGETS]:
            if _ABSORBED in op.attributes:
                op.erase()
                continue
            at = next((i for i, (r, _s) in enumerate(seeded) if r == op), None)
            plan = _get_loop_plan(op, lanes, seeded.pop(at)[1] if at is not None else {})
            with _build_loops(op, plan) as nest:
                _build_body(op, plan, nest)
            op.erase()
    PassManager.parse("builtin.module(func.func(canonicalize,cse))").run(module.operation)
    _check_nothing_unbanked(module)
    return module


def _rewrite_lane_folds(module):
    """A reduce whose input rides the lanes and whose result sits in one bank folds the lanes away: it
    is no nest but a move across the lanes, of kind FOLD, for the machine's lowering, which combines
    its result with the destination's by the body's one op (FOLD_BY). The anchor keeps a reduce on
    the lanes only when the machine folds it, so only the lane axis is reduced and the body is that
    combine alone; anything else broke that promise."""
    from mlir.ir import InsertionPoint, Operation, StringAttr
    for red in [o for o in walk(module.operation) if o.name == "linalg.reduce"]:
        if len(red.operands) != 2 or not all(is_memref(v) for v in red.operands):
            continue
        src, dst = red.operands
        if not (in_scratchpad(src.type) and in_scratchpad(dst.type)
                and is_uniform(dst) and not is_uniform(src)):
            continue
        _check_contract(list(int_array(red.attributes["dimensions"])) == [0],
                        "a reduce across the lanes reduces the lane axis alone", red)
        body = [i.operation for i in red.regions[0].blocks[0].operations]
        args = list(red.regions[0].blocks[0].arguments)
        comb = body[0] if len(body) == 2 else None
        _check_contract(comb is not None and set(comb.operands) == set(args)
                        and list(body[1].operands) == [comb.results[0]],
                        "a reduce across the lanes is one combine of its element and its init", red)
        with InsertionPoint(red):
            move = Operation.create(CROSS_LANE_OP, operands=[src, dst])
        move.attributes[CROSS_LANE_KIND] = StringAttr.get(FOLD)
        move.attributes[FOLD_BY] = StringAttr.get(comb.name)
        red.erase()


def _rewrite_trailing_units_of_reduces(module):
    """A linalg.reduce whose input and output both end in unit dims it does not reduce reads and
    writes views without them: its last dim is then the one a lane's vector walks, so the reduce
    over it is a vector reduce, not one element a step."""
    from mlir.dialects import memref
    from mlir.ir import InsertionPoint, MemRefType, Operation
    for red in [o for o in walk(module.operation) if o.name == "linalg.reduce"]:
        if len(red.operands) != 2 or not all(is_memref(v) for v in red.operands):
            continue
        src, dst = red.operands
        ss, ds = list(MemRefType(src.type).shape), list(MemRefType(dst.type).shape)
        dims = list(int_array(red.attributes["dimensions"]))
        k = 0
        while (k < len(ds) - 1 and k < len(ss) - 1 and ss[-1 - k] == 1 and ds[-1 - k] == 1
               and len(ss) - 1 - k not in dims):
            k += 1
        if not k or any(d >= len(ss) - k for d in dims):
            continue
        views = []
        with InsertionPoint(red):
            for v, shape in ((src, ss), (dst, ds)):
                groups = [[i] for i in range(len(shape) - k - 1)] + [list(range(len(shape) - k - 1, len(shape)))]
                views.append(memref.CollapseShapeOp(_collapsed_type(v, len(shape) - k), v, groups).result)
        red.operands[0], red.operands[1] = views
    return module


def _collapsed_type(v, rank):
    """`v`'s memref type with its trailing unit dims dropped to `rank` dims, the same strides kept."""
    from mlir.ir import AffineMapAttr, MemRefType, StridedLayoutAttr
    t = MemRefType(v.type)
    if isinstance(t.layout, AffineMapAttr):
        layout = None           # the identity layout stays the identity
    else:
        strides, offset = t.get_strides_and_offset()
        layout = StridedLayoutAttr.get(offset, strides[:rank])
    return MemRefType.get(list(t.shape)[:rank], t.element_type, layout=layout, memory_space=t.memory_space)


def _find_seeding_fills(module):
    from mlir.ir import UnitAttr
    out = []
    for red in [o for o in walk(module.operation)
                if o.name == "linalg.reduce" or _generic_reduced(o)]:
        n = len(red.operands) // 2 if red.name == "linalg.reduce" else _n_ins(red)
        reduced = list(int_array(red.attributes["dimensions"])) if red.name == "linalg.reduce" \
            else _generic_reduced(red)
        maps, _r, extent = _find_maps(red, [v for v in red.operands if is_memref(v)])
        if _folds_nothing(reduced, extent):
            continue            # walked elementwise: its init is read from the buffer the fill wrote
        seeds = {}
        for k in range(n, len(red.operands)):
            if any(d in maps[k] for d in reduced):
                continue        # a tile written beside the reduce, not one it combines into
            dst = red.operands[k]
            users = [op_of(u.owner) for u in dst.uses if not ends_lifetime(op_of(u.owner), dst)]
            if any(u.operation.block != red.operation.block or u.name in VIEW_OPS for u in users):
                continue
            before = [u for u in users if u.operation.is_before_in_block(red.operation)]
            last = before[0] if before else None
            for u in before[1:]:
                if last.operation.is_before_in_block(u.operation):
                    last = u
            if name_of(last) == "linalg.fill" and last.operands[1] == dst:
                seeds[k] = last.operands[0]
                last.operation.attributes[_ABSORBED] = UnitAttr.get()
        if seeds:
            out.append((red, seeds))
    return out


def _get_loop_plan(op, lanes, seeds):
    from mlir.ir import MemRefType, VectorType
    memrefs = [v for v in op.operands if is_memref(v)]
    n_in = len(op.operands) // 2 if op.name == "linalg.reduce" else \
        _n_ins(op) if _generic_reduced(op) else 0
    shape = list(MemRefType(memrefs[-1].type).shape)
    elem = MemRefType(memrefs[-1].type).element_type
    maps, reduced, iter_shape = _find_maps(op, memrefs)

    one_lane = is_uniform(memrefs[-1])
    for v, m in zip(memrefs, maps):
        spad = in_scratchpad(v.type)
        _check_contract(spad, "" if spad else why_no_lane_axis(v), op)
        _check_contract(one_lane or m[:1] == maps[-1][:1] or (is_uniform(v) and m[:1] == [None]),
                        "every operand of one op rides its lane dim, or is UNIFORM read at row 0",
                        op)
        _check_contract(not one_lane or is_uniform(v),
                        "every operand of a UNIFORM op is UNIFORM", op)
    lane_dim = None if one_lane else maps[-1][0]
    try:
        depth, lanes_used = lane_split(shape, one_lane, lanes)
    except ValueError as e:
        raise NotVectorisable(f"{e}: {op}")

    nord = [a for a in range(len(shape)) if shape[a] != 1]
    bounds = [depth if (a == 0 and not one_lane) else shape[a] for a in nord]
    if not bounds:
        bounds, nord = [1], [None]
    vec_len = bounds[-1]
    steps = [1] * (len(bounds) - 1) + [vec_len]
    pos = {maps[-1][a]: p for p, a in enumerate(nord) if a is not None}
    vec_dim = maps[-1][nord[-1]] if nord[-1] is not None else None

    reduce = None
    if reduced and _folds_nothing(reduced, iter_shape):
        reduced = []            # out = combine(in, init) per element: walked as an elementwise op
    if reduced:
        _check_contract(len(shape) > 0, "a reduce's result has an axis to write", op)
        real = [d for d in reduced if iter_shape[d] != 1]
        _check_contract(len(real) <= 1, "a reduce reduces exactly one dim", op)
        dim = real[0] if real else reduced[0]
        if real:
            pos[dim] = len(bounds)
        rows = _row_outputs(maps, n_in, dim)
        _check_contract(len(rows) == len(maps) - n_in or op.name == "linalg.generic",
                        "only a generic writes tiles beside its reduce", op)
        kind = _get_reduce_kind(op, n_in) if n_in == 1 and len(rows) == len(maps) - n_in else \
            _get_reduce_kind(op, n_in, rows[0] - n_in) if op.name == "linalg.generic" \
            and len(rows) == 1 else None
        _check_contract(kind is not None or len(rows) == len(maps) - n_in,
                        "a generic writing tiles beside its reduce combines with one op", op)
        # The operands that walk the reduced dim decide, not the first one: an input made per row
        # (a row's max, its index) never walks it and is splatted along it like any other.
        last = bool(walk := [m for m in maps if dim in m]) and all(m[-1] == dim for m in walk)
        along = last and (kind is not None or _is_pure_combiner(op, n_in))
        if last:
            steps, vec_dim = None, dim if along else None
            vec_len = iter_shape[dim] if along else 1
        reduce = _Reduce(dim, iter_shape[dim], along, kind,
                         {k: v for k, v in seeds.items() if k in rows}, rows)

    if reduce is None:
        n = _contiguous_run(op, memrefs, maps, nord, bounds)
        if n > 1:
            for a in nord[-n:-1]:
                del pos[maps[-1][a]]
            bounds = bounds[:-n] + [math.prod(bounds[-n:])]
            vec_len = bounds[-1]
            steps = [1] * (len(bounds) - 1) + [vec_len]
            pos[vec_dim] = len(bounds) - 1

    bcast = {k: vec_dim is None or vec_dim not in m for k, m in enumerate(maps)}
    index_of = [[pos.get(d) for d in m] for m in maps]

    vec_pos = len(bounds) if reduce and reduce.along else len(bounds) - 1
    strided = {}
    for k in range(len(memrefs)):
        walked = [a for a, p in enumerate(index_of[k]) if p == vec_pos]
        if vec_len > 1 and not bcast[k] and walked:
            st = bank_strides(memrefs[k])[walked[0]]
            if st != 1:
                strided[k] = st

    body = None if op.name == "linalg.fill" else op.regions[0].blocks[0]
    for inner in (body.operations if body is not None else []):
        nm = inner.operation.name
        if nm not in ("linalg.index", "linalg.yield") and nm.split(".")[0] not in VECTORISABLE:
            raise NotVectorisable(f"{nm} in a linalg body is not known elementwise: {op}")
    return _Plan(memrefs, n_in, body, lane_dim, depth, lanes_used, bounds, steps, vec_len,
                 VectorType.get([vec_len], elem), pos, vec_dim, bcast, index_of, strided, reduce)


def _contiguous_run(op, memrefs, maps, nord, bounds):
    """How many of an elementwise nest's trailing loops walk ONE run of every operand's bank: every
    operand maps the dims alike over one shape, the last steps 1, and each one before steps the
    elements of the loops inside it -- one vector then covers them all, and legalize cuts it to
    what an instruction issues."""
    from mlir.ir import MemRefType
    body = None if op.name == "linalg.fill" else op.regions[0].blocks[0]
    if body is not None and any(o.operation.name == "linalg.index" for o in body.operations):
        return 1
    shape = list(MemRefType(memrefs[-1].type).shape)
    if nord[-1] is None or any(m != maps[-1] for m in maps) \
            or any(list(MemRefType(v.type).shape) != shape for v in memrefs):
        return 1
    strides = [bank_strides(v) for v in memrefs]
    if any(s[nord[-1]] != 1 for s in strides):
        return 1
    n = 1
    while n < len(nord) and all(s[nord[-n - 1]] == math.prod(bounds[-n:]) for s in strides):
        n += 1
    # a run is worth one vector only if it is longer than the innermost loop it replaces: loops
    # of one trip around it add no element, and merging them only rewrites the nest
    while n > 1 and bounds[-n] == 1:
        n -= 1
    return n


def _find_maps(op, memrefs):
    from mlir.ir import AffineConstantExpr, AffineDimExpr, ArrayAttr, AffineMapAttr, MemRefType
    shapes = [list(MemRefType(v.type).shape) for v in memrefs]
    reduced = []
    if op.name == "linalg.generic":
        reduced = _generic_reduced(op)
        _check_contract(len(reduced) <= 1,
                        "a generic walks only parallel dims, or one reduction dim", op)
        maps = []
        for m in ArrayAttr(op.attributes["indexing_maps"]):
            dims = [None if isinstance(r, AffineConstantExpr) and r.value == 0
                    else r.position if isinstance(r, AffineDimExpr) else -1
                    for r in AffineMapAttr(m).value.results]
            real = [d for d in dims if d is not None]
            _check_contract(-1 not in real and len(set(real)) == len(real),
                            "every map is a projected permutation", op)
            maps.append(dims)
    elif op.name == "linalg.reduce":
        rank = len(shapes[0])
        reduced = list(int_array(op.attributes["dimensions"]))
        n = len(memrefs) // 2
        maps = [list(range(rank))] * n + [[d for d in range(rank) if d not in reduced]] * n
    else:
        maps = [list(range(len(shapes[-1])))]
    out = maps[-1]
    nd = len(out) + len(reduced)
    _check_contract(sorted(out + reduced) == list(range(nd)), "the outs map every kept dim once", op)
    extent = [1] * nd
    for m, sh in zip(maps, shapes):
        for ax, d in enumerate(m):
            if d is not None and sh[ax] != 1:
                extent[d] = sh[ax]
    return maps, reduced, extent


def _get_reduce_kind(op, n_in=1, out=0):
    """The combining kind of a reduce whose body makes each element from its `n_in` inputs (and
    values made outside the body) with vectorisable ops, then combines it into output `out`'s init
    with one op of REDUCE_KIND that only the yield reads; None for any other body. The body's
    other outputs are elements it writes, made the same way."""
    from mlir.dialects import vector
    body = op.regions[0].blocks[0]
    args = list(body.arguments)
    ops_ = [i.operation for i in body.operations]
    yld = ops_[-1]
    ret = yld.operands[out]
    comb = next((st for st in ops_[:-1] if ret in list(st.results)), None)
    if comb is None or REDUCE_KIND.get(comb.name) is None \
            or [op_of(u.owner) for u in comb.results[0].uses] != [yld]:
        return None
    steps = [st for st in ops_[:-1] if st != comb]
    inside = {r for st in ops_[:-1] for r in st.results}
    seen = set(args[:n_in])
    for st in steps:
        if st.name not in ("arith.constant", "linalg.index") and (
                st.name.split(".")[0] not in VECTORISABLE
                or any((o in inside or o in args) and o not in seen for o in st.operands)):
            return None
        seen.update(st.results)
    ops = list(comb.operands)
    acc = args[n_in + out]
    if len(ops) != 2 or acc not in ops:
        return None
    other = ops[1] if ops[0] == acc else ops[0]
    if other not in seen:
        return None
    return vector.CombiningKind[REDUCE_KIND[comb.name]]


def _row_outputs(maps, n_in, dim):
    """The outputs a reduce over `dim` combines into: those whose map does not walk it. The others
    are tiles it writes element for element as it walks."""
    return [k for k in range(n_in, len(maps)) if dim not in maps[k]]


def _idempotent(kind):
    """Whether folding a value in twice leaves the result as once: max, min, and, or."""
    k = str(kind).lower()
    return k in ("and", "or") or k.startswith(("max", "min"))


def _identity_vector(kind, width, elem):
    """A vector<width> of `kind`'s identity: 0 for add and xor, 1 for mul."""
    from mlir.dialects import arith
    from mlir.ir import DenseElementsAttr, FloatAttr, FloatType, IntegerAttr, VectorType
    one = "mul" in str(kind).lower()
    ty = VectorType.get([width], elem)
    if isinstance(elem, FloatType) or str(elem) in ("f32", "f16", "bf16", "f64"):
        a = FloatAttr.get(elem, 1.0 if one else 0.0)
    else:
        a = IntegerAttr.get(elem, 1 if one else 0)
    return arith.ConstantOp(ty, DenseElementsAttr.get_splat(ty, a)).result


def _combine(kind, a, b):
    """`a` and `b` combined by `kind` (add or mul), elementwise."""
    from mlir.dialects import arith
    from mlir.ir import VectorType
    is_float = str(VectorType(a.type).element_type).startswith(("f", "bf"))
    if "mul" in str(kind).lower():
        return (arith.MulFOp if is_float else arith.MulIOp)(a, b).result
    if str(kind).lower() == "xor":
        return arith.XOrIOp(a, b).result
    return (arith.AddFOp if is_float else arith.AddIOp)(a, b).result


def _folds_nothing(reduced, extent):
    """A reduce whose reduced dims all have extent 1 combines each element with its init once:
    no loop folds anything, so none is built (and no vector<1> carries the one element)."""
    return all(extent[d] == 1 for d in reduced)


def _generic_reduced(op):
    """The reduction dims of a linalg.generic."""
    from mlir.ir import ArrayAttr
    if op.name != "linalg.generic":
        return []
    return [d for d, it in enumerate(ArrayAttr(op.attributes["iterator_types"]))
            if attr_is(it, "#linalg.iterator_type<reduction>")]


def _n_ins(op):
    return op.attributes["operandSegmentSizes"][0]


def _is_pure_combiner(op, n_in):
    args = list(op.regions[0].blocks[0].arguments)
    if [a.type for a in args[:n_in]] != [a.type for a in args[n_in:]]:
        return False
    tainted = set(args[n_in:])
    for inner in op.regions[0].blocks[0].operations:
        o = inner.operation
        if o.name == "linalg.yield":
            return True
        if len(o.operands) and not any(v in tainted for v in o.operands):
            return False
        tainted.update(o.results)
    return False


@contextlib.contextmanager
def _build_loops(op, plan):
    from mlir.ir import BoolAttr, InsertionPoint, UnitAttr
    from mlir.dialects import affine
    with InsertionPoint(op):
        lane = affine.AffineForOp(0, plan.lanes_used, 1)
        lane.operation.attributes[VLANE_LOOP] = BoolAttr.get(True)
        with InsertionPoint(lane.body):
            affine.AffineYieldOp([])
    with InsertionPoint.at_block_terminator(lane.body):
        steps = plan.steps or [1] * len(plan.bounds)
        loops, ips = [], []
        for i, (b, st) in enumerate(zip(plan.bounds, steps)):
            f = affine.AffineForOp(0, b, st)
            if plan.steps is not None and i == len(plan.bounds) - 1:
                f.operation.attributes[VECTOR_LOOP] = UnitAttr.get()
            loops.append(f)
            with InsertionPoint(f.body):
                term = affine.AffineYieldOp([])
            ip = InsertionPoint(term)
            ip.__enter__()
            ips.append(ip)
        try:
            yield _Nest([f.induction_variable for f in loops], plan)
        finally:
            for ip in reversed(ips):
                ip.__exit__(None, None, None)


def _build_body(op, plan, nest):
    from mlir.ir import InsertionPoint, MemRefType, UnitAttr, VectorType
    from mlir.dialects import affine, vector
    if plan.body is None:
        nest.store(vector.BroadcastOp(plan.vec_ty, op.operands[0]).result, len(plan.memrefs) - 1)
        return
    arg_mem = {a: k for k, a in enumerate(plan.body.arguments)}

    def vector_body(env):
        return build_vector_body(
            plan.body.operations, plan.vec_len, env,
            provide=lambda v: nest.load(arg_mem[v]) if v in arg_mem else None,
            index_value=nest.index_value, loc=op.location if plan.reduce else None)

    red = plan.reduce
    if red is None:
        results = vector_body({})
        first_out = len(plan.memrefs) - len(results)
        for j, val in enumerate(results):
            nest.store(val, first_out + j)
        return

    outs = list(red.rows)
    tiles = [k for k in range(plan.n_in, len(plan.memrefs)) if k not in outs]
    if red.along and red.kind is None:
        _build_combined_reduce(op, plan, nest, vector_body, outs)
        return
    _check_contract(not tiles or not any(list(plan.body.arguments[k].uses) for k in tiles),
                    "a tile written beside a reduce is never read by its body", op)
    # A vector of partials folded at the end starts from the combiner's identity when the init is
    # not a seeded fill: broadcast across it, the init would be folded in once per element (an add
    # counts it vec_len times). The init then joins once, as the fold's accumulator.
    late = {k: nest.load_splat(k) for k in outs
            if red.along and k not in red.seeds and not _idempotent(red.kind)}
    seeds = [vector.BroadcastOp(VectorType.get(
                 [plan.vec_len], MemRefType(plan.memrefs[k].type).element_type), red.seeds[k]).result
             if k in red.seeds else
             _identity_vector(red.kind, plan.vec_len, MemRefType(plan.memrefs[k].type).element_type)
             if k in late else nest.load(k) for k in outs]
    loop = affine.AffineForOp(0, red.extent, red.extent if red.along else 1, iter_args=seeds)
    if red.along:
        loop.operation.attributes[VECTOR_LOOP] = UnitAttr.get()
        loop.operation.attributes[REDUCE_LOOP] = UnitAttr.get()
    with InsertionPoint(loop.body):
        nest.ivs.append(loop.induction_variable)
        made = vector_body(dict(zip([plan.body.arguments[k] for k in outs],
                                    loop.inner_iter_args)))
        for k in tiles:
            nest.store(made[k - plan.n_in], k)
        affine.AffineYieldOp([made[k - plan.n_in] for k in outs])
        nest.ivs.pop()
    for i, k in enumerate(outs):
        got = loop.results[i]
        if red.along:
            elem = VectorType(got.type).element_type
            got = vector.BroadcastOp(VectorType.get([1], elem),
                                     vector.ReductionOp(elem, red.kind, got).result).result
            if k in late:
                got = _combine(red.kind, got, late[k])
        nest.store(got, k)


def _build_combined_reduce(op, plan, nest, vector_body, outs):
    from mlir.ir import InsertionPoint, MemRefType, UnitAttr, VectorType
    from mlir.dialects import affine, arith, vector
    red = plan.reduce
    elems = [MemRefType(plan.memrefs[k].type).element_type for k in outs]
    accs = [vector.BroadcastOp(VectorType.get([1], e), red.seeds[k]).result
            if k in red.seeds else nest.load_splat(k) for k, e in zip(outs, elems)]
    seeds = [vector.BroadcastOp(VectorType.get([plan.vec_len], e), a).result
             for e, a in zip(elems, accs)]
    loop = affine.AffineForOp(0, red.extent, red.extent, iter_args=seeds)
    loop.operation.attributes[VECTOR_LOOP] = UnitAttr.get()
    loop.operation.attributes[REDUCE_LOOP] = UnitAttr.get()
    with InsertionPoint(loop.body):
        iv = loop.induction_variable
        nest.ivs.append(iv)
        xs = [nest.load(k) for k in range(plan.n_in)]
        got = vector_body(dict(zip(plan.body.arguments, xs + list(loop.inner_iter_args))))
        first = arith.CmpIOp(arith.CmpIPredicate.eq, iv, const_index(0)).result
        affine.AffineYieldOp([arith.SelectOp(first, x, g).result for x, g in zip(xs, got)])
        nest.ivs.pop()
    folded = _build_reduction(plan.body, list(loop.results), accs, op.location)
    for k, e, g in zip(outs, elems, folded.results):
        nest.store(vector.BroadcastOp(VectorType.get([1], e), g).result, k)


def _build_reduction(body, vectors, accs, loc):
    from mlir.ir import InsertionPoint, Operation, VectorType
    red = Operation.create("torchsim.reduction",
                           results=[VectorType(v.type).element_type for v in vectors],
                           operands=vectors + accs, regions=1, loc=loc)
    blk = red.regions[0].blocks.append(*[a.type for a in body.arguments])
    env = dict(zip(body.arguments, blk.arguments))
    with InsertionPoint(blk):
        for inner in body.operations:
            o = inner.operation
            operands = [env.get(v, v) for v in o.operands]
            if o.name == "linalg.yield":
                Operation.create("torchsim.yield", operands=operands, loc=loc)
                break
            new = Operation.create(o.name, results=[r.type for r in o.results],
                                   operands=operands, attributes=attrs_of(o), loc=loc)
            env.update(zip(o.results, new.results))
    return red


class _Nest:

    def __init__(self, ivs, plan):
        self.ivs = ivs
        self.plan = plan

    def indices(self, k):
        return [const_index(0) if p is None else self.ivs[p] for p in self.plan.index_of[k]]

    def amap_of(self, k):
        from mlir.ir import AffineMap, AffineMapAttr
        return AffineMapAttr.get(AffineMap.get_identity(len(self.plan.index_of[k])))

    def load(self, k):
        """Operand `k` at this trip as a vector of the nest's width; a splatted one is one element
        per lane, broadcast."""
        from mlir.dialects import affine, memref, vector
        from mlir.ir import MemRefType, VectorType
        mem = self.plan.memrefs[k]
        e = MemRefType(mem.type).element_type
        vt = VectorType.get([self.plan.vec_len], e)
        if not MemRefType(mem.type).shape:
            return vector.BroadcastOp(vt, memref.LoadOp(mem, []).result).result
        if k in self.plan.strided:
            return self._strided("torchsim.strided_load", [vt], [mem], k)
        if self.plan.bcast[k]:
            return vector.BroadcastOp(vt, self.load_splat(k)).result
        return affine.AffineVectorLoadOp(vt, mem, self.indices(k), self.amap_of(k)).result

    def load_splat(self, k):
        """Operand `k` at this trip, one element per lane: a vector<1> marked lane_splat."""
        from mlir.dialects import affine, memref, vector
        from mlir.ir import MemRefType, UnitAttr, VectorType
        mem = self.plan.memrefs[k]
        one_ty = VectorType.get([1], MemRefType(mem.type).element_type)
        if not MemRefType(mem.type).shape:
            return vector.BroadcastOp(one_ty, memref.LoadOp(mem, []).result).result
        one = affine.AffineVectorLoadOp(one_ty, mem, self.indices(k), self.amap_of(k))
        one.operation.attributes[LANE_SPLAT] = UnitAttr.get()
        return one.result

    def store(self, val, k):
        """`val` into operand `k` at this trip; a rank-0 one written by every lane into its bank."""
        from mlir.dialects import affine
        from mlir.ir import MemRefType
        dst = self.plan.memrefs[k]
        if not MemRefType(dst.type).shape:
            _build_uniform_store(val, dst)
            return
        if k in self.plan.strided:
            self._strided("torchsim.strided_store", [], [val, dst], k)
            return
        affine.AffineVectorStoreOp(val, dst, self.indices(k), self.amap_of(k))

    def _strided(self, name, results, operands, k):
        from mlir.ir import IntegerAttr, IntegerType, Operation
        i64 = IntegerType.get_signless(64)
        op = Operation.create(name, results=results, operands=operands + self.indices(k),
                              attributes={"stride": IntegerAttr.get(i64, self.plan.strided[k]),
                                          "count": IntegerAttr.get(i64, self.plan.vec_len)})
        return op.results[0] if results else None

    def index_value(self, dim):
        """A `linalg.index` in the nest -- a lane number, a vector position, or a sum of the two."""
        from mlir.dialects import arith, vector
        from mlir.ir import IndexType, Operation, VectorType
        p = self.plan
        idx_ty = VectorType.get([p.vec_len], IndexType.get())
        if dim == p.vec_dim and dim != p.lane_dim:
            step = vector.StepOp(idx_ty).result
            base = vector.BroadcastOp(idx_ty, self.ivs[-1]).result
            return arith.AddIOp(base, step).result
        if dim != p.lane_dim:
            iv = self.ivs[p.pos[dim]] if dim in p.pos else const_index(0)
            return vector.BroadcastOp(idx_ty, iv).result
        lane = Operation.create("torchsim.vlane_idx", results=[idx_ty], operands=[],
                                attributes={}).results[0]
        if p.depth > 1:
            slot = vector.BroadcastOp(idx_ty, self.ivs[p.pos[p.lane_dim]]).result
            if p.lane_dim == p.vec_dim:
                slot = arith.AddIOp(slot, vector.StepOp(idx_ty).result).result
            span = vector.BroadcastOp(idx_ty, const_index(p.lanes_used)).result
            lane = arith.AddIOp(arith.MulIOp(slot, span).result, lane).result
        return lane


def _build_uniform_store(val, dst):
    from mlir.dialects import affine, arith, memref
    from mlir.ir import AffineMap, AffineMapAttr, ArrayAttr, IndexType, MemRefType, VectorType
    flat = memref.ExpandShapeOp(
        MemRefType.get([1], VectorType(val.type).element_type,
                       memory_space=MemRefType(dst.type).memory_space), dst,
        ArrayAttr.get([]), [], [1]).result
    zero = arith.ConstantOp(IndexType.get(), 0).result
    affine.AffineVectorStoreOp(val, flat, [zero], AffineMapAttr.get(AffineMap.get_identity(1)))


def _check_nothing_unbanked(module):
    from mlir.ir import MemRefType

    def in_spad(v):
        return is_memref(v) and MemRefType(v.type).memory_space is not None

    for func in funcs(module):
        for op in walk(func.operation):
            if op.name in ("memref.load", "memref.store"):
                buf = op.operands[0 if op.name == "memref.load" else 1]
                if in_spad(buf) and math.prod(MemRefType(buf.type).shape) > 1:
                    raise NotVectorisable(
                        f"'{op.name}' addresses a scratchpad buffer as if it were flat, but "
                        f"the DMA banked it one slice per lane: {op}")

