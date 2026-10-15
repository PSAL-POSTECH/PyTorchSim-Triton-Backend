"""torchsim-vectorize: every elementwise linalg op and reduce -> the banked affine nest."""

import contextlib
import math
from collections import namedtuple

from .lib_bank_layout import bank_strides, is_one_bank, lane_split, lanes_of, why_no_lane_axis
from .lib_ir import (VIEW_OPS, attr_is, attrs_of, build_vector_body, const_index, funcs, int_array,
                     is_memref, name_of, op_of, walk)

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

_Reduce = namedtuple("_Reduce", "dim extent along kind seeds")

_ABSORBED = "torchsim.absorbed_into_reduce"

# Ops a reader may move above when it is handed a reduce's value: none of them touches memory.
_INERT = ("arith.constant", "memref.get_global", "memref.reinterpret_cast")

class NotVectorisable(Exception):
    """The op is not the elementwise-on-one-iteration-space shape this widens."""


def _check_contract(holds, what, op):
    if not holds:
        raise NotVectorisable(f"an upstream pass broke its promise -- {what}: {op}")


def run(module, lanes, vlen=None, lmul=None, **_):
    """Every reduce and elementwise op in program order: its plan, its loops, its body."""
    from mlir.ir import Location
    from mlir.passmanager import PassManager

    with Location.unknown():
        seeded = _find_seeding_fills(module)

        def plan_of(op):
            at = next((i for i, (r, _s) in enumerate(seeded) if r == op), None)
            return _get_loop_plan(op, lanes, seeded.pop(at)[1] if at is not None else {})

        ops = [o for o in walk(module.operation) if o.name in _TARGETS]
        done = set()
        for i, op in enumerate(ops):
            if i in done:
                continue
            if _ABSORBED in op.attributes:
                op.erase()
                done.add(i)
                continue
            plan = plan_of(op)
            j = next((j for j in range(i + 1, len(ops)) if _ABSORBED not in ops[j].attributes), None)
            nxt = ops[j] if j is not None else None
            hand = _find_hand_on(op, plan, nxt, lanes, vlen, lmul) if nxt is not None else None
            if hand is None:
                with _build_loops(op, plan) as nest:
                    _build_body(op, plan, nest)
                op.erase()
                continue
            nplan, k_in, keep_store = hand
            with _build_loops(op, plan) as nest:
                got = _build_body(op, plan, nest, keep=True, store=keep_store)
                _build_reader_inside(nxt, nplan, nest, {k_in: got})
            op.erase()
            nxt.erase()
            done.add(j)
    PassManager.parse("builtin.module(func.func(canonicalize,cse))").run(module.operation)
    _check_nothing_unbanked(module)
    return module


# A reduce hands its value on only when its row is at most this many vector strips long. Joining
# the reader's loop to the reduce's puts heavy vector code right after each loop's back-edge, and
# gem5 pays for every mispredicted back-edge by fetching and dropping that code: on longer rows
# that cost outweighs the store, reload and gather the hand-on saves (softmax, C >= 256: -3% to
# +8%; C = 128: -3% to -6%).
_HAND_ON_MAX_STRIPS = 2


def _find_hand_on(red, plan, reader, lanes, vlen=None, lmul=None):
    """(the reader's plan, the reader's operand that is `red`'s result, whether that result must
    still be written) when `reader`, the next op, reads a row reduce's one result one element a row
    and runs over the same lanes and rows: then the reader's vector loop runs inside the reduce's
    nest and takes the value in a register, with no store, reload and gather between them."""
    if (plan.reduce is None or not plan.reduce.along or plan.reduce.kind is None
            or len(plan.memrefs) - plan.n_in != 1 or reader.name != "linalg.generic"
            or _generic_reduced(reader)):
        return None
    out = plan.memrefs[-1]
    if red.operation.block != reader.operation.block:
        return None
    if len(plan.memrefs) - 1 not in plan.reduce.seeds and not _idempotent(plan.reduce.kind):
        return None  # its init joins after the fold (_build_body's late), a value the hand-on cannot carry
    if vlen and lmul:
        from mlir.ir import MemRefType
        from .lib_ir import elem_bits
        bits = max(elem_bits(MemRefType(m.type).element_type) for m in plan.memrefs)
        if plan.reduce.extent > _HAND_ON_MAX_STRIPS * max(1, vlen * lmul // bits):
            return None
    seen = False
    for x in red.operation.block.operations:
        x = x.operation
        if x == red.operation:
            seen = True
            continue
        if x == reader.operation:
            break
        if seen and not (x.name in _INERT or _ABSORBED in x.attributes):
            return None  # the reader would move above it: a DMA in, another op's write
    try:
        rplan = _get_loop_plan(reader, lanes, {})
    except NotVectorisable:
        return None
    if (rplan.reduce is not None or rplan.steps is None or rplan.bounds[:-1] != plan.bounds
            or (rplan.depth, rplan.lanes_used, rplan.lane_dim is None)
            != (plan.depth, plan.lanes_used, plan.lane_dim is None)):
        return None
    ks = [k for k, v in enumerate(rplan.memrefs[:len(rplan.memrefs) - len(reader.results or [])])
          if v == out]
    ins = [k for k in ks if k < len(rplan.memrefs) - 1]
    if len(ins) != 1 or not rplan.bcast[ins[0]] or out in rplan.memrefs[ins[0] + 1:]:
        return None
    users = {op_of(u.owner) for u in out.uses}
    if any(name_of(u) in VIEW_OPS for u in users):
        return None
    keep_store = any(u not in (red, reader) and _ABSORBED not in u.attributes for u in users)
    return rplan, ins[0], keep_store


def _build_reader_inside(reader, rplan, nest, given):
    """`reader`'s vector loop inside `nest` (the rows it shares with the reduce just built), its
    operands in `given` taken from registers."""
    from mlir.ir import InsertionPoint, UnitAttr
    from mlir.dialects import affine
    lp = affine.AffineForOp(0, rplan.bounds[-1], rplan.steps[-1])
    lp.operation.attributes[VECTOR_LOOP] = UnitAttr.get()
    with InsertionPoint(lp.body):
        term = affine.AffineYieldOp([])
    with InsertionPoint(term):
        _build_body(reader, rplan, _Nest(nest.ivs + [lp.induction_variable], rplan, given))


def _find_seeding_fills(module):
    from mlir.ir import UnitAttr
    out = []
    for red in [o for o in walk(module.operation)
                if o.name == "linalg.reduce" or _generic_reduced(o)]:
        n = len(red.operands) // 2 if red.name == "linalg.reduce" else _n_ins(red)
        reduced = list(int_array(red.attributes["dimensions"])) if red.name == "linalg.reduce" \
            else _generic_reduced(red)
        if _folds_nothing(reduced, _find_maps(red, [v for v in red.operands if is_memref(v)])[2]):
            continue            # walked elementwise: its init is read from the buffer the fill wrote
        seeds = {}
        for k in range(n, len(red.operands)):
            dst = red.operands[k]
            users = [op_of(u.owner) for u in dst.uses]
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

    one_lane = is_one_bank(shape)
    for v, m in zip(memrefs, maps):
        n = lanes_of(v)
        _check_contract(n is not None, why_no_lane_axis(v) if n is None else "", op)
        _check_contract((n == 1) == one_lane and (one_lane or m[:1] == maps[-1][:1]),
                        "every operand of one op rides the same lane dim", op)
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
        kind = _get_reduce_kind(op, n_in) if n_in == 1 or op.name == "linalg.generic" else None
        along = dim == maps[0][-1] and (kind is not None or _is_pure_combiner(op, n_in))
        if dim == maps[0][-1]:
            steps, vec_dim = None, dim if along else None
            vec_len = iter_shape[dim] if along else 1
        reduce = _Reduce(dim, iter_shape[dim], along, kind, seeds)

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


def _get_reduce_kind(op, n_in=1):
    """The combining kind of a reduce whose body makes each element from its `n_in` inputs (and
    values made outside the body) with vectorisable ops, then combines it into the init with one
    op of REDUCE_KIND; None for any other body."""
    from mlir.dialects import vector
    body = op.regions[0].blocks[0]
    args = list(body.arguments)
    steps = [i.operation for i in body.operations if i.operation.name != "linalg.yield"]
    comb = steps[-1].name if steps else None
    if REDUCE_KIND.get(comb) is None:
        return None
    inside = {r for st in steps for r in st.results}
    seen = set(args[:n_in])
    for st in steps[:-1]:
        if st.name not in ("arith.constant", "linalg.index") and (
                st.name.split(".")[0] not in VECTORISABLE
                or any((o in inside or o in args) and o not in seen for o in st.operands)):
            return None
        seen.update(st.results)
    ops = list(steps[-1].operands)
    acc = args[n_in]
    if len(ops) != 2 or acc not in ops:
        return None
    other = ops[1] if ops[0] == acc else ops[0]
    if other not in seen:
        return None
    return vector.CombiningKind[REDUCE_KIND[comb]]


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


def _build_body(op, plan, nest, keep=False, store=True):
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

    outs = list(range(plan.n_in, len(plan.memrefs)))
    if red.along and red.kind is None:
        _build_combined_reduce(op, plan, nest, vector_body, outs)
        return
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
        affine.AffineYieldOp(vector_body(dict(zip(plan.body.arguments[plan.n_in:],
                                              loop.inner_iter_args))))
        nest.ivs.pop()
    kept = None
    for i, k in enumerate(outs):
        got = loop.results[i]
        if red.along:
            elem = VectorType(got.type).element_type
            got = vector.BroadcastOp(VectorType.get([1], elem),
                                     vector.ReductionOp(elem, red.kind, got).result).result
            if k in late:
                got = _combine(red.kind, got, late[k])
        if store:
            nest.store(got, k)
        kept = got
    return kept if keep else None


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

    def __init__(self, ivs, plan, given=None):
        self.ivs = ivs
        self.plan = plan
        self.given = given or {}

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
        if k in self.given:
            return vector.BroadcastOp(vt, self.given[k]).result
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

