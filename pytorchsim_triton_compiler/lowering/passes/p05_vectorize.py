"""torchsim-vectorize: every elementwise linalg op and reduce -> the banked affine nest."""

import contextlib
import math
from collections import namedtuple

from .lib_bank_layout import bank_strides, is_one_bank, lane_split, lanes_of, why_no_lane_axis
from .lib_ir import (VIEW_OPS, attr_is, attrs_of, build_vector_body, const_index, const_int, funcs,
                     int_array, is_memref, name_of, op_of, operand_segments, producer, walk)

VECTORISABLE = ("arith", "math")

VECTOR_LOOP = "torchsim.vector_loop"

VLANE_LOOP = "torchsim.vlane_loop"

REDUCE_LOOP = "torchsim.reduce_loop"
LIVE_EXTENT = "torchsim.live_extent"
LIVE_TERM = "torchsim.live_term"
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
    "index_of strided reduce live_terms")

_Reduce = namedtuple("_Reduce", "dim extent along kind seeds live")

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
        seeded = _find_seeding_fills(module)
        for op in [o for o in walk(module.operation) if o.name in _TARGETS]:
            if _ABSORBED in op.attributes:
                op.erase()
                continue
            at = next((i for i, (r, _s) in enumerate(seeded) if r == op), None)
            plan = _get_loop_plan(op, lanes, seeded.pop(at)[1] if at is not None else {})
            if plan.live_terms is not None:
                _mark_live_terms(op, plan.live_terms, plan.reduce.along)
            with _build_loops(op, plan) as nest:
                _build_body(op, plan, nest)
            op.erase()
    PassManager.parse("builtin.module(func.func(canonicalize,cse))").run(module.operation)
    _check_nothing_unbanked(module)
    return module


def _find_seeding_fills(module):
    from mlir.ir import UnitAttr
    out = []
    for red in [o for o in walk(module.operation) if _reduces(o)]:
        seeds = {}
        for k in range(_n_ins(red), len(red.operands)):
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
    n_in = _n_ins(op) if _reduces(op) else 0
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

    reduce = live_terms = None
    if reduced:
        _check_contract(len(shape) > 0, "a reduce's result has an axis to write", op)
        real = [d for d in reduced if iter_shape[d] != 1]
        _check_contract(len(real) <= 1, "a reduce reduces exactly one dim", op)
        dim = real[0] if real else reduced[0]
        if real:
            pos[dim] = len(bounds)
        kind = _get_reduce_kind(op, n_in)
        along = dim == maps[0][-1] and (kind is not None or _is_pure_combiner(op, n_in))
        if dim == maps[0][-1]:
            steps, vec_dim = None, dim if along else None
            vec_len = iter_shape[dim] if along else 1
        live = _get_live_extent(op, n_in, dim, iter_shape[dim]) if kind is not None else None
        live_terms = live[1] if live else None
        extent = live[0] if live and not along else iter_shape[dim]
        reduce = _Reduce(dim, extent, along, kind, seeds, live[0] if live and along else None)

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
                 VectorType.get([vec_len], elem), pos, vec_dim, bcast, index_of, strided, reduce,
                 live_terms)


def _find_maps(op, memrefs):
    from mlir.ir import AffineConstantExpr, AffineDimExpr, ArrayAttr, AffineMapAttr, MemRefType
    shapes = [list(MemRefType(v.type).shape) for v in memrefs]
    reduced = []
    if op.name == "linalg.generic":
        reduced = [d for d, it in enumerate(ArrayAttr(op.attributes["iterator_types"]))
                   if not attr_is(it, "#linalg.iterator_type<parallel>")]
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
    out = [d for d in maps[-1] if d is not None]
    nd = len(out) + len(reduced)
    _check_contract(sorted(out + reduced) == list(range(nd)), "the outs map every kept dim once", op)
    extent = [1] * nd
    for m, sh in zip(maps, shapes):
        for ax, d in enumerate(m):
            if d is not None and sh[ax] != 1:
                extent[d] = sh[ax]
    return maps, reduced, extent


def _get_reduce_kind(op, n_in):
    """The combining kind when the body is a prologue that never reads the one accumulator, then
    one combiner of the prologue's last value and that accumulator; else None."""
    from mlir.dialects import vector
    body = op.regions[0].blocks[0]
    args = list(body.arguments)
    if len(args) != n_in + 1:
        return None
    steps = [i.operation for i in body.operations if i.operation.name != "linalg.yield"]
    comb = steps[-1].name if steps else None
    yielded = list(op_of(list(body.operations)[-1]).operands)
    if REDUCE_KIND.get(comb) is None or yielded != [steps[-1].results[0]]:
        return None
    acc = args[n_in]
    for st in steps[:-1]:
        known = st.name == "linalg.index" or st.name.split(".")[0] in VECTORISABLE
        if not known or acc in list(st.operands):
            return None
    cur_v = steps[-2].results[0] if len(steps) > 1 else args[0]
    if set(steps[-1].operands) != {cur_v, acc}:
        return None
    return vector.CombiningKind[REDUCE_KIND[comb]]


_IDENTITY = {"ADD": 0, "OR": 0, "XOR": 0, "MAXUI": 0, "MUL": 1, "AND": -1, "MINUI": -1,
             "MAXNUMF": -math.inf, "MAXIMUMF": -math.inf,
             "MINNUMF": math.inf, "MINIMUMF": math.inf}

_BELOW = {2: 0, 6: 0, 3: 1, 7: 1}
_ABOVE = {4: 0, 8: 0, 5: 1, 9: 1}


def _get_live_extent(op, n_in, dim, extent):
    """(B, the compares saying it) when the combiner's other operand is select(p, x, identity) and
    p is a conjunction with a term index(dim) < B, B a known constant below `extent`: every trip
    from B on leaves the accumulator as it was, and on every trip below B those terms are true."""
    from mlir.ir import FloatAttr, IntegerAttr
    body = op.regions[0].blocks[0]
    comb = [i.operation for i in body.operations if i.operation.name != "linalg.yield"][-1]
    cur = [v for v in comb.operands if v != body.arguments[n_in]]
    sel = producer(cur[0]) if len(cur) == 1 else None
    if name_of(sel) != "arith.select":
        return None
    fill = producer(sel.operands[2])
    if name_of(fill) != "arith.constant":
        return None
    attr = fill.attributes["value"]
    got = (FloatAttr(attr).value if isinstance(attr, FloatAttr)
           else IntegerAttr(attr).value if isinstance(attr, IntegerAttr) else None)
    if got is None or got != _IDENTITY.get(REDUCE_KIND[comb.name]):
        return None
    bounds = [(b, c) for c in _get_conjuncts(sel.operands[0])
              for b in [_get_index_bound(c, dim)] if b is not None]
    if not bounds:
        return None
    live = min(b for b, _c in bounds)
    terms = [c for b, c in bounds if b == live]
    return (live, terms) if 0 < live < extent else None


def _get_conjuncts(v):
    """The terms an `arith.andi` tree over `v` is the conjunction of."""
    o = producer(v)
    if name_of(o) == "arith.andi":
        return [t for x in o.operands for t in _get_conjuncts(x)]
    return [o]


def _get_index_bound(cmp, dim):
    """B when `cmp` holds exactly for index(dim) < B, B a constant; else None."""
    from mlir.ir import IntegerAttr
    if name_of(cmp) != "arith.cmpi":
        return None
    pred = IntegerAttr(cmp.attributes["predicate"]).value
    for side, table in ((0, _BELOW), (1, _ABOVE)):
        b = const_int(cmp.operands[1 - side])
        if pred in table and b is not None and _is_index_of(cmp.operands[side], dim):
            return b + table[pred]
    return None


def _mark_live_terms(op, terms, along):
    """A reduce run trip by trip stops at B, so its `terms` are `true`; one run along the vector
    keeps them, marked, for legalize to drop where its width ends the loop at B."""
    from mlir.dialects import arith
    from mlir.ir import InsertionPoint, IntegerAttr, IntegerType, UnitAttr
    if along:
        for c in terms:
            c.attributes[LIVE_TERM] = UnitAttr.get()
        return
    with InsertionPoint(op):
        true = arith.ConstantOp(IntegerType.get_signless(1),
                                IntegerAttr.get(IntegerType.get_signless(1), 1)).result
    for c in terms:
        c.results[0].replace_all_uses_with(true)


def _is_index_of(v, dim):
    """Whether `v` is `linalg.index dim`, through casts that keep every value below the extent."""
    from mlir.ir import IntegerAttr
    o = producer(v)
    while name_of(o) in ("arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui"):
        o = producer(o.operands[0])
    return name_of(o) == "linalg.index" and IntegerAttr(o.attributes["dim"]).value == dim


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
    from mlir.ir import IndexType, InsertionPoint, IntegerAttr, MemRefType, UnitAttr, VectorType
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
    seeds = [vector.BroadcastOp(VectorType.get(
                 [plan.vec_len], MemRefType(plan.memrefs[k].type).element_type), red.seeds[k]).result
             if k in red.seeds else nest.load(k) for k in outs]
    loop = affine.AffineForOp(0, red.extent, red.extent if red.along else 1, iter_args=seeds)
    if red.along:
        loop.operation.attributes[VECTOR_LOOP] = UnitAttr.get()
        loop.operation.attributes[REDUCE_LOOP] = UnitAttr.get()
    if red.live is not None:
        loop.operation.attributes[LIVE_EXTENT] = IntegerAttr.get(IndexType.get(), red.live)
    with InsertionPoint(loop.body):
        nest.ivs.append(loop.induction_variable)
        affine.AffineYieldOp(vector_body(dict(zip(plan.body.arguments[plan.n_in:],
                                              loop.inner_iter_args))))
        nest.ivs.pop()
    for i, k in enumerate(outs):
        got = loop.results[i]
        if red.along:
            elem = VectorType(got.type).element_type
            got = vector.BroadcastOp(VectorType.get([1], elem),
                                     vector.ReductionOp(elem, red.kind, got).result).result
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


def _reduces(op):
    """Whether `op` walks a reduction iterator -- a named reduce, or a generic that states one."""
    from mlir.ir import ArrayAttr
    if op.name == "linalg.reduce":
        return True
    return op.name == "linalg.generic" and any(
        not attr_is(it, "#linalg.iterator_type<parallel>")
        for it in ArrayAttr(op.attributes["iterator_types"]))


def _n_ins(op):
    """How many of `op`'s operands are `ins`."""
    return len(op.operands) // 2 if op.name == "linalg.reduce" else operand_segments(op)[0]
