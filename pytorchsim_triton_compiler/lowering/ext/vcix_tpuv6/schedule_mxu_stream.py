"""tritonpytorchsim-schedule-mxu-stream: where each op of a block that drives vcix_tpuv6's matrix unit is
issued, the unit's vcix ops kept in tritonpytorchsim-lower-matmul-tpuv6's order so its lag bound holds.
They take and give registers only: a bank access moves past them, and past another it cannot alias."""
import heapq
import sys

from .... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from ...passes import lib_matmul as mm  # noqa: E402
from ...passes.lib_ir import (build_trip_loop, const_index, const_int, loop_bounds,  # noqa: E402
                              producer, trip_count, walk)
from ...passes.lib_pack import PACK_OP  # noqa: E402
from .lower_matmul import VMATPREP  # noqa: E402

NAME = "tritonpytorchsim-schedule-mxu-stream"

REQUIRES = ("vcix_tpuv6",)

SLOT = "schedule-matmul"

MARKERS = ("vcix.iv", "vcix.v.i")

FORBIDS = ()

UNROLL = 1


def run(module, **_):
    """Every steady loop unrolled by UNROLL, every block that holds a vcix op with its repeats
    merged and its accesses rebased, every loop that drives the unit with what it does not vary
    moved out, and every such block's ops placed."""
    for loop in [o for o in walk(module.operation) if _is_steady(o)]:
        _unroll(loop, UNROLL)
    for block in _driving_blocks(module):
        _merge_repeats(block)
        _rebase(block)
    loops = [o for o in walk(module.operation) if o.name == "affine.for" and _drives(o)]
    for loop in reversed(loops):
        _hoist_invariants(loop)
    for block in _driving_blocks(module):
        _merge_repeats(block)
        _schedule(block)
    return module


def _is_vcix(op):
    return op.name.split(".", 1)[0] == "vcix"


def _drives(op):
    return any(_is_vcix(o) for o in walk(op))


def _is_steady(op):
    """An innermost loop of no results whose body both pushes and pops: the stream past its lag."""
    if op.name != "affine.for" or len(op.results):
        return False
    body = [o.operation for o in op.regions[0].blocks[0].operations]
    units = [o for o in body if _is_vcix(o)]
    return (not any(o.regions for o in body) and any(len(o.results) for o in units)
            and any(not len(o.results) for o in units))


def _unroll(loop, factor):
    """`loop` as a loop of `factor` copies of its body a trip, then the trips left over: the vcix
    ops run in the order they did. A factor of 1 leaves it as it is."""
    from mlir.dialects import arith
    trips, (first, _, step) = trip_count(loop), loop_bounds(loop)
    if factor < 2 or trips is None or trips < factor:
        return
    body = loop.regions[0].blocks[0]
    iv, ops = body.arguments[0], [o.operation for o in body.operations][:-1]
    ip = ir.InsertionPoint(loop)
    outer, inner = build_trip_loop(ip, trips // factor)
    with inner:
        at = arith.MulIOp(outer, const_index(factor * step)).result
        if first:
            at = arith.AddIOp(at, const_index(first)).result
        for u in range(factor):
            _clone(ops, {iv: at if not u else arith.AddIOp(at, const_index(u * step)).result}, inner)
    for t in range(trips // factor * factor, trips):
        with ip:
            _clone(ops, {iv: const_index(first + t * step)}, ip)
    loop.erase()


def _clone(ops, mapping, ip):
    """Each of `ops` cloned at `ip`, reading `mapping`'s values for the ones it names."""
    for o in ops:
        new = o.clone(ip).operation
        for x in [new] + walk(new):
            for k, v in enumerate(list(x.operands)):
                if v in mapping:
                    x.operands[k] = mapping[v]
        mapping.update(zip(o.results, new.results))


def _driving_blocks(module):
    """Every block a vcix op sits in directly, each once, in program order."""
    seen, out = set(), []
    for o in walk(module.operation):
        if _is_vcix(o) and o.block not in seen:
            seen.add(o.block)
            out.append(o.block)
    return out


def _is_memref(t):
    return isinstance(t, ir.MemRefType) or isinstance(t, ir.UnrankedMemRefType)


def _is_value(op):
    """An op that takes no buffer and gives a value, outside the unit: it moves wherever its
    operands are defined."""
    return (not _is_vcix(op) and not op.regions and len(op.results) > 0
            and not any(_is_memref(v.type) for v in op.operands)
            and not any(_is_memref(r.type) for r in op.results))


def _defined_inside(op):
    inside = set()
    for o in walk(op):
        inside.update(o.results)
        for region in o.regions:
            for block in region.blocks:
                inside.update(block.arguments)
    for region in op.regions:
        for block in region.blocks:
            inside.update(block.arguments)
    return inside


def _hoist_invariants(loop):
    """Each value op of `loop`'s body whose operands come from outside it, moved before it; a loop
    that runs at least once, so nothing is computed that the program would not."""
    if not (trip_count(loop) or 0):
        return
    inside = _defined_inside(loop)
    body = loop.regions[0].blocks[0]
    for o in [x.operation for x in body.operations]:
        if _is_value(o) and not any(v in inside for v in o.operands):
            o.move_before(loop)
            inside.difference_update(o.results)


def _merge_repeats(block):
    """A value op of `block` that repeats an earlier one -- same op, operands, attributes and
    types -- replaced by it."""
    held = {}
    for o in [x.operation for x in block.operations]:
        if not _is_value(o):
            continue
        attrs = tuple((a.name, a.attr) for a in (o.attributes[k] for k in range(len(o.attributes))))
        key = (o.name, tuple(o.operands), tuple(r.type for r in o.results))
        for prev_attrs, prev in held.get(key, ()):
            if prev_attrs == attrs:
                for new, old in zip(o.results, prev.results):
                    new.replace_all_uses_with(old)
                o.erase()
                break
        else:
            held.setdefault(key, []).append((attrs, o))


def _linear(v, memo):
    """`v` as ({leaf: coefficient}, constant) over arith's integer add, sub and multiply by a
    constant; any other value is a leaf of its own."""
    if v in memo:
        return memo[v]
    c = const_int(v)
    got = ({}, c) if c is not None else None
    owner = v.owner if not isinstance(v, ir.BlockArgument) else None
    name = owner.name if got is None and isinstance(owner, (ir.Operation, ir.OpView)) else None
    if name in ("arith.addi", "arith.subi"):
        (a, ca), (b, cb) = (_linear(x, memo) for x in owner.operands)
        s = 1 if name == "arith.addi" else -1
        terms = dict(a)
        for k, n in b.items():
            terms[k] = terms.get(k, 0) + s * n
        got = ({k: n for k, n in terms.items() if n}, ca + s * cb)
    elif name == "arith.muli":
        (a, ca), (b, cb) = (_linear(x, memo) for x in owner.operands)
        if not a or not b:
            (t, ct), k = ((a, ca), cb) if not b else ((b, cb), ca)
            got = ({x: n * k for x, n in t.items() if n * k}, ct * k)
    if got is None:
        got = ({v: 1}, 0)
    memo[v] = got
    return got


def _access(op):
    """(buffer, indices, length, writes) of a one-dimensional vector transfer, length None where
    the access is not one run of the buffer's last dimension; None for any other op."""
    from mlir.dialects import vector
    view = op.opview
    if isinstance(view, vector.TransferReadOp):
        base, vec, writes, at = op.operands[0], op.results[0].type, False, 1
    elif isinstance(view, vector.TransferWriteOp):
        base, vec, writes, at = op.operands[1], op.operands[0].type, True, 2
    else:
        return None
    rank = len(ir.MemRefType(base.type).shape)
    shape = ir.VectorType(vec).shape
    perm = ir.AffineMapAttr(op.attributes["permutation_map"]).value
    run = len(shape) == 1 and perm == ir.AffineMap.get_minor_identity(rank, 1)
    return base, list(op.operands[at:at + rank]), shape[0] if run else None, writes


def _root(v):
    """The buffer `v` names, as lib_matmul reads it, through a packed view too."""
    while True:
        r = mm._root(v)
        p = None if isinstance(r, tuple) else producer(r)
        if p is None or p.name != PACK_OP:
            return r
        v = p.operands[0]


def _fresh(root):
    """Is `root` a buffer no other root can name: a global, or an allocation of this function?"""
    if isinstance(root, tuple):
        return True
    owner = root.owner if not isinstance(root, ir.BlockArgument) else None
    return isinstance(owner, (ir.Operation, ir.OpView)) and owner.name in ("memref.alloc",
                                                                          "memref.alloca")


def _apart(a, b, memo):
    """Do accesses `a` and `b` touch no element in common?"""
    ra, rb = _root(a[0]), _root(b[0])
    if ra != rb:
        return _fresh(ra) and _fresh(rb)
    if a[0] != b[0] or a[2] is None or b[2] is None:
        return False
    diffs = []
    for x, y in zip(a[1], b[1]):
        (tx, cx), (ty, cy) = _linear(x, memo), _linear(y, memo)
        if tx != ty:
            return False
        diffs.append(cx - cy)
    last = diffs.pop()
    return not any(diffs) and (last >= b[2] or -last >= a[2])


def _rebase(block):
    """Each access of `block` at a constant distance from an earlier one of the same buffer, row
    and terms reads its offset as that one's plus the distance: one base a set, kept in a register
    by llc and stepped by an immediate, where `x + c` each would be a constant of its own."""
    from mlir.dialects import arith
    sets, memo = {}, {}
    for o in [x.operation for x in block.operations]:
        acc = _access(o)
        if acc is None or acc[2] is None:
            continue
        at = (2 if acc[3] else 1) + len(acc[1]) - 1
        terms, c = _linear(o.operands[at], memo)
        if terms:
            key = (acc[0], tuple(acc[1][:-1]), frozenset(terms.items()))
            sets.setdefault(key, []).append((c, o, at))
    for (first_c, first, first_at), *rest in sets.values():
        base, made = first.operands[first_at], {}
        for c, o, at in rest:
            if c != first_c and c not in made:
                with ir.InsertionPoint(o):
                    made[c] = arith.AddIOp(base, const_index(c - first_c)).result
            o.operands[at] = base if c == first_c else made[c]
    _drop_unread(block)


def _drop_unread(block):
    """Every value op of `block` nothing reads, erased."""
    for o in reversed([x.operation for x in block.operations]):
        if _is_value(o) and not any(len(list(r.uses)) for r in o.results):
            o.erase()


def _uses_in(op, index):
    """The ops of the block (by position in `index`) whose values `op` reads, its regions' too."""
    out = set()
    for o in [op] + walk(op):
        for v in o.operands:
            owner = v.owner if not isinstance(v, ir.BlockArgument) else None
            if isinstance(owner, (ir.Operation, ir.OpView)) and owner.operation in index:
                out.add(index[owner.operation])
    return out


def _depends(ops):
    """For each op, the earlier ones it must stay after: what it reads, the unit's op before it,
    an access it could alias of which one writes, and every effect across an op of unknown effects."""
    index = {o: i for i, o in enumerate(ops)}
    kind = []
    for o in ops:
        acc = _access(o)
        kind.append("vcix" if _is_vcix(o) else acc if acc else "value" if _is_value(o)
                    else "barrier")
    reads = [_uses_in(o, index) - {i} for i, o in enumerate(ops)]
    preds = [set(r) for r in reads]
    memo, last_vcix, last_barrier, since = {}, None, None, []
    for i, k in enumerate(kind):
        if k == "value":
            continue
        if last_barrier is not None:
            preds[i].add(last_barrier)
        if k == "barrier":
            preds[i].update(since)
            if last_vcix is not None:
                preds[i].add(last_vcix)
            last_barrier, since = i, []
            continue
        if k == "vcix":
            if last_vcix is not None:
                preds[i].add(last_vcix)
            last_vcix = i
            continue
        for j in since:
            other = kind[j]
            if (other[3] or k[3]) and not _apart(other, k, memo):
                preds[i].add(j)
        since.append(i)
    return kind, preds, reads


def _scalar(op):
    return all(not isinstance(r.type, (ir.VectorType, ir.ShapedType)) for r in op.results)


def _schedule(block):
    """The block's ops by (slot, tier, level): a unit op at its place in the stream, what a pop run's
    results are combined with and the operand pushed after it read before the run, the results right
    after it, the rest before what waits on them; in a slot address arithmetic, then breadth-first."""
    ops = [o.operation for o in block.operations]
    term = ops.pop() if ops and ops[-1].has_trait(ir.IsTerminatorTrait) else None
    kind, preds, reads = _depends(ops)
    n = len(ops)
    succs, users = [[] for _ in range(n)], [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
        for p in reads[i]:
            users[p].append(i)
    unit = [i for i in range(n) if kind[i] == "vcix"]
    order_of = {i: k for k, i in enumerate(unit)}
    slot = [None] * n
    for i in unit:
        slot[i] = order_of[i]
    for s, e in _pop_runs(ops, unit):
        reached = set()
        for k in range(s, e + 1):
            _forward(unit[k], users, kind, reached)
        for i in reached:
            slot[i] = e + 1 if slot[i] is None else max(slot[i], e + 1)
        ahead = (_backward(reached, reads, kind) - reached) | _backward(
            _first_operand(ops, unit, e + 1), reads, kind)
        for i in ahead:
            if slot[i] is None or slot[i] > s:
                slot[i] = s
    before = 0
    default = []
    for i in range(n):
        default.append(before)
        before += kind[i] == "vcix"
    for i in reversed(range(n)):
        if kind[i] == "vcix":
            continue
        wait = [slot[u] for u in users[i]] + ([slot[i]] if slot[i] is not None else [])
        slot[i] = min(wait) if wait else default[i]
    tier = [0 if kind[i] == "value" and _scalar(ops[i]) else 1 for i in range(n)]
    level = [0] * n
    for i in range(n):
        same = [level[p] + 1 for p in preds[i]
                if slot[p] == slot[i] and tier[p] == tier[i] == 1 and kind[p] != "vcix"]
        level[i] = max(same) if same else 0
    key = [(slot[i], 1 if kind[i] == "vcix" else 0, tier[i], level[i], i) for i in range(n)]
    order = _topological(n, preds, succs, key)
    if [i for i in order if kind[i] == "vcix"] != unit:
        raise RuntimeError(f"{NAME} moved a vcix op past another")
    if term is not None:
        for i in order:
            ops[i].move_before(term)
    return order


def _pop_runs(ops, unit):
    """(first, last) positions in the stream of each run of unit ops that give a value -- pops."""
    runs, start = [], None
    for k, i in enumerate(unit + [None]):
        pop = i is not None and len(ops[i].results) > 0
        if pop and start is None:
            start = k
        elif not pop and start is not None:
            runs.append((start, k - 1))
            start = None
    return runs


def _first_operand(ops, unit, k):
    """The pushes from stream position `k` that carry one operand: a vmatprep and the push after
    it, or the one push; none where `k` is past the stream or a pop."""
    out = []
    while k < len(unit) and not len(ops[unit[k]].results):
        out.append(unit[k])
        if not _prepares(ops[unit[k]]):
            break
        k += 1
    return out


def _prepares(op):
    return op.name == "vcix.iv" and ir.IntegerAttr(op.attributes["opcode"]).value == VMATPREP


def _backward(starts, reads, kind):
    """Every op the ops `starts` read, transitively, short of the unit's ops and the barriers."""
    seen, stack = set(), list(starts)
    while stack:
        for p in reads[stack.pop()]:
            if p not in seen and kind[p] not in ("vcix", "barrier"):
                seen.add(p)
                stack.append(p)
    return seen


def _forward(i, users, kind, reached):
    """Every op that reads op `i`'s values, transitively, short of the unit's ops."""
    stack = [i]
    while stack:
        for u in users[stack.pop()]:
            if u not in reached and kind[u] != "vcix" and kind[u] != "barrier":
                reached.add(u)
                stack.append(u)


def _topological(n, preds, succs, key):
    """The ops in an order every pred precedes, the smallest key first among those ready."""
    left = [len(p) for p in preds]
    ready = [key[i] for i in range(n) if not left[i]]
    heapq.heapify(ready)
    out = []
    while ready:
        i = heapq.heappop(ready)[-1]
        out.append(i)
        for u in succs[i]:
            left[u] -= 1
            if not left[u]:
                heapq.heappush(ready, key[u])
    if len(out) != n:
        raise RuntimeError(f"{NAME} found a cycle among a block's ops")
    return out
