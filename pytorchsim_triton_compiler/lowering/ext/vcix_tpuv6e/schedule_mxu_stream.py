"""tritonpytorchsim-schedule-mxu-stream: where each op of a block that drives vcix_tpuv6e's matrix unit
is issued, its vcix ops in tritonpytorchsim-lower-matmul-tpuv6e's order. A steady loop is pipelined
across inputs: the next input's reads, the writes of two inputs back, the pops behind the last adds."""
import heapq
import sys

from .... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from ...passes.lib_ir import (build_trip_loop, const_index, const_int, elem_bits,  # noqa: E402
                              loop_bounds, producer, trip_count, walk)
from ...passes.lib_pack import PACK_OP  # noqa: E402
from ...passes.lower import SCHEDULE_MATMUL  # noqa: E402
from . import lib_mxu as mxu  # noqa: E402
from .lower_matmul import VMATPREP  # noqa: E402

NAME = "tritonpytorchsim-schedule-mxu-stream"

REQUIRES = ("vcix_tpuv6e",)

SLOT = SCHEDULE_MATMUL.name

MARKERS = ("vcix.iv", "vcix.v.i")

FORBIDS = SCHEDULE_MATMUL.lowers

UNROLL = 1

LINE_BYTES, BANKS = 64, 2

READ, VCIX, ADD, WRITE = -1, 0, 1, 2

MAX_LEAD = 3


def run(module, **_):
    """Every steady loop unrolled by UNROLL and pipelined, every block that holds a vcix op with its
    repeats merged and its accesses rebased, loops that drive the unit with what they do not vary
    moved out, and every other such block's ops placed."""
    for loop in [o for o in walk(module.operation) if _is_steady(o)]:
        _unroll(loop, UNROLL)
    for block in _driving_blocks(module):
        _merge_repeats(block)
        _rebase(block)
    loops = [o for o in walk(module.operation) if o.name == "affine.for" and _drives(o)]
    for loop in reversed(loops):
        _hoist_invariants(loop)
    kernels = []
    for loop in [o for o in walk(module.operation) if _is_steady(o)]:
        kernel = _pipeline(loop)
        if kernel is not None:
            kernels.append(kernel)
    for block in _driving_blocks(module):
        _merge_repeats(block)
        if any(block == k for k in kernels):
            _rebase(block)
        else:
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
    """The buffer `v` names, as lib_mxu reads it, through a packed view too."""
    while True:
        r = mxu.buffer_root(v)
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


def _offsets(a, b, memo, shift=None, apart=0):
    """The constant distance, index by index, from access `a` to access `b` of the same view, with
    `b`'s `shift` leaf `apart` trips after `a`'s; None where it is not a constant."""
    if a[0] != b[0]:
        return None
    out = []
    for x, y in zip(a[1], b[1]):
        (tx, cx), (ty, cy) = _linear(x, memo), _linear(y, memo)
        if tx != ty:
            return None
        out.append(cy - cx + (ty.get(shift, 0) * apart if shift is not None else 0))
    return out


def _apart(a, b, memo, shift=None, apart=0):
    """Do accesses `a` and `b` touch no element in common, `b` taken `apart` trips of the
    `shift` leaf after `a`?"""
    ra, rb = _root(a[0]), _root(b[0])
    if ra != rb:
        return _fresh(ra) and _fresh(rb)
    if a[2] is None or b[2] is None:
        return False
    d = _offsets(a, b, memo, shift, apart)
    if d is None:
        return False
    last = d.pop()
    return not any(d) and (last >= a[2] or -last >= b[2])


def _bank_step(a, b, memo, shift=None, apart=0):
    """Whether access `b` lies in the other bank from `a` ("other"), the same ("same") or either
    (None): a bank is a line of LINE_BYTES in turn, so a constant distance says which."""
    d = _offsets(a, b, memo, shift, apart)
    if d is None or any(d[:-1]):
        return None
    span = d[-1] * elem_bits(ir.MemRefType(a[0].type).element_type) // 8 % (LINE_BYTES * BANKS)
    return "other" if span == LINE_BYTES else "same" if span == 0 else None


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


def _owner_in(v, index):
    owner = v.owner if not isinstance(v, ir.BlockArgument) else None
    if isinstance(owner, (ir.Operation, ir.OpView)) and owner.operation in index:
        return owner.operation
    return None


def _stages(ops, iv):
    """Each op's stage, by the trips it runs after its input's pushes and pops (READ before them,
    ADD and WRITE after), for a body of reads, pure values, vcix ops and writes of the pops' sums;
    an op not in the map is address arithmetic, made again for each stage. None otherwise."""
    index = {o: i for i, o in enumerate(ops)}
    users = {o: [] for o in ops}
    for o in ops:
        for v in o.operands:
            p = _owner_in(v, index)
            if p is not None:
                users[p].append(o)
    units = [o for o in ops if _is_vcix(o)]
    stage = {o: VCIX for o in units}
    stack, fwd = [o for o in units if len(o.results)], set()
    while stack:
        for u in users[stack.pop()]:
            if u in fwd:
                continue
            if _is_vcix(u):
                return None
            fwd.add(u)
            stack.append(u)
    for o in ops:
        acc = _access(o)
        if o in fwd:
            if acc is not None and acc[3]:
                stage[o] = WRITE
            elif _is_value(o):
                stage[o] = ADD
            else:
                return None
        elif acc is not None and not acc[3]:
            stage[o] = READ
        elif not _is_vcix(o) and not _is_value(o):
            return None
    for p, q in zip(units, units[1:]):
        if _prepares(p) and not len(q.results):
            _lead(_source_read(p, index), _source_read(q, index), stage, iv)
    for o in ops:
        if o in stage or not _is_value(o):
            continue
        held = [stage[x] for x in (_owner_in(v, index) for v in o.operands) if x in stage]
        if held:
            stage[o] = max(held)
    return stage


def _source_read(push, index):
    """The one read a push's register comes from, through the values between; None if not one."""
    seen, stack, reads = set(), [push], []
    while stack:
        for v in stack.pop().operands:
            p = _owner_in(v, index)
            if p is None or p in seen or _is_vcix(p):
                continue
            seen.add(p)
            acc = _access(p)
            if acc is not None and not acc[3]:
                reads.append(p)
            elif _is_value(p):
                stack.append(p)
    return reads[0] if len(reads) == 1 else None


def _lead(prep, push, stage, iv):
    """The read of an input's vmatprep run `lead` trips ahead of the read of its push, the fewest
    that puts the two in different banks at every trip: two reads issued together then alternate."""
    if prep is None or push is None:
        return
    a, b, memo = _access(push), _access(prep), {}
    for lead in (0, 1, -1, 2, -2, 3, -3)[:2 * MAX_LEAD + 1]:
        if _bank_step(a, b, memo, iv, lead) == "other":
            if lead > 0:
                stage[prep] = READ - lead
            elif lead < 0:
                stage[push] = READ + lead
            return


def _keeps_memory(ops, stage, iv):
    """Does every read and write that the pipeline issues in another order than the loop did touch
    elements apart? A read of input r is issued at trip r + its stage, before that trip's writes."""
    pos = {o: i for i, o in enumerate(ops)}
    reads = [o for o in ops if o in stage and _is_read(o)]
    writes = [o for o in ops if o in stage and stage[o] == WRITE]
    span = max(stage.values()) - min(stage.values()) + 1
    memo = {}
    for r in reads:
        for w in writes:
            for d in range(-span, span + 1):
                before = d < 0 or (d == 0 and pos[r] < pos[w])
                now = d + stage[r] <= stage[w]
                if before != now and not _apart(_access(w), _access(r), memo, iv, d):
                    return False
    return True


def _trip_order(ops, stage, iv):
    """A trip's ops: the reads, each in the other bank from the one before where that is known,
    with the values on them; the writes; then the unit's ops, each pop behind the adds of the
    same pop an input earlier."""
    index = {o: i for i, o in enumerate(ops)}
    units = [o for o in ops if _is_vcix(o)]
    source = _pop_sources(ops, units)
    reads = [o for o in ops if o in stage and _is_read(o)]
    held = [o for o in ops if o in stage and _is_value(o) and o not in source]
    memo = {}
    order, left = [], sorted(reads, key=lambda o: (-stage[o], index[o]))
    while left:
        pick = left[0]
        if order:
            prev = order[-1]
            for r in left:
                ahead = stage[prev] - stage[r]
                if _bank_step(_access(prev), _access(r), memo, iv, ahead) == "other":
                    pick = r
                    break
        left.remove(pick)
        order.append(pick)
    placed = set(order)
    while held:
        ready = [o for o in held if all(p in placed or p not in stage
                                        for p in (_owner_in(v, index) for v in o.operands))]
        for o in ready:
            held.remove(o)
            placed.add(o)
            order.append(o)
        if not ready:
            raise RuntimeError(f"{NAME}: a value of the reads waits on itself")
    order += [o for o in ops if stage.get(o) == WRITE]
    adds = [o for o in ops if o in source and _is_value(o)]
    for k, u in enumerate(units):
        order += [o for o in adds if source.get(o) == k]
        order.append(u)
    if [o for o in order if _is_vcix(o)] != units:
        raise RuntimeError(f"{NAME} moved a vcix op past another")
    return order


def _is_read(op):
    acc = _access(op)
    return acc is not None and not acc[3]


def _pop_sources(ops, units):
    """Each op of `ops` a pop's value reaches, by the position in `units` of the first such pop."""
    index = {o: i for i, o in enumerate(ops)}
    source = {}
    for k, u in enumerate(units):
        stack = [u] if len(u.results) else []
        while stack:
            for x in [y for v in stack.pop().results for y in _users_of(v, index)]:
                if x not in source and not _is_vcix(x):
                    source[x] = k
                    stack.append(x)
    return source


def _pipeline(loop):
    """`loop` (0 to trips by 1) as a prologue, a loop whose trip t runs each op of stage s for the
    input t - s, and an epilogue: what crosses trips rides the loop's iter_args. The kernel's body
    block, or None for a loop it leaves as it is."""
    from mlir.dialects import affine, arith
    trips, bounds = trip_count(loop), loop_bounds(loop)
    if trips is None or bounds[0] != 0 or bounds[2] != 1:
        return None
    body = loop.regions[0].blocks[0]
    iv = body.arguments[0]
    ops = [o.operation for o in body.operations][:-1]
    stage = _stages(ops, iv)
    if stage is None:
        return None
    lo, hi = min(stage.values()), max(stage.values())
    kernel = trips + lo - hi
    if kernel < 2 or not _keeps_memory(ops, stage, iv):
        return None
    order = _trip_order(ops, stage, iv)
    index = {o: i for i, o in enumerate(ops)}
    slots = []
    for o in order:
        for v in o.results:
            need = max([stage[x] - stage[o] for x in _users_of(v, index) if x in stage] + [0])
            slots += [(v, j) for j in range(1, need + 1)]
    outer = ir.InsertionPoint(loop)
    vals, made = {}, {}
    straight = _Emitter(outer, iv, index, stage, made)
    for t in range(lo, hi):
        straight.trip(order, t, trips, vals)
    inits = [vals[(v, hi - j - stage[_owner_in(v, index)])] for v, j in slots]
    with outer:
        new_loop = affine.AffineForOp(0, kernel, 1, iter_args=inits)
    inner = ir.InsertionPoint(new_loop.body)
    with inner:
        ats = {s: arith.AddIOp(new_loop.induction_variable, const_index(hi - s)).result
               for s in sorted(set(stage.values()))}
    carried, now = dict(zip(slots, new_loop.inner_iter_args)), {}
    looped = _Emitter(inner, iv, index, stage, {})
    for o in order:
        def operand(x, s=stage[o]):
            p = _owner_in(x, index)
            if p not in stage:
                return looped.address(x, ats[s], s)
            j = s - stage[p]
            return now[x] if j == 0 else carried[(x, j)]
        now.update(zip(o.results, looped.emit(o, operand).results))
    with inner:
        affine.AffineYieldOp([now[v] if j == 1 else carried[(v, j - 1)] for v, j in slots])
    for (v, j), r in zip(slots, new_loop.results):
        vals[(v, hi + kernel - j - stage[_owner_in(v, index)])] = r
    for t in range(hi + kernel, trips + hi):
        straight.trip(order, t, trips, vals)
    loop.erase()
    return new_loop.body


class _Emitter:
    """Clones of a loop body's ops at one insertion point: address arithmetic made again for the
    input it addresses, once an input, and the staged ops of a trip of straight-line code."""

    def __init__(self, ip, iv, index, stage, made):
        self.ip, self.iv, self.index, self.stage = ip, iv, index, stage
        self.made, self.at = made, {}

    def address(self, v, at, key):
        """`v` of the address arithmetic for input `at` (an index value), made once a key."""
        if v == self.iv:
            return at
        p = _owner_in(v, self.index)
        if p is None:
            return v
        if (v, key) not in self.made:
            new = self.emit(p, lambda x: self.address(x, at, key))
            for old, got in zip(p.results, new.results):
                self.made[(old, key)] = got
        return self.made[(v, key)]

    def emit(self, o, operand):
        """A clone of `o` at the point, its operands `operand` of each, made before it."""
        got = [operand(x) for x in o.operands]
        new = o.clone(self.ip).operation
        for k, x in enumerate(got):
            new.operands[k] = x
        return new

    def _operand(self, x, b, vals):
        if _owner_in(x, self.index) in self.stage:
            return vals[(x, b)]
        return self.address(x, self.at[b], b)

    def trip(self, order, t, trips, vals):
        """Trip `t` as straight-line code: each op of stage s for input t - s, if there is one."""
        for o in order:
            b = t - self.stage[o]
            if not 0 <= b < trips:
                continue
            if b not in self.at:
                with self.ip:
                    self.at[b] = const_index(b)
            new = self.emit(o, lambda x, b=b: self._operand(x, b, vals))
            vals.update({(r, b): n for r, n in zip(o.results, new.results)})


def _users_of(v, index):
    out = []
    for use in v.uses:
        x = use.owner.operation if hasattr(use.owner, "operation") else use.owner
        if x in index:
            out.append(x)
    return out


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
