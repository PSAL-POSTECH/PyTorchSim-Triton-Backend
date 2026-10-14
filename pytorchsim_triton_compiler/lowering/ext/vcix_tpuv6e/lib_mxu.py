"""lib_mxu: what vcix_tpuv6e's matrix unit lowering reads around a dot -- a C proven zero over the
pops and its dead fill, and a loop only reading C run as the dot's epilogue at its last pops (or
stamped KEPT_APART with why not); how far an operand's masked load leaves data, and who reads
C past it; and the uses of a buffer between two ops."""
import sys
from collections import namedtuple

from .... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from ....dialects.torchsim import TransferOp, WaitOp, is_load  # noqa: E402
from ...passes.lib_bank_layout import BANK_VIEW_OP, bank_layout, bank_view, lanes_of  # noqa: E402
from ...passes.lib_ir import (VIEW_OPS, const_int, int_array, loop_bounds, op_of,  # noqa: E402
                              producer, transfer_read, transfer_write,
                              writes_through_this_operand, zero_of)
from ...passes.lib_matmul import KEPT_APART  # noqa: E402

Pops = namedtuple("Pops", "rows words reach")


GemmOperand = namedtuple("GemmOperand", "value vec pad")


ZeroInit = namedtuple("ZeroInit", "dot acc runs copy fill")


def dead_initializers(inits):
    """The copy nests into a zero-proven C whose pops overwrite all they wrote, nothing using C
    in between, and the zero fills nothing reads any more before a first-K dot overwrites them."""
    copies = [i.copy[0] for i in inits if i.copy is not None
              and i.copy[1] == i.runs and used_between(i.acc, i.copy[0], i.dot) is None]
    dead, fills = list(copies), []
    for i in inits:
        if not any(i.fill[0] == f for f, _ in fills):
            fills.append(i.fill)
    for fill, runs in fills:
        users = _block_users(_access_buffer(_nest(fill)[1][0]), fill.block)
        for o in _ops_after(fill):
            if not _among(o, users) or _among(o, copies):
                continue
            if any(o == i.dot and i.copy is None and i.fill[0] == fill
                   and covers(i.runs, runs) for i in inits):
                dead.append(fill)
            break
    return dead


def buffer_root(v):
    """The buffer `v` names: a global by its symbol, else the value under its view ops."""
    while True:
        p = producer(v)
        if p is not None and p.name == "memref.get_global":
            return ("global", ir.FlatSymbolRefAttr(p.attributes["name"]).value)
        if p is None or not _is_view(p):
            return v
        v = p.operands[0]


def _is_view(o):
    return o.name in VIEW_OPS or o.name == BANK_VIEW_OP


def is_plain(v):
    p = producer(v)
    return isinstance(v.type, ir.MemRefType) and (p is None or not _is_view(p))


_Use = namedtuple("_Use", "operand_number")


def _users(v):
    """(op, operand number) of every op that takes `v`'s buffer, past view ops, off the use-lists
    of each value naming that buffer -- for a global, every get_global of it in the function."""
    from ...passes.lib_ir import walk
    root = buffer_root(v)
    names = [root]
    if isinstance(root, tuple):
        func = producer(v)
        while func.name != "memref.get_global":
            func = producer(func.operands[0])
        while func.name != "func.func":
            func = func.parent
        names = [o.results[0] for o in walk(func) if o.name == "memref.get_global"
                 and ir.FlatSymbolRefAttr(o.attributes["name"]).value == root[1]]
    out = []
    while names:
        for use in names.pop().uses:
            o = op_of(use.owner)
            if _is_view(o):
                names.extend(o.results)
            else:
                out.append((o, use.operand_number))
    return out


def _block_users(v, block, writes=False):
    """The ops of `block` that hold a user of `v`'s buffer (`writes`: one writing it)."""
    out = []
    for o, k in _users(v):
        if writes and not writes_through_this_operand(o, _Use(k)):
            continue
        while o is not None and o.name != "func.func" and o.block != block:
            o = _parent(o)
        if o is not None and o.name != "func.func" and not _among(o, out):
            out.append(o)
    return out


def _among(o, ops):
    return any(o == x for x in ops)


def _ops_after(op):
    ops = [o.operation for o in op.block.operations]
    return ops[next(i for i, o in enumerate(ops) if o == op) + 1:]


def last_writer(op, buf):
    """The last op before `op` in its block that writes `buf`'s buffer, or None."""
    ops = [o.operation for o in op.block.operations]
    writers = _block_users(buf, op.block, True)
    for o in reversed(ops[:next(i for i, o in enumerate(ops) if o == op)]):
        if _among(o, writers):
            return o
    return None


def used_between(buf, a, b, writes=False, first=1):
    """The first op of the block from `a` (`first` = 0) or past it (1) up to, not at, `b` that uses
    (`writes`: writes) `buf`'s buffer, or None."""
    ops = [o.operation for o in a.block.operations]
    lo = next(i for i, o in enumerate(ops) if o == a)
    hi = next(i for i, o in enumerate(ops) if o == b)
    users = _block_users(buf, a.block, writes)
    return next((o for o in ops[lo + first:hi] if _among(o, users)), None)


def _nest(loop):
    """(the perfect affine.for nest from `loop`, the ops of its innermost body)."""
    nest = [loop]
    while True:
        inner = _body_ops(nest[-1])
        if len(inner) == 1 and inner[0].name == "affine.for":
            nest.append(inner[0])
            continue
        return nest, inner


def store_nest(loop, C, copy):
    """The runs of C's bank a nest at `loop` writes: one zero-splat store (copy None), or
    (source, runs) for one load of another buffer stored unchanged to the same element (copy)."""
    if loop.name != "affine.for":
        return None
    nest, inner = _nest(loop)
    if len(inner) != (2 if copy else 1) or inner[-1].name not in _STORES:
        return None
    store = inner[-1]
    dst, val = _access_buffer(store), store.operands[0]
    if buffer_root(dst) != buffer_root(C) or not _one_vector(val.type) or not _banked_alike(dst, C):
        return None
    if copy:
        load = inner[0]
        if (load.name not in _LOADS or val != load.results[0] or _access(load) != _access(store)
                or buffer_root(_access_buffer(load)) == buffer_root(C)
                or not _banked_alike(_access_buffer(load), C)):
            return None
    elif not _is_zero(_splat_of(val)):
        return None
    bounds = [loop_bounds(lp) for lp in nest]
    if any(b is None or None in b or b[2] <= 0 for b in bounds):
        return None
    ivs = [lp.regions[0].blocks[0].arguments[0] for lp in nest]
    try:
        starts = _consumer_starts(_access(store), bounds, ivs, bank_layout(C)[0])
    except _KeptApart as e:
        loop.attributes[KEPT_APART] = ir.StringAttr.get(f"not read as a zero initializer: {e}")
        return None
    runs = starts and runs_of(starts, ir.VectorType(val.type).shape[0])
    if not runs:
        return None
    return (_access_buffer(inner[0]), runs) if copy else runs


def _is_zero(attr):
    if attr is None:
        return False
    for kind in (ir.FloatAttr, ir.IntegerAttr):
        try:
            return kind(attr).value == 0
        except (TypeError, ValueError):
            continue
    return False


def covers(outer, inner):
    """Is every [lo, hi) run of `inner` inside one of `outer`'s?"""
    return all(any(a <= lo and hi <= b for a, b in outer) for lo, hi in inner)


_Epilogue = namedtuple("_Epilogue", "nest loaded inputs body stored d waits invariant width")


_LOADS = ("affine.vector_load", "vector.load")


_STORES = ("affine.vector_store", "vector.store")


_ELEMENTWISE_DIALECTS = ("arith", "math")


_SPLATS = ("vector.broadcast", "vector.splat")


_MAX_POINTS = 1 << 16


class _KeptApart(Exception):
    """Why a loop beside a dot stays a loop; stamped on it as KEPT_APART."""


def find_epilogue(op, g, c, ip):
    """The first loop after `op`, past nothing but pure scalar ops, when it reads C's elements only
    to write each, through elementwise ops (by dialect: the bindings' MemoryEffectsOpInterface can
    attach effects, not query them), to the same element of one other buffer."""
    ops = [o.operation for o in op.operation.block.operations]
    at = next(i for i, o in enumerate(ops) if o == op.operation)
    for o in ops[at + 1:]:
        if o.name == "affine.for":
            try:
                return _match_epilogue(op, o, g, c, ip)
            except _KeptApart as e:
                if _among(o, _block_users(op.operands[2], o.block)):
                    o.attributes[KEPT_APART] = ir.StringAttr.get(f"not run at the pops: {e}")
                return None
        if (len(o.regions) or not len(o.results) or o.name.split(".")[0] != "arith"
                or any(isinstance(v.type, (ir.MemRefType, ir.VectorType))
                       for v in list(o.operands) + list(o.results))):
            return None
    return None


def _match_epilogue(op, loop, g, c, ip):
    """find_epilogue's loop as an _Epilogue, or _KeptApart saying why it stays a loop."""
    C = op.operands[2]
    nest, inner = _nest(loop)
    bounds = [loop_bounds(lp) for lp in nest]
    if any(b is None or None in b or b[2] <= 0 for b in bounds):
        raise _KeptApart("its loop bounds are not constants")
    ivs = [lp.regions[0].blocks[0].arguments[0] for lp in nest]
    loads, stores, waits, body = [], [], [], []
    for o in inner:
        if o.name in _LOADS:
            loads.append(o)
        elif o.name in _STORES:
            stores.append(o)
        elif o.name == WaitOp.OPERATION_NAME:
            waits.append(o)
        elif not len(o.regions) and (o.name.split(".")[0] in _ELEMENTWISE_DIALECTS
                                     or o.name in _SPLATS):
            body.append(o)
        else:
            raise _KeptApart(f"its body holds {o.name}, not an elementwise op")
    if len(stores) != 1:
        raise _KeptApart(f"it stores {len(stores)} times, not once")
    store = stores[0]
    D, stored = _access_buffer(store), store.operands[0]
    reads_c = [x for x in loads if _access_buffer(x) == C]
    if len(reads_c) != 1 or not _one_vector(stored.type):
        raise _KeptApart("it does not read C once into a 1-D vector")
    width_cons = ir.VectorType(stored.type).shape[0]
    access = _access(store)
    buffers = [C, D] + [_access_buffer(x) for x in loads if _access_buffer(x) != C]
    if len({buffer_root(x) for x in buffers}) != len(buffers):
        raise _KeptApart("two of C, its output and its other inputs are one buffer")
    _refuse_overlap(op, loop, D, [x for x in buffers[2:]])
    for x in loads:
        if _access(x) != access or not _one_vector(x.results[0].type) \
                or ir.VectorType(x.results[0].type).shape[0] != width_cons:
            raise _KeptApart("a load does not read the element its store writes")
    if not all(_banked_alike(x, C) for x in buffers[1:]):
        raise _KeptApart("a buffer it touches is not banked as C is")
    defined = {r for o in loads + body for r in o.results}
    for o in body:
        for v in o.operands:
            if v in ivs or (v not in defined and not _dominates(v, op.operation)):
                raise _KeptApart(f"{o.name} takes a value the pops cannot see")
        for v in list(o.operands) + list(o.results):
            if isinstance(v.type, ir.VectorType) and not (
                    _one_vector(v.type) and ir.VectorType(v.type).shape[0] == width_cons):
                raise _KeptApart(f"{o.name} works on a vector of another width")
    if stored not in defined:
        raise _KeptApart("what it stores is not computed in it")
    for w in waits:
        if not all(_dominates(v, op.operation) for v in w.operands):
            raise _KeptApart("a wait in it takes a value the pops cannot see")
    width = ir.VectorType(c.vec).shape[0]
    strides = bank_layout(C)[0]
    cons = _consumer_starts(access, bounds, ivs, strides)
    gr, _, M, rows, SS = g.reach
    pops = [n * M + m * rows + i for n in range(gr[0]) for m in range(gr[2])
            for i in range(0, g.rows, g.words)]
    if runs_of(cons, width_cons) is None or runs_of(cons, width_cons) != runs_of(pops, width):
        raise _KeptApart("the elements it reads are not the ones the pops write")
    invariant = {}
    for o in body:
        for v in o.operands:
            if v in defined or v in invariant or not isinstance(v.type, ir.VectorType):
                continue
            splat = _splat_of(v)
            if width != width_cons and splat is None:
                raise _KeptApart("an operand of another width is not a splat")
            invariant[v] = splat
    with ip:
        for v, splat in list(invariant.items()):
            if splat is None:
                invariant[v] = v
                continue
            ty = _retype(v.type, width)
            invariant[v] = _build_constant(ty, splat)
        inputs = []
        for x in loads:
            buf = _access_buffer(x)
            if buf == C:
                continue
            elt = ir.MemRefType(buf.type).element_type
            inputs.append((x.results[0], GemmOperand(bank_view(buf, SS),
                                                      ir.VectorType.get([width], elt),
                                                      zero_of(elt))))
        d = GemmOperand(bank_view(D, SS),
                         ir.VectorType.get([width], ir.MemRefType(D.type).element_type), None)
    return _Epilogue(loop, reads_c[0].results[0], inputs, body, stored, d, waits, invariant,
                     width)


def _refuse_overlap(op, loop, D, inputs):
    """_KeptApart unless, from the dot to the loop, nothing but the loop uses D's buffer -- the
    pops write D from the first last-K one on, while the dot still reads A and B -- and nothing
    writes an input the loop reads."""
    user = used_between(D, op.operation, loop, first=0)
    if user is not None:
        raise _KeptApart(f"its output buffer is also used by {user.name} before it, which the "
                         f"stream would overwrite while it is still being read")
    for x in inputs:
        user = used_between(x, op.operation, loop, writes=True, first=0)
        if user is not None:
            raise _KeptApart(f"{user.name} writes an input it reads before it runs")


def _body_ops(loop):
    return [o.operation for o in loop.regions[0].blocks[0].operations
            if o.operation.name != "affine.yield"]


def _access_buffer(o):
    return o.operands[1] if o.name in _STORES else o.operands[0]


def _access(o):
    """(map or None, index operands) of a vector load or store."""
    idx = list(o.operands[2:] if o.name in _STORES else o.operands[1:])
    m = ir.AffineMapAttr(o.attributes["map"]).value if "map" in o.attributes else None
    return (m, tuple(idx))


def _one_vector(ty):
    return isinstance(ty, ir.VectorType) and len(ir.VectorType(ty).shape) == 1


def _banked_alike(x, C):
    """Does element o of `x` sit, in each lane, where element o of C does -- same shape, strides,
    lanes, and neither a view whose own offset would shift it?"""
    if any(producer(v) is not None and producer(v).name in VIEW_OPS for v in (x, C)):
        return False
    tx, tc = ir.MemRefType(x.type), ir.MemRefType(C.type)
    return (list(tx.shape) == list(tc.shape) and bank_layout(x) == bank_layout(C)
            and lanes_of(x) == lanes_of(C) and lanes_of(C) is not None)


def _dominates(v, at):
    p = producer(v)
    anc = at
    if p is None:
        blk = v.owner
        while anc is not None:
            if anc.block == blk:
                return True
            anc = _parent(anc)
        return False
    while anc is not None and anc.block != p.block:
        anc = _parent(anc)
    return anc is not None and p.is_before_in_block(anc)


def _parent(o):
    try:
        return o.parent
    except (ValueError, AttributeError):
        return None


def _consumer_starts(access, bounds, ivs, strides):
    """Each in-bank offset the loop's access starts a vector at, in iteration order; _KeptApart
    when they cannot be listed."""
    import itertools
    trips = 1
    for lo, hi, st in bounds:
        trips *= max(0, -(-(hi - lo) // st))
    if trips > _MAX_POINTS:
        raise _KeptApart(f"its {trips} iterations are more than the {_MAX_POINTS} whose offsets "
                         f"this pass checks one by one")
    m, idx = access
    out = []
    for point in itertools.product(*[range(lo, hi, st) for lo, hi, st in bounds]):
        env = list(zip(ivs, point))
        vals = [_index_value(v, env) for v in idx]
        if None in vals:
            raise _KeptApart("an index is neither a loop counter nor a constant")
        if m is not None:
            dims, syms = vals[:m.n_dims], vals[m.n_dims:]
            vals = [_eval_affine(e, dims, syms) for e in m.results]
        if len(vals) != len(strides):
            raise _KeptApart("its access has not one index per dimension")
        out.append(sum(v * s for v, s in zip(vals, strides)))
    return out


def _index_value(v, env):
    from ...passes.lib_ir import const_int
    for iv, x in env:
        if v == iv:
            return x
    return const_int(v)


def _eval_affine(e, dims, syms):
    if isinstance(e, ir.AffineConstantExpr):
        return e.value
    if isinstance(e, ir.AffineDimExpr):
        return dims[e.position]
    if isinstance(e, ir.AffineSymbolExpr):
        return syms[e.position]
    for kind, f in ((ir.AffineAddExpr, lambda x, y: x + y), (ir.AffineMulExpr, lambda x, y: x * y),
                    (ir.AffineModExpr, lambda x, y: x % y),
                    (ir.AffineFloorDivExpr, lambda x, y: x // y),
                    (ir.AffineCeilDivExpr, lambda x, y: -(-x // y))):
        if isinstance(e, kind):
            return f(_eval_affine(e.lhs, dims, syms), _eval_affine(e.rhs, dims, syms))
    raise ValueError(f"an affine expression of no known kind: {e}")


def runs_of(starts, width):
    """The [lo, hi) runs `width`-wide vectors at `starts` cover, or None when two overlap."""
    out = []
    for s in sorted(starts):
        if out and s < out[-1][1]:
            return None
        if out and s == out[-1][1]:
            out[-1][1] = s + width
        else:
            out.append([s, s + width])
    return out


def _splat_of(v):
    p = producer(v)
    if p is None or p.name != "arith.constant":
        return None
    attr = ir.DenseElementsAttr(p.attributes["value"])
    return attr.get_splat_value() if attr.is_splat else None


def _retype(ty, width):
    if isinstance(ty, ir.VectorType):
        return ir.VectorType.get([width], ir.VectorType(ty).element_type)
    return ty


def _build_constant(ty, splat):
    from mlir.dialects import arith
    return arith.ConstantOp(ty, ir.DenseElementsAttr.get_splat(ty, splat)).result


def build_waits(ip, epi):
    """The epilogue's waits, once, before its first store."""
    with ip:
        for w in epi.waits:
            WaitOp(w.operands[0], w.operands[1])


def build_epilogue(epi, total, at):
    """The epilogue on one pop's sum: its other inputs read where the pop's C element sits, its
    ops retyped to the pop's width, the result written there in its own buffer."""
    vals = {epi.loaded: total}
    for loaded, operand in epi.inputs:
        vals[loaded] = transfer_read(operand.vec, operand.value, at, operand.pad, True)
    for o in epi.body:
        attrs = {n: o.attributes[n] for n in (getattr(x, 'name', x) for x in o.attributes)}
        if o.name == "arith.constant" and isinstance(o.results[0].type, ir.VectorType):
            ty = _retype(o.results[0].type, epi.width)
            attrs["value"] = ir.DenseElementsAttr.get_splat(ty, _splat_of(o.results[0]))
        new = ir.Operation.create(
            o.name, results=[_retype(r.type, epi.width) for r in o.results],
            operands=[vals.get(v, epi.invariant.get(v, v)) for v in o.operands],
            attributes=attrs)
        for old, r in zip(o.results, new.results):
            vals[old] = r
    transfer_write(vals[epi.stored], epi.d.value, at, True)


def interval(v, known, depth=0):
    """(low, high), inclusive, of the integer `v`, or None: constants, the values `known` gives
    (the program ids, each within its axis of the launch grid), and what arith's add, sub, mul,
    min, max, and div and rem of non-negatives by positives make of them. An over-estimate: the
    two sides of an op are taken apart, so `x - x` is not 0."""
    for value, bounds in known:
        if value == v:
            return bounds
    o = producer(v)
    if o is None or depth > 64:
        return None
    c = const_int(v)
    if c is not None:
        return c, c
    if o.name in ("arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui"):
        return interval(o.operands[0], known, depth + 1)
    if o.name not in ("arith.addi", "arith.subi", "arith.muli", "arith.minsi", "arith.maxsi",
                      "arith.divsi", "arith.remsi"):
        return None
    a = interval(o.operands[0], known, depth + 1)
    b = interval(o.operands[1], known, depth + 1)
    if a is None or b is None:
        return None
    if o.name == "arith.addi":
        return a[0] + b[0], a[1] + b[1]
    if o.name == "arith.subi":
        return a[0] - b[1], a[1] - b[0]
    if o.name == "arith.muli":
        corners = [x * y for x in a for y in b]
        return min(corners), max(corners)
    if o.name == "arith.minsi":
        return min(a[0], b[0]), min(a[1], b[1])
    if o.name == "arith.maxsi":
        return max(a[0], b[0]), max(a[1], b[1])
    if a[0] < 0 or b[0] <= 0:
        return None
    if o.name == "arith.divsi":
        return a[0] // b[1], a[1] // b[0]
    return 0, min(a[1], b[1] - 1)


def program_ids(op, grid):
    """What is known of the kernel's program ids, for interval(): the last three integer arguments
    of `op`'s function are the ids on x, y and z, each below its axis of the launch `grid`."""
    func = op.operation
    while func.name != "func.func":
        func = op_of(func.parent)
    args = [a for a in func.regions[0].blocks[0].arguments if isinstance(a.type, ir.IntegerType)]
    if len(args) < 6 or not grid:
        return []           # no launch grid said: nothing is known of the ids
    g = (list(grid) + [1, 1, 1])[:3]
    return [(arg, (0, max(1, int(n)) - 1)) for arg, n in zip(args[-3:], g)]


def data_reach(op, buf, axis, known):
    """How far the data of `buf` reaches along its `axis` when `op` runs, or None for all of it:
    its last writer in the block is a load of the whole tile, `buf` itself, that clamps the axis
    and fills what is past the clamp with zeros -- the high end of that clamp."""
    w = last_writer(op.operation, buf)
    if w is None or w.name != TransferOp.OPERATION_NAME or not is_load(w):
        return None
    tx = w.opview
    if (tx.index is not None or tx.dyn_fill is not None or tx.masked_axes is None
            or tx.masked_fill is None or ir.IntegerAttr(tx.masked_fill).value != 0
            or tx.sram.type != buf.type or interval(tx.sram_idx, known) != (0, 0)):
        return None
    if tx.subtile_size is not None and \
            list(int_array(tx.subtile_size)) != list(ir.MemRefType(buf.type).shape):
        return None
    axes = list(int_array(tx.masked_axes))
    if axis not in axes:
        return None
    high = interval(list(tx.clamp)[2 * axes.index(axis) + 1], known)
    return None if high is None else high[1]


_NEST_OPS = ("affine.for", "affine.yield", "affine.apply", "scf.for", "scf.yield",
             "vector.broadcast", "vector.splat")


def within(o, nest):
    """Whether the op `o` is `nest` or inside it."""
    while o is not None and o.name != "func.func":
        if o == nest:
            return True
        o = _parent(o)
        o = op_of(o) if o is not None else None
    return False


def _carried_to(load, buf, op):
    """The buffer an elementwise loop nest moves `buf` to, column for column, or None: `load`
    reads `buf` in a nest of loops that holds nothing but loads, arithmetic and stores, the dot
    `op` is not in it, and every store is to one other buffer at the indices a load of `buf` has.
    What is past a column of `buf` is then past the same column of that buffer, and nowhere else."""
    from ...passes.lib_ir import walk
    if load.name not in _LOADS or buffer_root(load.operands[0]) != buffer_root(buf):
        return None
    nest, o = None, load
    while True:
        o = _parent(o)
        o = op_of(o) if o is not None else None
        if o is None or o.name not in ("affine.for", "scf.for"):
            break
        nest = o
    if nest is None or within(op.operation, nest):
        return None
    reads, stores = set(), []
    for x in walk(nest):
        if x.name in _LOADS:
            if buffer_root(x.operands[0]) == buffer_root(buf):
                reads.add(_access(x))
        elif x.name in _STORES:
            stores.append(x)
        elif x.name not in _NEST_OPS and x.name.split(".")[0] not in _ELEMENTWISE_DIALECTS:
            return None
    if not stores:
        return None
    to = stores[0].operands[1]
    for x in stores:
        if buffer_root(x.operands[1]) != buffer_root(to) or buffer_root(to) == buffer_root(buf) \
                or _access(x) not in reads:
            return None
    return to


def read_only_within(op, buf, axis, columns, known, depth=0):
    """Whether nothing reads `buf` past its first `columns` along `axis` but what cannot tell:
    every user is the dot `op`, a write, a store to memory that clamps the axis within the
    columns, or a loop that moves the buffer element for element to another of which the same
    holds (_carried_to: the same indices, so the same axis)."""
    for o, k in _users(buf):
        if o == op.operation or writes_through_this_operand(o, _Use(k)):
            continue
        if o.name == TransferOp.OPERATION_NAME and not is_load(o):
            tx = o.opview
            axes = [] if tx.masked_axes is None else list(int_array(tx.masked_axes))
            high = (interval(list(tx.clamp)[2 * axes.index(axis) + 1], known) if axis in axes
                    else None)
            if high is None or high[1] > columns:
                return False
            continue
        to = _carried_to(o, buf, op)
        if (to is None or depth > 2 or ir.MemRefType(to.type).rank != ir.MemRefType(buf.type).rank
                or not read_only_within(op, to, axis, columns, known, depth + 1)):
            return False
    return True
