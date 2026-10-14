"""lib_matmul: what lowering a torchsim.dot needs whatever unit it runs on -- the refusals, a C
proven zero and its dead fill, and a loop only reading C run as the dot's epilogue at its last pops
(or stamped KEPT_APART with why not); and the uses of a buffer between two ops."""
import sys
from collections import namedtuple

from ... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from .lib_bank_layout import BANK_VIEW_OP, bank_layout, bank_view, lanes_of  # noqa: E402
from ...dialects.torchsim import WaitOp  # noqa: E402
from .lib_ir import (DOT_OP, VIEW_OPS, array_kind, loop_bounds, op_of, producer,  # noqa: E402
                     transfer_read, transfer_write, writes_through_this_operand, zero_of)
from .lib_indexing_maps import MATMUL_K, MATMUL_M, MATMUL_N, matmul_axis  # noqa: E402

MATMUL_OPS = (DOT_OP,)


_Gemm = namedtuple(
    "_Gemm",
    "spad_map a64 rvl rvl_acc elen nr_element nr_acc M_val push_val M_LOOP arith reach epi zero")


_GemmOperand = namedtuple("_GemmOperand", "value vec word pad")


class ArrayRefusal(Exception):
    """The array cannot take this matmul."""


def _refuse_unlowerable(op, elt, acc_elt, b_elt):
    if array_kind(op) == "batch_matmul":
        raise ArrayRefusal(f"a batched torchsim.dot: this lowering walks n, k and m and has no batch "
                           f"loop, so it would compute the first batch only: {op}")
    if b_elt != elt:
        raise NotImplementedError(
            f"matmul operands A ({elt}) and B ({b_elt}) have "
            f"different element types; every push is typed from A's element type at one sew, "
            f"so B's elements would be read as A's")
    if isinstance(acc_elt, ir.BF16Type):
        raise NotImplementedError(
            f"a bf16 accumulator ({acc_elt}): the pop is added into C, and bf16 arithmetic "
            f"has no instruction on this machine; accumulate in f32")
    for t in (elt, acc_elt):
        if isinstance(t, ir.IntegerType):
            raise NotImplementedError(
                f"integer matmul ({t}) has no path to the matrix unit: its words "
                f"are f32, two f16 or two bf16 (vcix-accelerator examples/tpuv6, "
                f"formats 0, 1 and 3), so there is no integer MAC to lower to")
    _refuse_accumulator_in_its_loop(op)


def _refuse_accumulator_in_its_loop(op):
    innermost = None
    parent = op.operation.parent
    while parent is not None:
        if parent.name in ("affine.for", "scf.for") and innermost is None:
            innermost = parent
        parent = parent.parent

    if innermost is not None:
        owner = op.operands[2].owner
        owner = op_of(owner)
        anc = getattr(owner, "parent", None)
        while anc is not None:
            if anc == innermost:
                raise NotImplementedError(
                    "the matmul accumulator is allocated inside its own loop -- "
                    "that loop is not a reduction axis")
            anc = anc.parent


def _get_mnk(op):
    mtA, mtB = ir.MemRefType(op.operands[0].type), ir.MemRefType(op.operands[1].type)
    return (mtA.shape[matmul_axis(op, 0, MATMUL_M)],
            mtB.shape[matmul_axis(op, 1, MATMUL_N)],
            mtA.shape[matmul_axis(op, 0, MATMUL_K)])


_ZeroInit = namedtuple("_ZeroInit", "dot acc runs whole copy fill")


def _dead_initializers(inits):
    """The copy nests into a zero-proven C whose pops overwrite all they wrote, nothing using C
    in between, and the zero fills nothing reads any more before a first-K dot overwrites them."""
    copies = [i.copy[0] for i in inits if i.copy is not None and i.whole
              and i.copy[1] == i.runs and _used_between(i.acc, i.copy[0], i.dot) is None]
    dead, fills = list(copies), []
    for i in inits:
        if not any(i.fill[0] == f for f, _ in fills):
            fills.append(i.fill)
    for fill, runs in fills:
        users = _block_users(_access_buffer(_nest(fill)[1][0]), fill.block)
        for o in _ops_after(fill):
            if not _among(o, users) or _among(o, copies):
                continue
            if any(o == i.dot and i.copy is None and i.fill[0] == fill and i.whole
                   and _covers(i.runs, runs) for i in inits):
                dead.append(fill)
            break
    return dead


def _root(v):
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


def _plain(v):
    p = producer(v)
    return isinstance(v.type, ir.MemRefType) and (p is None or not _is_view(p))


_Use = namedtuple("_Use", "operand_number")


def _users(v):
    """(op, operand number) of every op that takes `v`'s buffer, past view ops, off the use-lists
    of each value naming that buffer -- for a global, every get_global of it in the function."""
    from .lib_ir import walk
    root = _root(v)
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


def _last_writer(op, buf):
    """The last op before `op` in its block that writes `buf`'s buffer, or None."""
    ops = [o.operation for o in op.block.operations]
    writers = _block_users(buf, op.block, True)
    for o in reversed(ops[:next(i for i, o in enumerate(ops) if o == op)]):
        if _among(o, writers):
            return o
    return None


def _used_between(buf, a, b, writes=False, first=1):
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


def _store_nest(loop, C, copy):
    """The runs of C's bank a nest at `loop` writes: one zero-splat store (copy None), or
    (source, runs) for one load of another buffer stored unchanged to the same element (copy)."""
    if loop.name != "affine.for":
        return None
    nest, inner = _nest(loop)
    if len(inner) != (2 if copy else 1) or inner[-1].name not in _STORES:
        return None
    store = inner[-1]
    dst, val = _access_buffer(store), store.operands[0]
    if _root(dst) != _root(C) or not _one_vector(val.type) or not _banked_alike(dst, C):
        return None
    if copy:
        load = inner[0]
        if (load.name not in _LOADS or val != load.results[0] or _access(load) != _access(store)
                or _root(_access_buffer(load)) == _root(C)
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
    runs = starts and _runs(starts, ir.VectorType(val.type).shape[0])
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


def _covers(outer, inner):
    """Is every [lo, hi) run of `inner` inside one of `outer`'s?"""
    return all(any(a <= lo and hi <= b for a, b in outer) for lo, hi in inner)


_Epilogue = namedtuple("_Epilogue", "nest loaded inputs body stored d waits invariant width")


_LOADS = ("affine.vector_load", "vector.load")


_STORES = ("affine.vector_store", "vector.store")


_ELEMENTWISE_DIALECTS = ("arith", "math")


_SPLATS = ("vector.broadcast", "vector.splat")


_MAX_POINTS = 1 << 16


KEPT_APART = "torchsim.kept_apart"


class _KeptApart(Exception):
    """Why a loop beside a dot stays a loop; stamped on it as KEPT_APART."""


def _find_epilogue(op, g, c, ip):
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
    """_find_epilogue's loop as an _Epilogue, or _KeptApart saying why it stays a loop."""
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
    if len({_root(x) for x in buffers}) != len(buffers):
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
            for i in range(0, g.M_LOOP, g.nr_acc)]
    if _runs(cons, width_cons) is None or _runs(cons, width_cons) != _runs(pops, width):
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
            inputs.append((x.results[0], _GemmOperand(bank_view(buf, SS),
                                                      ir.VectorType.get([width], elt),
                                                      None, zero_of(elt))))
        d = _GemmOperand(bank_view(D, SS),
                         ir.VectorType.get([width], ir.MemRefType(D.type).element_type),
                         None, None)
    return _Epilogue(loop, reads_c[0].results[0], inputs, body, stored, d, waits, invariant,
                     width)


def _refuse_overlap(op, loop, D, inputs):
    """_KeptApart unless, from the dot to the loop, nothing but the loop uses D's buffer -- the
    pops write D from the first last-K one on, while the dot still reads A and B -- and nothing
    writes an input the loop reads."""
    user = _used_between(D, op.operation, loop, first=0)
    if user is not None:
        raise _KeptApart(f"its output buffer is also used by {user.name} before it, which the "
                         f"stream would overwrite while it is still being read")
    for x in inputs:
        user = _used_between(x, op.operation, loop, writes=True, first=0)
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
    from .lib_ir import const_int
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


def _runs(starts, width):
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


def _build_waits(ip, epi):
    """The epilogue's waits, once, before its first store."""
    with ip:
        for w in epi.waits:
            WaitOp(w.operands[0], w.operands[1])


def _build_epilogue(epi, total, at, fits):
    """The epilogue on one pop's sum: its other inputs read where the pop's C element sits, its
    ops retyped to the pop's width, the result written there in its own buffer."""
    vals = {epi.loaded: total}
    for loaded, operand in epi.inputs:
        vals[loaded] = transfer_read(operand.vec, operand.value, at, operand.pad, fits)
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
    transfer_write(vals[epi.stored], epi.d.value, at, fits)
