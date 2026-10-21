"""lib_mxu: what vcix_tpuv6e's matrix unit lowering reads around a dot -- a C proven zero over the
pops and its dead fill, and a loop only reading C run as the dot's epilogue at its last pops, what it
folds along C's rows carried by them (or stamped KEPT_APART with why not); how far an operand's
masked load leaves data, and who reads C past it; and the uses of a buffer between two ops."""
import sys
from collections import namedtuple

from .... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from ....dialects.torchsim import TransferOp, WaitOp, effects_on, is_load  # noqa: E402
from ...passes.lib_bank_layout import BANK_VIEW_OP, bank_layout, bank_view, lane_extent  # noqa: E402
from ...passes.lib_ir import (VIEW_OPS, const_int, eval_affine, int_array, loop_bounds, op_of,  # noqa: E402
                              producer, touches_data, transfer_read, transfer_write,
                              writes_through_this_operand, zero_of)
from ...passes.lib_live import interval, program_ids  # noqa: E402,F401
from ...passes.lib_matmul import KEPT_APART  # noqa: E402
from ...passes.p05_vectorize import REDUCE_KIND  # noqa: E402

Pops = namedtuple("Pops", "rows words reach")


GemmOperand = namedtuple("GemmOperand", "value vec pad")


ZeroInit = namedtuple("ZeroInit", "dot acc runs fill")


def dead_initializers(inits):
    """The zero fills nothing reads any more before a first-K dot overwrites them."""
    dead, fills = [], []
    for i in inits:
        if not any(i.fill[0] == f for f, _ in fills):
            fills.append(i.fill)
    for fill, runs in fills:
        users = _block_users(_access_buffer(_nest(fill)[1][0]), fill.block)
        for o in _ops_after(fill):
            if not _among(o, users):
                continue
            if any(o == i.dot and i.fill[0] == fill and covers(i.runs, runs) for i in inits):
                dead.append(fill)
            break
    return dead


def buffer_root(v):
    """The buffer `v` names: a global by its symbol, else the value under its view ops -- an
    allocated buffer itself."""
    while True:
        p = producer(v)
        if p is not None and p.name == "memref.get_global":
            return ("global", ir.FlatSymbolRefAttr(p.attributes["name"]).value)
        if p is None or not _is_view(p):
            return v
        v = p.operands[0]


def is_fresh(root):
    """Is `root` (a buffer_root) a buffer no other root can name: a global, or one the op making it
    states it allocates (its effects, not its name)?"""
    if isinstance(root, tuple):
        return True
    p = producer(root)
    return p is not None and "allocate" in effects_on(p, root)


def _is_view(o):
    from ...passes.lib_pack import PACK_OP
    return o.name in VIEW_OPS or o.name in (BANK_VIEW_OP, PACK_OP)


def is_plain(v):
    p = producer(v)
    return isinstance(v.type, ir.MemRefType) and (p is None or not _is_view(p))


_Use = namedtuple("_Use", "operand_number")


def _users(v):
    """(op, operand number) of every op that may read or write `v`'s buffer (touches_data), past
    view ops, off the use-lists of each value naming it -- for a global, its every get_global."""
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
            elif touches_data(o, use):
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


def store_nest(loop, C):
    """The runs of C's bank a nest at `loop` writes with one zero-splat store, or None."""
    if loop.name != "affine.for":
        return None
    nest, inner = _nest(loop)
    if len(inner) != 1 or inner[-1].name not in _STORES:
        return None
    store = inner[-1]
    dst, val = _access_buffer(store), store.operands[0]
    if buffer_root(dst) != buffer_root(C) or not _one_vector(val.type) or not _banked_alike(dst, C):
        return None
    if not _is_zero(_splat_of(val)):
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
    return runs or None


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


_Epilogue = namedtuple("_Epilogue", "nest loaded inputs body stored d waits invariant width walk "
                                    "coords n_axis prelude reduce outputs")

# The running reduction of an epilogue's innermost loop: the value it carries, what each trip
# combines into it, its seed, the loop's result, and the ops after the loop that make the row's
# answer of that result and store it.
_Reduction = namedtuple("_Reduction", "carried combined init result post")


_LOADS = ("affine.vector_load", "vector.load")


_STORES = ("affine.vector_store", "vector.store")


_ELEMENTWISE_DIALECTS = ("arith", "math")


_SPLATS = ("vector.broadcast", "vector.splat")


_MAX_POINTS = 1 << 16


class _KeptApart(Exception):
    """Why a loop beside a dot stays a loop; stamped on it as KEPT_APART."""


# Ops whose element i the pops make, at their own width, as the walk makes it at its own: from the
# operands' element i, from i itself (vector.step), or the lane's index (torchsim.vlane_idx).
_ALIGNED = _SPLATS + ("vector.step", "torchsim.vlane_idx")


# Ops after a reducing loop that fold what it carried into its row's answer.
_FOLDS = ("vector.shape_cast", "vector.multi_reduction", "vector.shuffle",
          "vector.extract_strided_slice")


# The combining ops whose answer does not depend on the order they take the terms in: the pops
# reach a row's elements in another order, and fewer at a time, than the walk.
_ORDERLESS = tuple(k for k in REDUCE_KIND if k not in ("arith.addf", "arith.mulf"))


# The _ORDERLESS ops whose answer changes with how many times a term is folded, by their identity:
# a row's seed is folded once a lane, so at the pops' width as many times as it has lanes there.
_COUNTED = {"arith.addi": 0, "arith.muli": 1, "arith.xori": 0}


def _is_aligned(o):
    return o.name.split(".")[0] in _ELEMENTWISE_DIALECTS or o.name in _ALIGNED


def find_epilogue(op, g, c, ip):
    """The first loop after `op`, past nothing but ops of no effect that make values from values
    (elementwise, _ALIGNED), when it reads C's elements only to write each, through elementwise
    ops (by dialect: the bindings' MemoryEffectsOpInterface can attach effects, not query them), to
    the same element of one other buffer, or to fold them along each row of C (_Reduction). The
    loops around its innermost may make, a row at a time, what the row's elements take."""
    o = epilogue_loop(op)
    if o is None:
        return None
    try:
        return _match_epilogue(op, o, g, c, ip)
    except _KeptApart as e:
        if _among(o, _block_users(op.operands[2], o.block)):
            o.attributes[KEPT_APART] = ir.StringAttr.get(f"not run at the pops: {e}")
        return None


def epilogue_loop(op):
    """The loop find_epilogue tries to run at `op`'s pops, or None."""
    ops = [o.operation for o in op.operation.block.operations]
    at = next(i for i, o in enumerate(ops) if o == op.operation)
    for o in ops[at + 1:]:
        if o.name == "affine.for":
            return o
        if (len(o.regions) or not len(o.results) or not _is_aligned(o)
                or any(isinstance(v.type, ir.MemRefType)
                       for v in list(o.operands) + list(o.results))):
            return None
    return None


def runs_at_pops(op, g, vec):
    """The loop find_epilogue would run at `op`'s pops of `g`, each a `vec`, or None; nothing is built
    or stamped."""
    o = epilogue_loop(op)
    if o is None:
        return None
    try:
        _match_epilogue(op, o, g, GemmOperand(None, vec, None), None)
    except _KeptApart:
        return None
    return o


def widest_element_bits(loop):
    """The widest element, in bits, of a vector `loop` makes or takes (an index is 64): what a pop
    of the elements the loop runs at it holds at the most, in the epilogue's widest type."""
    from mlir.ir import IndexType
    from ...passes.lib_ir import elem_bits, walk
    widest = 0
    for o in [loop] + walk(loop):
        for v in list(o.operands) + list(o.results):
            if isinstance(v.type, ir.VectorType):
                e = ir.VectorType(v.type).element_type
                widest = max(widest, 64 if isinstance(e, IndexType) else elem_bits(e))
    return widest


def registers_at_pops(loop, words, vlen):
    """The vector registers the epilogue `loop` holds at once where it runs at pops of `words`
    elements: the most its ops (the rows' and the innermost body's, in order) keep live together,
    the values it takes from outside, and the ones its loops carry, each `words` of its own element
    wide (an index is 64 bits, a mask a register). A splat constant holds none: it goes to an
    instruction as a scalar."""
    from mlir.ir import BlockArgument, IndexType
    from ...passes.lib_ir import elem_bits

    def regs(v):
        if not isinstance(v.type, ir.VectorType) or ir.VectorType(v.type).shape == [1]:
            return 0
        if _splat_of(v) is not None:
            return 0
        e = ir.VectorType(v.type).element_type
        bits = 64 if isinstance(e, IndexType) else elem_bits(e)
        return -(-words * bits // vlen)
    nest, prelude, inner, post = _levels(loop)
    ops = prelude + inner + post
    made = {r for o in ops for r in o.results}
    last = {}
    for i, o in enumerate(ops):
        for v in o.operands:
            last[v] = i
    carried = [a for lp in nest for a in list(lp.regions[0].blocks[0].arguments)[1:]]
    outside = {v for o in ops for v in o.operands
               if v not in made and not isinstance(v, BlockArgument) and isinstance(v.type, ir.VectorType)}
    peak = 0
    for i in range(len(ops)):
        peak = max(peak, sum(regs(r) for o in ops[:i + 1] for r in o.results if last.get(r, -1) >= i))
    return peak + sum(regs(v) for v in outside) + sum(regs(a) for a in carried)


def _levels(loop):
    """The nest from `loop`: its loops, outermost first; the ops before the next loop in each but
    the innermost, in order, which run once a row of the loops around them; the innermost body;
    and the ops after the innermost loop in the loop around it. _KeptApart when a body holds two
    loops, or ops after a loop that holds another."""
    nest, prelude, post = [loop], [], []
    while True:
        ops = _body_ops(nest[-1])
        at = [k for k, o in enumerate(ops) if o.name == "affine.for"]
        if not at:
            return nest, prelude, ops, post
        if len(at) > 1:
            raise _KeptApart("a body of its loops holds more than one loop")
        if post:
            raise _KeptApart("ops follow a loop of it that holds another")
        prelude += ops[:at[0]]
        post = ops[at[0] + 1:]
        nest.append(ops[at[0]])


def _reduction_of(loop, inner, post):
    """The running reduction of the innermost loop `loop` (`inner` its body, `post` the ops after
    it), or None when it carries nothing; _KeptApart when it carries more than one value, or folds
    it by anything but one _ORDERLESS op that only the yield reads."""
    block = loop.regions[0].blocks[0]
    carried = list(block.arguments)[1:]
    if not carried:
        if post:
            raise _KeptApart("ops follow its innermost loop, which folds nothing")
        return None
    if len(carried) != 1:
        raise _KeptApart(f"its innermost loop carries {len(carried)} values, not one")
    carried = carried[0]
    yld = [o.operation for o in block.operations][-1]
    combined = yld.operands[0]
    comb = producer(combined)
    if (comb is None or not _among(comb, inner) or comb.name not in _ORDERLESS
            or len(comb.operands) != 2 or carried not in list(comb.operands)
            or [op_of(u.owner) for u in carried.uses] != [comb]
            or [op_of(u.owner) for u in combined.uses] != [yld]):
        raise _KeptApart("its innermost loop does not fold by one op whose answer is the same "
                         "in any order")
    result = loop.results[0]
    if not all(_among(op_of(u.owner), post) for u in result.uses):
        raise _KeptApart("what its innermost loop folds is read past the row")
    return _Reduction(carried, combined, loop.opview.inits[0], result, post)


def _match_epilogue(op, loop, g, c, ip):
    """find_epilogue's loop as an _Epilogue, or _KeptApart saying why it stays a loop."""
    from ...passes.lib_indexing_maps import MATMUL_N, matmul_axis
    C = op.operands[2]
    nest, prelude, inner, post = _levels(loop)
    bounds = [loop_bounds(lp) for lp in nest]
    if any(b is None or None in b or b[2] <= 0 for b in bounds):
        raise _KeptApart("its loop bounds are not constants")
    ivs = [lp.regions[0].blocks[0].arguments[0] for lp in nest]
    reduce = _reduction_of(nest[-1], inner, post)
    loads, stores, waits, body = [], [], [], []
    for o in inner:
        if o.name in _LOADS:
            loads.append(o)
        elif o.name in _STORES:
            stores.append(o)
        elif o.name == WaitOp.OPERATION_NAME:
            waits.append(o)
        elif not len(o.regions) and _is_aligned(o):
            body.append(o)
        else:
            raise _KeptApart(f"its body holds {o.name}, not an elementwise op")
    if len(stores) > 1 or not (stores or reduce):
        raise _KeptApart(f"it stores {len(stores)} times, not once")
    reads_c = [x for x in loads if _access_buffer(x) == C]
    if len(reads_c) != 1 or not _one_vector(reads_c[0].results[0].type):
        raise _KeptApart("it does not read C once into a 1-D vector")
    width_cons = ir.VectorType(reads_c[0].results[0].type).shape[0]
    access = _access(reads_c[0])
    D, stored = (_access_buffer(stores[0]), stores[0].operands[0]) if stores else (None, None)
    if stores and (_access(stores[0]) != access or not _one_vector(stored.type)
                   or ir.VectorType(stored.type).shape[0] != width_cons):
        raise _KeptApart("it does not store the element it reads of C")
    row_loads, row_stores = ([o for o in prelude if o.name in _LOADS],
                             [o for o in post if o.name in _STORES])
    for o in prelude + post:
        if o.name in _LOADS + _STORES:
            if (o.name in _STORES) == _among(o, prelude):
                raise _KeptApart(f"a row of it runs {o.name} where it cannot")
        elif o.regions or not (_is_aligned(o) or (o.name in _FOLDS and _among(o, post))):
            raise _KeptApart(f"a row of it runs {o.name}, not an elementwise op")
    # A walk's rows are taken for what it folds along them. One that only maps each row (the
    # normalize after attention's second dot) would run at the pops as well, but its store is then
    # cut into one an N pair (_SplitStore), which TOGSim ran slower: prefill 16x8 S512 54151 ->
    # 64380 cycles, S1024 138993 -> 148437.
    if reduce is None and prelude:
        raise _KeptApart("the loops around its innermost make values for a walk that folds nothing")
    if reduce is not None and not row_stores:
        raise _KeptApart("what its rows fold is stored nowhere")
    outs = ([D] if D is not None else []) + [_access_buffer(x) for x in row_stores]
    ins = [_access_buffer(x) for x in loads if _access_buffer(x) != C]
    buffers = [C] + outs + ins + [_access_buffer(x) for x in row_loads]
    if len({buffer_root(x) for x in buffers}) != len(buffers):
        raise _KeptApart("two of C, its outputs and its other inputs are one buffer")
    _refuse_overlap(op, loop, outs, buffers[1 + len(outs):])
    for x in loads:
        if _access(x) != access or not _one_vector(x.results[0].type) \
                or ir.VectorType(x.results[0].type).shape[0] != width_cons:
            raise _KeptApart("a load does not read the element its store writes")
    if not all(_banked_alike(x, C) for x in ([D] if D is not None else []) + ins):
        raise _KeptApart("a buffer it touches is not banked as C is")
    width = ir.VectorType(c.vec).shape[0]
    if reduce is not None and width != width_cons:
        comb = producer(reduce.combined).name
        if comb in _COUNTED and not _splat_is(reduce.init, _COUNTED[comb]):
            raise _KeptApart(f"its {comb} fold's seed is not the identity, which a row folds once "
                             f"a lane: {width} times at the pops, {width_cons} in the walk")
    strides = bank_layout(C)[0]
    gr, _, M, rows, SS = g.reach
    # Where the walk takes a loop's counter for a value, the pops give it: the in-lane row of C
    # along N, the tile a pop's lanes are of; along M, the pop's first element. C must be the
    # pops' [N in-lane, M] bank for it.
    n_axis = matmul_axis(op, 2, MATMUL_N)
    m, idx = access
    by_rows = (len(strides) == 2 and strides[n_axis] == M and strides[1 - n_axis] == 1
               and (m is None or _is_identity(m)))
    coords = {v: d for d, v in enumerate(idx) if v in ivs} if by_rows else {}
    defined = {r for o in prelude + loads + body + post for r in o.results}
    if reduce is not None:
        defined |= {reduce.carried, reduce.result}
        if not (coords.get(ivs[-1]) == 1 - n_axis
                and all(coords.get(v) == n_axis for v in ivs[:-1])):
            raise _KeptApart("it folds along other than the rows of C the pops give")
    for o in row_loads + row_stores:
        if any(const_int(v) is None and coords.get(v) != n_axis for v in _access(o)[1]):
            raise _KeptApart(f"a row of it runs {o.name} at an index the pops cannot give")
    outside, counted = [], set()
    for o in prelude + body + post:
        if o.name in _LOADS + _STORES:
            counted.update(v for v in _access(o)[1] if v in ivs)
            continue
        if not _narrows(o, width_cons):
            raise _KeptApart(f"{o.name} works on a vector of another width")
        for v in o.operands:
            if v in ivs:
                if v not in coords or (not _among(o, body) and coords[v] != n_axis):
                    raise _KeptApart(f"{o.name} takes a loop counter the pops cannot give")
                counted.add(v)
            elif v not in defined:
                outside.append(v)
    coords = {v: d for v, d in coords.items() if v in counted}
    for o in body:
        for v in list(o.operands) + list(o.results):
            if isinstance(v.type, ir.VectorType) and not (
                    _one_vector(v.type) and ir.VectorType(v.type).shape[0] == width_cons):
                raise _KeptApart(f"{o.name} works on a vector of another width")
    for o in row_loads:
        if _wide(o.results[0].type, width_cons) and width != width_cons:
            raise _KeptApart("a row of it loads a vector as wide as its walk's")
    if stores and stored not in defined:
        raise _KeptApart("what it stores is not computed in it")
    for w in waits:
        if not all(_dominates(v, op.operation) for v in w.operands):
            raise _KeptApart("a wait in it takes a value the pops cannot see")
    cons = _consumer_starts(access, bounds, ivs, strides)
    pops = [n * M + m * rows + i for n in range(gr[0]) for m in range(gr[2])
            for i in range(0, g.rows, g.words)]
    if runs_of(cons, width_cons) is None or runs_of(cons, width_cons) != runs_of(pops, width):
        raise _KeptApart("the elements it reads are not the ones the pops write")
    if ip is None:
        return None
    invariant = {}
    with ip:
        for v in outside + ([reduce.init] if reduce is not None else []):
            _invariant(v, op.operation, width_cons, width, invariant)
        inputs = []
        for x in loads:
            buf = _access_buffer(x)
            if buf == C:
                continue
            elt = ir.MemRefType(buf.type).element_type
            inputs.append((x.results[0], GemmOperand(bank_view(buf, SS),
                                                      ir.VectorType.get([width], elt),
                                                      zero_of(elt))))
        d = None if D is None else GemmOperand(
            bank_view(D, SS), ir.VectorType.get([width], ir.MemRefType(D.type).element_type), None)
    return _Epilogue(loop, reads_c[0].results[0], inputs, body, stored, d, waits, invariant,
                     width, width_cons, coords, n_axis, prelude, reduce, outs)


def _is_identity(m):
    """Is the affine map (None: no map) one that takes each index as its own?"""
    return m is None or (m.n_symbols == 0 and len(m.results) == m.n_dims and all(
        isinstance(e, ir.AffineDimExpr) and e.position == k for k, e in enumerate(m.results)))


def _wide(ty, walk):
    """Is `ty` a vector whose last dimension is the walk's width -- one the pops narrow?"""
    return isinstance(ty, ir.VectorType) and len(ir.VectorType(ty).shape) > 0 \
        and ir.VectorType(ty).shape[-1] == walk


def _narrow(ty, walk, width):
    """`ty` as the pops make it: a vector the walk's `walk` wide `width` wide, the rest as is."""
    if not _wide(ty, walk):
        return ty
    vt = ir.VectorType(ty)
    return ir.VectorType.get(list(vt.shape)[:-1] + [width], vt.element_type)


def _narrows(o, walk):
    """Can `o` run on vectors of another width where the walk runs it on vectors `walk` wide:
    _is_aligned (a vector constant a splat), a shape_cast that keeps the last dimension, a
    multi_reduction over it?"""
    vals = list(o.operands) + list(o.results)
    if not any(_wide(v.type, walk) for v in vals):
        return True
    if o.name == "arith.constant":
        return _splat_of(o.results[0]) is not None
    if _is_aligned(o):
        return True
    if o.name == "vector.shape_cast":
        return all(_wide(v.type, walk) for v in vals)
    if o.name == "vector.multi_reduction":
        dims = list(ir.DenseI64ArrayAttr(o.attributes["reduction_dims"]))
        return len(ir.VectorType(o.operands[0].type).shape) - 1 in dims \
            and not _wide(o.results[0].type, walk)
    return False


def _invariant(v, dot, walk, width, made):
    """`v`, which the walk takes from outside it, entered in `made` as the pops take it, built at
    the insertion point when it has to be: `v` itself when it is defined before the `dot` and no
    vector the pops narrow; else a splat constant at the pops' width, or an op that _narrows of
    _is_aligned's, made again of such values -- one between the dot and the walk the pops cannot
    see. _KeptApart for anything else."""
    if v in made:
        return made[v]
    narrowed = _wide(v.type, walk) and walk != width
    splat = _splat_of(v) if isinstance(v.type, ir.VectorType) else None
    if not narrowed and _dominates(v, dot):
        got = v
    elif splat is not None:
        got = _build_constant(_narrow(v.type, walk, width), splat)
    else:
        p = producer(v)
        if (p is None or len(p.regions) or not _is_aligned(p) or not _narrows(p, walk)
                or any(isinstance(x.type, ir.MemRefType) for x in p.operands)):
            raise _KeptApart(f"it takes a value the pops cannot see"
                             f"{', not a splat' if narrowed else ''}")
        vals = {x: _invariant(x, dot, walk, width, made) for x in p.operands}
        new = _clone(p, vals, walk, width)
        got = new.results[[r for r in p.results].index(v)]
    made[v] = got
    return got


def _clone(o, vals, walk, width):
    """`o` again at the insertion point, on what `vals` gives for its operands (the rest as they
    are), a vector it makes `walk` wide `width` wide; its results entered in `vals`."""
    attrs = {n: o.attributes[n] for n in (getattr(x, 'name', x) for x in o.attributes)}
    if o.name == "arith.constant" and _wide(o.results[0].type, walk):
        attrs["value"] = ir.DenseElementsAttr.get_splat(_narrow(o.results[0].type, walk, width),
                                                        _splat_of(o.results[0]))
    new = ir.Operation.create(o.name, results=[_narrow(r.type, walk, width) for r in o.results],
                              operands=[vals.get(v, v) for v in o.operands], attributes=attrs)
    for old, r in zip(o.results, new.results):
        vals[old] = r
    return new


def _refuse_overlap(op, loop, outs, inputs):
    """_KeptApart unless, from the dot to the loop, nothing but the loop uses a buffer it writes
    -- the pops write them from the first last-K one on, while the dot still reads A and B -- and
    nothing writes an input the loop reads."""
    for D in outs:
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
            and lane_extent(x) == lane_extent(C) and lane_extent(C) is not None)


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
            vals = [eval_affine(e, dims, syms) for e in m.results]
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


def _splat_is(v, value):
    """Is `v` a vector constant of integers that are all `value`?"""
    if not (isinstance(v.type, ir.VectorType)
            and isinstance(ir.VectorType(v.type).element_type, ir.IntegerType)):
        return False
    splat = _splat_of(v)
    return splat is not None and ir.IntegerAttr(splat).value == value


def _build_constant(ty, splat):
    from mlir.dialects import arith
    return arith.ConstantOp(ty, ir.DenseElementsAttr.get_splat(ty, splat)).result


class Row:
    """One row of an epilogue's walk the pops reach: in-lane row `n` of C along N, the tile a pop's
    lanes are of. What the walk's loops around its innermost one make for the row (`vals`), and
    the running value of its reduction (`acc`, None without one), which the pops carry along."""

    def __init__(self, n, vals, acc):
        self.n, self.vals, self.acc = n, vals, acc


def open_row(epi, n):
    """Row `n` of the epilogue `epi`, what its loops make for it built at the insertion point."""
    from ...passes.lib_ir import const_index
    vals = dict(epi.invariant)
    for iv, d in epi.coords.items():
        if d == epi.n_axis:
            vals[iv] = const_index(n)
    for o in epi.prelude:
        _clone(o, vals, epi.walk, epi.width)
    return Row(n, vals, None if epi.reduce is None else epi.invariant[epi.reduce.init])


def close_row(epi, row):
    """What the walk does at the end of `row`, at the insertion point: what it folded made into
    the row's answer and stored."""
    if epi.reduce is None:
        return
    vals = dict(row.vals)
    vals[epi.reduce.result] = row.acc
    for o in epi.reduce.post:
        _clone(o, vals, epi.walk, epi.width)


def build_waits(ip, epi):
    """The epilogue's waits, once, before its first store."""
    with ip:
        for w in epi.waits:
            v = w.opview
            WaitOp(sem=v.sem, amount=v.amount, landing=list(v.landing))


def build_epilogue(epi, total, at, row, column):
    """The epilogue on one pop's sum, of `row`: its other inputs read where the pop's C element
    sits, its ops retyped to the pop's width, the result written there in its own buffer and folded
    into the row's. `column()` is the index along M of the pop's first element, for an op that
    takes it."""
    vals = dict(row.vals)
    vals[epi.loaded] = total
    if epi.reduce is not None:
        vals[epi.reduce.carried] = row.acc
    for iv, d in epi.coords.items():
        if d != epi.n_axis:
            vals[iv] = column()
            break
    for loaded, operand in epi.inputs:
        vals[loaded] = transfer_read(operand.vec, operand.value, at, operand.pad, True)
    for o in epi.body:
        _clone(o, vals, epi.walk, epi.width)
    if epi.d is not None:
        transfer_write(vals[epi.stored], epi.d.value, at, True)
    if epi.reduce is not None:
        row.acc = vals[epi.reduce.combined]


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
