"""torchsim-bufferize: tensor -> memref, and the repair of what that leaves behind.
Above this pass everything is tensors, below it everything is memrefs and lane
banking. Refuses both tensor dialects, and any Triton op that got this far."""

from .lib_bank_layout import build_a_scratchpad_buffer
from .lib_ir import (CONDITION_VALUES, VIEW_OPS, FOR_ARGS, FOR_INITS, attrs_of, funcs,
                     get_condition_forwarded, indexing_maps, is_memref, iterator_types, name_of,
                     op_of, operand_segments, parallel_iterator, reassoc_groups, walk,
                     reads_through_this_operand, writes_through_this_operand)

FORBIDS = ("tensor.*", "bufferization.*", "memref.alloc", "memref.dealloc")

BUFFERIZE_OPTIONS = ("bufferize-function-boundaries=true "
                     "allow-return-allocs-from-loops=true")

BUFFERIZE_PIPELINE = (
    "builtin.module("
    "inline,"
    "one-shot-bufferize{" + BUFFERIZE_OPTIONS + "})"
)

_WIDTH_AGNOSTIC = ("arith.andi", "arith.ori", "arith.xori")

class _Fatal(Exception):
    pass


def run(module, **_opts):
    """Bufferize, then repair what it leaves, then place what that invented.
    THE ORDER IS THE ARGUMENT: every step invents buffers, so each one sees the
    buffers of all the steps above it as well as bufferization's own."""
    from mlir.ir import Location
    from mlir.passmanager import PassManager

    PassManager.parse(BUFFERIZE_PIPELINE, module.context).run(module.operation)

    with Location.unknown():
        for func in funcs(module):
            _rewrite_arms_to_write_one_buffer(func)
    PassManager.parse(
        "builtin.module(canonicalize,cse)").run(module.operation)

    with Location.unknown():
        for func in funcs(module):
            entry = func.operation.regions[0].blocks[0]
            _rewrite_loop_yield_to_carried_buffer(func)
            _rewrite_copies_to_bankable_form(func, entry)
            _rewrite_linalg_map_to_linalg_fill(func)
            _rewrite_i1_buffers_to_i8(func)
        _rewrite_alloc_to_spad_global(module)
    PassManager.parse(
        "builtin.module(func.func(canonicalize,cse))").run(module.operation)
    return module


def _rewrite_arms_to_write_one_buffer(func):
    """Upstream's AddCopiesForConditional: where the arms of an `scf.if` yield different buffers,
    each copies its own into ONE made before it and yields that. It is allocated in the identity
    layout and cast to the result's, which bufferization may have left fully dynamic; the
    readers take the buffer itself, so no vector load below meets a stride it cannot see."""
    from mlir.dialects import memref
    from mlir.ir import InsertionPoint, MemRefType
    for cond in [o for o in walk(func.operation) if o.name == "scf.if"]:
        arms = [r.blocks[0] for r in cond.regions if r.blocks]
        if len(arms) < 2:
            continue
        ylds = [list(a.operations)[-1] for a in arms]
        for k, res in enumerate(cond.results):
            if not is_memref(res.type):
                continue
            if all(y.operands[k] == ylds[0].operands[k] for y in ylds):
                continue
            mt = MemRefType(res.type)
            with InsertionPoint(cond):
                one = memref.AllocOp(MemRefType.get(list(mt.shape), mt.element_type,
                                                    memory_space=mt.memory_space),
                                     [], []).result
                held = one if one.type == res.type else memref.CastOp(res.type, one).result
            for y in ylds:
                with InsertionPoint(y):
                    _build_copy_into(y.operands[k], one)
                y.operands[k] = held
            res.replace_all_uses_with(one)


def _rewrite_loop_yield_to_carried_buffer(func):
    """A loop that yields a buffer it did not take clobbers its own readers from
    trip two on, so where a reader of the old value -- through any view -- follows
    a write of the new one's buffer, or IS it and cannot share, the new is copied."""
    from mlir.ir import InsertionPoint, Operation

    for lp in [o for o in walk(func.operation) if o.name == "scf.for"]:
        body = lp.regions[0].blocks[0]
        yld = list(body.operations)[-1]
        if yld.name != "scf.yield":
            continue
        order = {}
        for i, o in enumerate(walk(lp)):
            order[o] = i
        args = list(body.arguments)[FOR_ARGS:]
        for n, v in enumerate(list(yld.operands)):
            if n >= len(args) or not is_memref(v.type):
                continue
            barg = args[n]
            if v == barg:
                continue
            if not len(list(barg.uses)):
                continue
            init = list(lp.operands)[FOR_INITS + n]
            if _rewrite_the_body_to_write_the_carried_buffer(lp, yld, v, barg,
                                                             init):
                yld.operands[n] = barg
                continue
            root, _reshapes = _get_the_reshapes_between(v)
            first_w = min((order.get(op_of(w.owner), 1 << 30)
                           for w in _get_uses_through_views(root)
                           if writes_through_this_operand(op_of(w.owner), w)),
                          default=None)
            if first_w is None:
                continue
            if not any(order.get(op_of(u.owner), -1) > first_w
                       or (order.get(op_of(u.owner), -1) == first_w
                           and not _can_share_operand_buffer_with_user(
                               op_of(u.owner), u))
                       for u in _get_uses_through_views(barg)):
                continue
            with InsertionPoint(yld):
                Operation.create("memref.copy", results=[],
                                 operands=[v, barg])
            yld.operands[n] = barg


def _rewrite_the_body_to_write_the_carried_buffer(lp, yld, v, barg, init):
    """The body's writes of `v`'s BUFFER pointed at the buffer the loop already
    carries, so it carries ONE from trip one. THE INIT IS OVERWRITTEN ON TRIP
    ONE, so the body must only WRITE it and nothing outside may read either."""
    root, reshapes = _get_the_reshapes_between(v)
    if reshapes is None:
        return False
    inside, outside = [], []
    for u in list(root.uses):
        o = op_of(u.owner)
        if o == yld or o in reshapes:
            continue
        (inside if _is_inside(o, lp) else outside).append((o, u))
    if outside or not inside:
        return False
    if any(not writes_through_this_operand(o, u) for o, u in inside):
        return False
    if any(op_of(u.owner) != lp and not _is_inside(op_of(u.owner), lp)
           and not writes_through_this_operand(op_of(u.owner), u)
           for u in init.uses):
        return False
    alias = _build_the_reshapes_that_invert(reshapes, barg, lp)
    if alias is None:
        return False
    for o, u in inside:
        o.operands[u.operand_number] = alias
    return True


def _get_the_reshapes_between(v):
    """(the buffer `v` is a reshaped view of, the reshapes between it and `v`),
    or (v, None) when something that is not a reshape stands in the way. ONLY
    THESE TWO INVERT EXACTLY: each states one reassociation and undoes the other."""
    chain = []
    while True:
        o = op_of(getattr(v, "owner", None))
        if name_of(o) not in ("memref.expand_shape", "memref.collapse_shape"):
            return v, chain
        if reassoc_groups(o) is None:
            return v, None
        chain.insert(0, o)
        v = o.operands[0]


def _is_inside(op, host):
    """Does `op` sit anywhere inside `host`'s regions?"""
    p = op.parent
    while p is not None:
        if p == host:
            return True
        p = p.parent
    return False


def _build_the_reshapes_that_invert(reshapes, barg, lp):
    """`barg` re-viewed as the buffer those reshapes were taken of, by undoing
    each. EXPAND UNDOES COLLAPSE AND BACK: same reassociation, and the result
    type is the one the original reshape was handed. None when a size is dynamic."""
    from mlir.ir import (DenseI64ArrayAttr, InsertionPoint, Operation,
                         ShapedType)
    shapes = [list(ShapedType(o.operands[0].type).shape) for o in reshapes]
    if any(d == ShapedType.get_dynamic_size() for shape in shapes for d in shape):
        return None
    out = barg
    with InsertionPoint.at_block_begin(lp.regions[0].blocks[0]):
        for o in reversed(reshapes):
            was = o.operands[0].type
            attrs = {"reassociation": o.attributes["reassociation"]}
            if name_of(o) == "memref.collapse_shape":
                attrs["static_output_shape"] = DenseI64ArrayAttr.get(
                    list(ShapedType(was).shape))
                name = "memref.expand_shape"
            else:
                name = "memref.collapse_shape"
            out = Operation.create(name, results=[was], operands=[out],
                                   attributes=attrs).results[0]
    return out


def _get_uses_through_views(v):
    """Every use of `v`'s buffer: its own, and those of every view made of it,
    the view ops themselves left out -- a view reads and writes nothing."""
    out, todo = [], [v]
    while todo:
        x = todo.pop()
        for u in x.uses:
            o = op_of(u.owner)
            if o.name in VIEW_OPS:
                todo.extend(o.results)
            else:
                out.append(u)
    return out


def _can_share_operand_buffer_with_user(op, use):
    """Upstream's CanShareOperandBufferWithUser, at its last line: an op may read
    and write one buffer only where it is ELEMENTWISE on that operand -- a generic
    whose map for it is its output's. A transpose, a crossing, a matmul may not."""
    if name_of(op) != "linalg.generic":
        return False
    maps, iters = indexing_maps(op), iterator_types(op)
    n_in = operand_segments(op)[0]
    k = use.operand_number
    return (maps is not None and iters is not None
            and all("parallel" in t for t in iters)
            and all(m == maps[k] for m in maps[n_in:]))


def _rewrite_copies_to_bankable_form(func, entry):
    """`memref.copy` walks a banked buffer contiguously, so it reads lane 0's
    bank as the whole tile. THREE ANSWERS, ONE QUESTION -- does the op that last
    wrote the source READ it? Reads nothing: duplicate. Reads: move. None: 3."""
    from mlir.ir import InsertionPoint

    ops = [o.operation for o in entry.operations]
    for cp, src, dst in _get_copies(func):
        ci = _get_index(ops, cp)
        if ci is None:
            _rewrite_copy_to_linalg_generic(cp, src, dst)
            continue
        writer, wuse = _get_last_writer(src, ci, ops)
        if writer is None:
            _rewrite_copy_to_linalg_generic(cp, src, dst)
            continue
        if reads_through_this_operand(writer, wuse):
            _rewrite_copy_to_linalg_generic(cp, src, dst)
            continue
        if _is_repeatable(writer):
            clone = writer.clone(InsertionPoint(cp))
            clone.operands[wuse.operand_number] = dst
            cp.erase()
            continue
        alloc = op_of(getattr(dst, "owner", None))
        if (name_of(alloc) != "memref.alloc" or _get_index(ops, alloc) is None
                or _is_read_after(src, ci, ops)
                or any(op_of(u.owner) != cp
                       and writes_through_this_operand(op_of(u.owner), u)
                       for u in dst.uses)):
            _rewrite_copy_to_linalg_generic(cp, src, dst)
            continue
        alloc.move_before(writer)
        writer.operands[wuse.operand_number] = dst
        cp.erase()


def _get_copies(func):
    """Every copy in `func`, as (op, source, destination). TWO SPELLINGS AND ONE
    THING, so `_get_identity_copy` asks a candidate's BODY and not its name."""
    out = []
    for o in walk(func.operation):
        if o.name == "memref.copy":
            out.append((o, o.operands[0], o.operands[1]))
        else:
            pair = _get_identity_copy(o)
            if pair is not None:
                out.append((o, pair[0], pair[1]))
    return out


def _get_identity_copy(o):
    """(source, destination) when `o` is an elementwise copy and nothing else: one in, one out,
    the same map on both, and a body that yields what it was handed."""
    if o.name != "linalg.generic":
        return None
    try:
        n_in, n_out = operand_segments(o)
    except (KeyError, ValueError):
        return None
    if n_in != 1 or n_out != 1:
        return None
    blk = o.regions[0].blocks[0]
    body = list(blk.operations)
    if len(body) != 1 or body[0].name != "linalg.yield":
        return None
    if body[0].operands[0] != blk.arguments[0]:
        return None
    maps = indexing_maps(o)
    if maps is None or maps[0] != maps[1]:
        return None
    src, dst = o.operands[0], o.operands[1]
    return None if src == dst else (src, dst)


def _get_index(ops, target):
    """Where `target` sits among a block's `ops`, or None."""
    for i, o in enumerate(ops):
        if o == target:
            return i
    return None


def _get_top_index(ops, target):
    """Where `target` sits among a block's `ops`, followed out of any region to its ancestor."""
    while target is not None:
        i = _get_index(ops, target)
        if i is not None:
            return i
        target = target.parent
    return None


def _rewrite_copy_to_linalg_generic(cp, src, dst):
    """The third answer: a copy no writer explains, said as the identity
    `linalg.generic` the axis relation can read. NO MAP MEANS NO AXIS
    RELATION, which is the half `memref.copy` cannot state."""
    from mlir.ir import InsertionPoint
    if cp.name != "memref.copy":
        return
    roots = (_get_storage_root(src), _get_storage_root(dst))
    if roots == ("func.arg", "func.arg"):
        return
    if roots != ("memref.alloc", "memref.alloc"):
        raise _Fatal(
            f"memref.copy between storage this pass cannot place "
            f"({roots[0]} and {roots[1]}): it is neither DRAM to DRAM nor "
            f"between two allocs this pass will bank, so whether it walks a "
            f"lane-banked buffer contiguously cannot be decided: {cp}")
    with InsertionPoint(cp):
        _build_copy_into(src, dst)
    cp.erase()


def _get_storage_root(v, _hops=0):
    """Where this value's storage comes from -- "memref.alloc", "func.arg", or a
    name saying where the walk stopped, which the copy repair refuses. Every value
    that can flow in is rooted, and they must agree."""
    if _hops > 64:
        return "a view chain over 64 hops deep"
    sources = _get_storage_sources(v)
    if isinstance(sources, str):
        return sources
    roots = sorted({_get_storage_root(s, _hops + 1) for s in sources})
    return roots[0] if len(roots) == 1 else " and ".join(roots)


def _get_storage_sources(v):
    """The values `v`'s storage can come from, or the name of where it stops:
    what a region hands out is what was handed in -- an init, what scf.condition
    forwards, each arm's yield. Hops go outward only, so there are no cycles."""
    owner = getattr(v, "owner", None)
    op = getattr(owner, "operation", None)
    if op is None:
        host = getattr(getattr(owner, "owner", None), "operation", None)
        if host is None:
            return "unowned block"
        if host.name == "func.func":
            return "func.arg"
        if host.name == "scf.for":
            if v.arg_number == 0:
                return "scf.for induction variable"
            return [host.operands[v.arg_number + FOR_INITS - FOR_ARGS]]
        if host.name == "scf.while":
            if _get_terminator(owner).name == "scf.condition":
                return [host.operands[v.arg_number]]
            return [get_condition_forwarded(host)[v.arg_number]]
        return f"{host.name} block argument"
    if op.name in VIEW_OPS:
        return [op.operands[0]]
    if op.name == "scf.for":
        return [op.operands[v.result_number + FOR_INITS]]
    if op.name == "scf.while":
        return [get_condition_forwarded(op)[v.result_number]]
    if op.name == "scf.if":
        return [_get_terminator(r.blocks[0]).operands[v.result_number]
                for r in op.regions]
    return op.name


def _get_terminator(block):
    """The last op of a block, which is its terminator."""
    ops = block.operations
    return ops[len(ops) - 1].operation


def _get_last_writer(src, ci, ops):
    """(the op that wrote `src` last before position `ci`, the operand it wrote
    through), or (None, None). NOTHING MAY TOUCH IT IN BETWEEN and every use has
    to sit in this block -- a use this cannot place is a decline, not a pass."""
    wi = writer = wuse = None
    for u in src.uses:
        o = op_of(u.owner)
        k = _get_top_index(ops, o)
        if k is None:
            return None, None
        if k < ci and writes_through_this_operand(o, u) and (wi is None or k > wi):
            wi, writer, wuse = k, o, u
    if writer is None or _get_index(ops, writer) is None:
        return None, None
    if any(u.operand_number != wuse.operand_number
           for u in src.uses if op_of(u.owner) == writer):
        return None, None
    if any(wi < _get_top_index(ops, op_of(u.owner)) < ci for u in src.uses):
        return None, None
    return writer, wuse


def _is_repeatable(o):
    """Does `o` produce its destination out of nothing? A `linalg.fill` takes a
    scalar and no tile, so running it twice writes the same thing twice -- which
    is what makes DUPLICATING it sound where moving a real write is not."""
    try:
        return operand_segments(o)[0] == 0 or all(
            not is_memref(v) for v in list(o.operands)[:operand_segments(o)[0]])
    except KeyError:
        return False


def _is_read_after(src, ci, ops):
    """Does anything READ `src` after position `ci`? A writer that overwrites it
    whole does not count -- what it held is gone either way, which is what lets
    the write below be MOVED rather than copied."""
    after = sorted(((t, op_of(u.owner), u) for u in src.uses
                    for t in [_get_top_index(ops, op_of(u.owner))]
                    if t is not None and t > ci), key=lambda r: r[0])
    if not after:
        return False
    _t, first, use = after[0]
    if not writes_through_this_operand(first, use):
        return True
    return any(u.operand_number != use.operand_number
               for u in src.uses if op_of(u.owner) == first)


def _rewrite_linalg_map_to_linalg_fill(func):
    """A `linalg.map` with no inputs -- what tensor.splat bufferizes into -- is
    a fill written the long way round, and nothing below names linalg.map: it
    reached convert-linalg-to-loops and filled lane 0's bank alone."""
    from mlir.ir import DenseI32ArrayAttr, InsertionPoint, Operation
    from mlir.dialects import linalg

    for op in [o for o in walk(func.operation) if o.name == "linalg.map"]:
        if len(op.operands) != 1:
            continue
        blk = op.regions[0].blocks[0]
        yld = blk.operations[len(blk.operations) - 1]
        if yld.name != "linalg.yield" or len(yld.operands) != 1:
            continue
        val = yld.operands[0]
        if any(val == a for a in blk.arguments):
            continue
        dst = op.operands[0]
        with InsertionPoint(op):
            fill = Operation.create(
                "linalg.fill", results=[], operands=[val, dst],
                attributes={"operandSegmentSizes":
                            DenseI32ArrayAttr.get([1, 1])},
                regions=1)
            b = fill.regions[0].blocks.append(val.type, val.type)
            with InsertionPoint(b):
                linalg.YieldOp([b.arguments[0]])
        op.erase()


def _rewrite_i1_buffers_to_i8(func):
    """THE BUFFER ONLY -- the compute stays on i1. LLVM gives i1 a store size of
    one byte while the vector unit moves a mask PACKED, so a vector store would
    write one byte in eight. The four steps below are in order."""
    from mlir.ir import IntegerType

    i1 = IntegerType.get_signless(1)
    i8 = IntegerType.get_signless(8)

    for op in list(walk(func.operation)):
        if op.name == "memref.alloc" and _get_memref_elem(op.results[0]) == i1:
            _apply_i8_elem_type(op.results[0], i8)
    _apply_i8_to_every_holder(func, i1, i8)
    _rewrite_i1_edges_to_i8(func, i1, i8)
    _rewrite_trunc_ext_chain_to_i8(func, i1, i8)


def _get_memref_elem(v):
    """`v`'s element type when it is a memref, else None."""
    from mlir.ir import MemRefType
    try:
        return MemRefType(v.type).element_type
    except (ValueError, TypeError):
        return None


def _apply_i8_elem_type(v, i8):
    """Say this memref's elements in bytes, keeping shape, layout and space.
    IN PLACE: the value keeps its identity, so every op already holding it now
    holds the byte-typed one and nothing has to be rebuilt."""
    from mlir.ir import MemRefType
    ty = MemRefType(v.type)
    v.set_type(MemRefType.get(ty.shape, i8, ty.layout, ty.memory_space))


def _apply_i8_to_every_holder(func, i1, i8):
    """Every value that IS one of these buffers said in bytes too -- a view over
    it, a loop's iter_arg, a loop's result. TO FIXPOINT: a loop can hand one to a
    view that feeds the next loop, and the seed was only on the allocation."""
    moved = True
    while moved:
        moved = False
        for op in walk(func.operation):
            if op.name in VIEW_OPS:
                if (op.operands and op.results
                        and _get_memref_elem(op.operands[0]) == i8
                        and _get_memref_elem(op.results[0]) == i1):
                    _apply_i8_elem_type(op.results[0], i8)
                    moved = True
                continue
            if op.name != "scf.for":
                continue
            block = op.regions[0].blocks[0]
            inits = list(op.operands)[FOR_INITS:]
            args = list(block.arguments)[FOR_ARGS:]
            for k, init in enumerate(inits):
                if _get_memref_elem(init) != i8:
                    continue
                for v in (args[k], op.results[k]):
                    if _get_memref_elem(v) == i1:
                        _apply_i8_elem_type(v, i8)
                        moved = True


def _rewrite_i1_edges_to_i8(func, i1, i8):
    """Every place a byte buffer meets an i1 computation, in one walk: a load, a
    fill, a linalg body. ONLY THE EDGES -- the computation is not touched,
    which is what keeps this to two instructions per mask."""
    from mlir.ir import InsertionPoint, Operation
    from mlir.dialects import arith
    for op in list(walk(func.operation)):
        if (op.name == "memref.load"
                and _get_memref_elem(op.operands[0]) == i8
                and op.results[0].type == i1):
            res = op.results[0]
            readers = [(u.owner, u.operand_number) for u in res.uses]
            res.set_type(i8)
            with InsertionPoint(op.operation):
                narrow = arith.TruncIOp(i1, res)
            narrow.operation.move_after(op.operation)
            for owner, idx in readers:
                owner.operands[idx] = narrow.result
            continue
        if (op.name == "linalg.fill" and len(op.operands) == 2
                and _get_memref_elem(op.operands[1]) == i8
                and op.operands[0].type == i1):
            with InsertionPoint(op), op.location:
                wide = Operation.create("arith.extui", results=[i8],
                                        operands=[op.operands[0]])
            op.operands[0] = wide.results[0]
            blk = op.regions[0].blocks[0]
            if len(blk.arguments) and blk.arguments[0].type == i1:
                blk.arguments[0].set_type(i8)
        if not op.name.startswith("linalg.") or not op.regions:
            continue
        block = op.regions[0].blocks[0]
        yield_op = block.operations[len(block.operations) - 1]
        if yield_op.name != "linalg.yield":
            continue
        n_out = len(yield_op.operands)
        n_in = len(op.operands) - n_out
        if n_in < 0 or len(block.arguments) != n_in + n_out:
            continue
        for i in range(min(len(block.arguments), n_in + n_out)):
            arg = block.arguments[i]
            if _get_memref_elem(op.operands[i]) != i8 or arg.type != i1:
                continue
            arg.set_type(i8)
            with InsertionPoint.at_block_begin(block), op.location:
                narrow = Operation.create("arith.trunci", results=[i1],
                                          operands=[arg])
            arg.replace_all_uses_with(narrow.results[0])
            narrow.operands[0] = arg
        for j in range(len(yield_op.operands)):
            if (_get_memref_elem(op.operands[n_in + j]) == i8
                    and yield_op.operands[j].type == i1):
                with InsertionPoint(yield_op), yield_op.location:
                    wide = Operation.create("arith.extui", results=[i8],
                                            operands=[yield_op.operands[j]])
                yield_op.operands[j] = wide.results[0]


def _rewrite_trunc_ext_chain_to_i8(func, i1, i8):
    """`extui(bitwise(trunci a, trunci b))` -> `bitwise(a, b)`. NOT TIDYING BUT AN
    IDENTITY: and/or/xor on a zero-extended value mean the same thing at any
    width, which is the whole of why the narrowing in between can go."""
    from mlir.ir import InsertionPoint, Operation
    for op in list(walk(func.operation)):
        if not op.name.startswith("linalg.") or not op.regions:
            continue
        block = op.regions[0].blocks[0]
        for ext in [o for o in list(block.operations)
                    if name_of(o) == "arith.extui"]:
            if ext.results[0].type != i8 or ext.operands[0].type != i1:
                continue
            comb = op_of(getattr(ext.operands[0], "owner", None))
            if comb is None or name_of(comb) not in _WIDTH_AGNOSTIC:
                continue
            if len(list(comb.results[0].uses)) != 1:
                continue
            srcs = [_get_truncated_source(x, i1, i8) for x in comb.operands]
            if any(s is None for s in srcs):
                continue
            with InsertionPoint(comb), comb.location:
                wide = Operation.create(name_of(comb), results=[i8],
                                        operands=srcs)
            ext.results[0].replace_all_uses_with(wide.results[0])
            ext.erase()
            dead = [comb] + [op_of(x.owner) for x in comb.operands]
            comb.erase()
            for d in dead[1:]:
                if not len(list(d.results[0].uses)):
                    d.erase()


def _get_truncated_source(v, i1, i8):
    """The i8 value `v` is a plain `arith.trunci` of, or None."""
    o = op_of(getattr(v, "owner", None))
    if o is None or name_of(o) != "arith.trunci":
        return None
    src = o.operands[0]
    return src if src.type == i8 and v.type == i1 else None


def _rewrite_alloc_to_spad_global(module):
    """An alloc names no LEVEL of this machine's memory, and the default is the
    heap -- where the DMA does not write and every lane reads the same bytes. So
    becoming memory and becoming a BANK is ONE step, not two that could differ."""

    module_top = module.operation.regions[0].blocks[0]
    counter = 0
    for func in funcs(module):
        entry = func.operation.regions[0].blocks[0]
        for op in _get_unplaced_allocs(func):
            counter = _rewrite_one_alloc_to_one_global(
                op, module_top, entry, counter)
        _drop_deallocs(func)


def _get_unplaced_allocs(func):
    """Every `memref.alloc` this pass still owes a placement. A memory space is
    already a placement, so one that has it is somebody else's."""
    from mlir.ir import MemRefType
    return [o for o in walk(func.operation) if o.name == "memref.alloc"
            and MemRefType(o.results[0].type).memory_space is None]


def _rewrite_one_alloc_to_one_global(op, module_top, entry, counter):
    """One `memref.alloc` -> a scratchpad global named for its lanes (the leading extent), and the
    alloc goes; returns the next counter. p11 checks the name against the type."""
    from mlir.ir import MemRefType
    mt = MemRefType(op.results[0].type)
    if len(op.operands):
        raise _Fatal(f"dynamically sized alloc cannot be scratchpad: {op}")
    lanes = mt.shape[0] if len(mt.shape) else 1
    g = build_a_scratchpad_buffer(module_top, entry, mt.shape, mt.element_type,
                    f"buf{counter}_spad_{lanes}lane")
    op.results[0].replace_all_uses_with(g)
    _apply_spad_space_to_aliases(g, MemRefType(g.type).memory_space)
    op.erase()
    return counter + 1




def _apply_spad_space_to_aliases(value, space_attr):
    """Carry the scratchpad space to everything aliasing `value`: a region hands
    out what was handed into it (retyped in place), a view op is rebuilt around
    the new type, and anything else is refused rather than silently re-spaced."""
    from mlir.ir import InsertionPoint, MemRefType, Operation
    for use in list(value.uses):
        owner = op_of(use.owner)
        carried = _get_region_carried_values(owner, use)
        if carried:
            for v in carried:
                mt = MemRefType(v.type)
                v.set_type(MemRefType.get(list(mt.shape), mt.element_type,
                                          mt.layout, space_attr))
                _apply_spad_space_to_aliases(v, space_attr)
            continue
        mem_results = [r for r in owner.results if is_memref(r.type)]
        if not mem_results:
            continue
        if owner.name == "memref.memory_space_cast":
            continue
        if owner.name not in VIEW_OPS:
            _refuse_not_a_view(owner)
        new_results = [
            MemRefType.get(list(MemRefType(r.type).shape),
                           MemRefType(r.type).element_type,
                           MemRefType(r.type).layout, space_attr)
            if is_memref(r.type) else r.type
            for r in owner.results]
        attrs = attrs_of(owner)
        with InsertionPoint(owner):
            new_op = Operation.create(
                owner.name, results=new_results,
                operands=list(owner.operands), attributes=attrs)
        for old_r, new_r in zip(owner.results, new_op.results):
            old_r.replace_all_uses_with(new_r)
            _apply_spad_space_to_aliases(new_r, space_attr)
        owner.erase()


def _get_region_carried_values(owner, use):
    """What a region hands out for the value handed in at `use` -- a loop's block
    argument and result, a condition's forwarded arguments and results. Empty
    when `owner` is not a region carrying a value through unchanged."""
    if owner.name == "scf.for":
        n = use.operand_number - FOR_INITS
        if n < 0:
            return []
        return [owner.regions[0].blocks[0].arguments[n + FOR_ARGS],
                owner.results[n]]
    if owner.name == "scf.while":
        return [owner.regions[0].blocks[0].arguments[use.operand_number]]
    if owner.name == "scf.condition":
        k = use.operand_number - CONDITION_VALUES
        if k < 0:
            return []
        while_op = owner.parent
        return [while_op.regions[1].blocks[0].arguments[k], while_op.results[k]]
    return []


def _refuse_not_a_view(owner):
    """A memref derived from a scratchpad buffer by an op that is not a view keeping
    axis 0 as the lane axis: a transpose moves it, a byte view has none."""
    raise _Fatal(
        f"op '{owner.name}' produces a memref from a scratchpad "
        f"buffer but is not a view that keeps axis 0 on the lanes, so "
        f"the buffer's layout says nothing about it: {owner}")


def _drop_deallocs(func):
    """Scratchpad globals are not freed, so nothing may try."""
    for op in [o for o in walk(func.operation) if o.name == "memref.dealloc"]:
        op.erase()


def _build_copy_into(src, dst):
    """`src` copied into `dst` by an identity generic -- the spelling
    bufferization gives materialize_in_destination, which the repair below reads."""
    from mlir.dialects import linalg
    from mlir.ir import (AffineMap, AffineMapAttr, ArrayAttr, DenseI32ArrayAttr,
                         InsertionPoint, MemRefType, Operation)
    rank = len(MemRefType(src.type).shape)
    ident = AffineMapAttr.get(AffineMap.get_identity(rank))
    gen = Operation.create(
        "linalg.generic", results=[], operands=[src, dst],
        attributes={"indexing_maps": ArrayAttr.get([ident, ident]),
                    "iterator_types": ArrayAttr.get([parallel_iterator()] * rank),
                    "operandSegmentSizes": DenseI32ArrayAttr.get([1, 1])},
        regions=1)
    elt = MemRefType(src.type).element_type
    blk = gen.regions[0].blocks.append(elt, MemRefType(dst.type).element_type)
    with InsertionPoint(blk):
        linalg.YieldOp([blk.arguments[0]])
    return gen
