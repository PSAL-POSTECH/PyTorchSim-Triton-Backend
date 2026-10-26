"""pytorchsim-bufferize: tensor -> memref, and the repair of what that leaves behind. First an all-parallel
linalg op whose body never reads its init takes as init an input of the same type read through the same map,
so bufferize may write where it reads (one-shot-bufferize's conflict analysis decides whether it does). Last,
while dots and reductions still stand, the buffers only they read are stamped PREFETCH_OPERAND for
pytorchsim-plan-copy-node (mark_prefetch_operands); that changes no op."""

from ...dialects.pytorchsim import effects_on
from .lib_bank_layout import SPAD_SPACE, in_scratchpad
from .lib_ir import (CONDITION_VALUES, DOT_OP, FOR_ARGS, FOR_INITS, funcs,
                     get_condition_forwarded, indexing_maps, is_memref, is_view, iterator_types, view_source,
                     name_of, op_of, operand_segments, parallel_iterator, reassoc_groups, unread_destinations,
                     walk, reads_through_this_operand, writes_through_this_operand)

FORBIDS = ("tensor.*", "bufferization.*")

BUFFERIZE_OPTIONS = ("bufferize-function-boundaries=true "
                     "allow-return-allocs-from-loops=true use-encoding-for-memory-space=true")

BUFFERIZE_PIPELINE = (
    "builtin.module("
    "inline,"
    "one-shot-bufferize{" + BUFFERIZE_OPTIONS + "})"
)

DEALLOC_PIPELINE = (
    "builtin.module("
    "ownership-based-buffer-deallocation,canonicalize,buffer-deallocation-simplification,"
    "bufferization-lower-deallocations,cse,canonicalize)"
)

_WIDTH_AGNOSTIC = ("arith.andi", "arith.ori", "arith.xori")

class _Fatal(Exception):
    pass


def run(module, **_opts):
    """Reuse unread inits as inputs, bufferize, repair what it leaves, make every buffer a scratchpad alloc
    at its function's entry, then let the upstream dealloc pipeline free each, and refuse a life that is not
    that static span."""
    from mlir.ir import Location
    from mlir.passmanager import PassManager
    reuse_unread_inits(module)

    # No cse runs between this and one-shot-bufferize: it would make the generators one again.
    with Location.unknown():
        for func in funcs(module):
            _rewrite_generators_per_write(func)
    _apply_spad_encoding(module)
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
            _rewrite_i1_buffers_to_i8(func)
        _place_allocs_at_entry(module)
    PassManager.parse(
        "builtin.module(func.func(canonicalize,cse))").run(module.operation)
    symbols = _get_symbols(module)
    PassManager.parse(DEALLOC_PIPELINE, module.context).run(module.operation)
    _check_static_lifetimes(module, symbols)
    mark_prefetch_operands(module)
    return module


def reuse_unread_inits(module):
    """Every op that states its maps and iterators is read; none is recognised by name. No candidate, no
    change."""
    for func in funcs(module):
        for op in walk(func.operation):
            _rewrite_unread_inits_to_inputs(op)


def _rewrite_unread_inits_to_inputs(op):
    maps, iters = indexing_maps(op), iterator_types(op)
    if maps is None or iters is None or "operandSegmentSizes" not in op.attributes:
        return
    if any(t != parallel_iterator() for t in iters):
        return
    n_in = operand_segments(op)[0]
    if len(maps) != len(op.operands) or len(op.results) != len(op.operands) - n_in:
        return
    for slot in sorted(unread_destinations(op)):
        found = _get_candidates(op, slot, n_in, maps)
        if found:
            op.operands[slot] = found[0]


def _get_candidates(op, slot, n_in, maps):
    """The inputs that can stand as the init at `slot`, best first: the one the block hands back,
    then one nothing else uses, then operand order. An input read after `op` is not one."""
    out = op.operands[slot]
    inits = [op.operands[k] for k in range(n_in, len(op.operands))]
    ranked = []
    for k in range(n_in):
        v = op.operands[k]
        if v.type != out.type or maps[k] != maps[slot] or v in inits:
            continue
        others = [op_of(u.owner) for u in v.uses if op_of(u.owner) != op]
        if not _is_last_read_here(v, op, others):
            continue
        ranked.append((not _is_handed_back(v, op.results[slot - n_in]), bool(others), k))
    return [op.operands[k] for _, _, k in sorted(ranked)]


def _is_last_read_here(v, op, others):
    """`v` is made in `op`'s own block and every other user sits before `op` in it: the value is
    dead once `op` has read it, on this trip and on the next."""
    from mlir.ir import BlockArgument, InsertionPoint
    block = InsertionPoint(op).block
    made_in = v.owner if isinstance(v, BlockArgument) else InsertionPoint(op_of(v.owner)).block
    if made_in != block:
        return False
    before = []
    for o in block.operations:
        if op_of(o) == op:
            break
        before.append(op_of(o))
    for user in others:
        while InsertionPoint(user).block != block:
            user = op_of(user.parent)
        if user not in before:
            return False
    return True


def _is_handed_back(v, result):
    """`v` is an argument of the block whose terminator takes `result`: a loop's carried value and
    the value the trip yields for it."""
    from mlir.ir import BlockArgument
    if not isinstance(v, BlockArgument):
        return False
    last = list(v.owner.operations)[-1].operation
    return any(o == result for o in last.operands)


def _rewrite_generators_per_write(func):
    """A generator -- a linalg op that reads no tensor and none of its outs and computes nothing per
    element, as a fill: it only yields scalars -- made again (with the empty tensors it writes) before
    an op that writes it in place -- a linalg op's outs, a dot's accumulator, a loop's inits -- when
    another op takes it too, or a loop between them runs the writer again. cse makes equal
    generators one value (attention's row maxes take one -inf, its dots one zero), and a loop's
    zero accumulator is hoisted out of it; one-shot-bufferize then keeps one buffer for it and gives
    every other writer, and every trip, a copy of it, which vectorize and the matmul lowering read
    from memory instead of taking the constant it is. An empty tensor alone is not made again: what
    it holds is undefined, so a writer of it is given a buffer and no copy."""
    from mlir.ir import InsertionPoint
    for g in [o for o in walk(func.operation) if _is_generator(o)]:
        for res in g.results:
            uses = list(res.uses)
            writes = [(op_of(u.owner), u.operand_number) for u in uses
                      if _writes_in_place(op_of(u.owner), u)]
            again = [(o, n) for o, n in writes if _runs_again(o, g)]
            # with every use a writer and none run again, one keeps `g`
            if len(writes) == len(uses) and not again:
                writes = writes[1:]
            elif len(uses) < 2:
                writes = again
            for owner, n in writes:
                owner.operands[n] = _clone_generator(
                    g, InsertionPoint(owner)).results[res.result_number]
        if all(not list(r.uses) for r in g.results):
            g.erase()


def _is_generator(op):
    nm = name_of(op) or ""
    if not nm.startswith("linalg.") or not op.results:
        return False
    n_out = len(op.results)
    n_in = len(op.operands) - n_out
    if any(_is_shaped(v.type) for v in list(op.operands)[:n_in]):
        return False
    if unread_destinations(op) != set(range(n_in, len(op.operands))):
        return False
    body = op.regions[0].blocks[0]
    ops = [o.operation for o in body.operations]
    return len(ops) == 1 and all(_is_scalar_given(v, body, n_in) for v in ops[0].operands)


def _is_scalar_given(v, body, n_in):
    """`v` is one of the scalars the op takes, or a value from outside its body: the same at every
    element."""
    from mlir.ir import BlockArgument
    if isinstance(v, BlockArgument):
        return v.owner == body and v.arg_number < n_in
    return op_of(v.owner).block != body


def _is_shaped(ty):
    from mlir.ir import ShapedType
    return isinstance(ty, ShapedType)


_LOOPS = ("scf.for", "scf.while", "scf.parallel", "affine.for", "affine.parallel")


def _runs_again(o, g):
    """A loop holds `o` and not `g`: `o` runs once a trip, `g` once."""
    top = op_of(op_of(g).parent)
    p = op_of(o.parent)
    while p is not None and p != top:
        if name_of(p) in _LOOPS:
            return True
        p = op_of(p.parent) if p.parent is not None else None
    return False


def _writes_in_place(o, use):
    nm = name_of(o) or ""
    if nm == "scf.for":
        return use.operand_number >= FOR_INITS
    if nm == "scf.while":
        return True
    if nm.startswith("linalg.") or nm == DOT_OP:
        return writes_through_this_operand(o, use)
    return False


def _clone_generator(g, ip):
    """`g` made again at `ip`, each of its outs that is an empty tensor made again with it."""
    from mlir.ir import InsertionPoint
    new = g.clone(ip=ip)
    n_in = len(new.operands) - len(new.results)
    for k in range(n_in, len(new.operands)):
        e = op_of(getattr(new.operands[k], "owner", None))
        if name_of(e) == "tensor.empty":
            new.operands[k] = e.clone(ip=InsertionPoint(new)).results[0]
    return new


def _rewrite_arms_to_write_one_buffer(func):
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
            _apply_identity_layout_to_reshapes(one)


def _apply_identity_layout_to_reshapes(v):
    """A reshape of a buffer that now has the identity layout has it too: its type was inferred from
    the arms' result, whose layout was any."""
    from mlir.ir import MemRefType
    for u in list(v.uses):
        o = u.owner
        if o.name not in ("memref.expand_shape", "memref.collapse_shape"):
            continue
        r = o.results[0]
        mt = MemRefType(r.type)
        r.set_type(MemRefType.get(list(mt.shape), mt.element_type, memory_space=mt.memory_space))
        _apply_identity_layout_to_reshapes(r)


def _rewrite_loop_yield_to_carried_buffer(func):
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
    p = op.parent
    while p is not None:
        if p == host:
            return True
        p = p.parent
    return False


def _build_the_reshapes_that_invert(reshapes, barg, lp):
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
    out, todo = [], [v]
    while todo:
        x = todo.pop()
        for u in x.uses:
            o = op_of(u.owner)
            if is_view(o):
                todo.extend(o.results)
            else:
                out.append(u)
    return out


def _can_share_operand_buffer_with_user(op, use):
    if name_of(op) != "linalg.generic":
        return False
    maps, iters = indexing_maps(op), iterator_types(op)
    n_in = operand_segments(op)[0]
    k = use.operand_number
    return (maps is not None and iters is not None
            and all(t == parallel_iterator() for t in iters)
            and all(m == maps[k] for m in maps[n_in:]))


def _rewrite_copies_to_bankable_form(func, entry):
    """Each copy, in order, forwarded into its source's last writer or written as a linalg.generic; the
    entry's ops read again for each, so a writer an earlier copy's rewrite made is found."""
    from mlir.ir import InsertionPoint

    for cp, src, dst in _get_copies(func):
        ops = [o.operation for o in entry.operations]
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
    for i, o in enumerate(ops):
        if o == target:
            return i
    return None


def _get_top_index(ops, target):
    while target is not None:
        i = _get_index(ops, target)
        if i is not None:
            return i
        target = target.parent
    return None


def _rewrite_copy_to_linalg_generic(cp, src, dst):
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
    if _hops > 64:
        return "a view chain over 64 hops deep"
    sources = _get_storage_sources(v)
    if isinstance(sources, str):
        return sources
    roots = sorted({_get_storage_root(s, _hops + 1) for s in sources})
    return roots[0] if len(roots) == 1 else " and ".join(roots)


def _get_storage_sources(v):
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
    if is_view(op):
        return [view_source(op)]
    if op.name == "scf.for":
        return [op.operands[v.result_number + FOR_INITS]]
    if op.name == "scf.while":
        return [get_condition_forwarded(op)[v.result_number]]
    if op.name == "scf.if":
        return [_get_terminator(r.blocks[0]).operands[v.result_number]
                for r in op.regions]
    return op.name


def _get_terminator(block):
    ops = block.operations
    return ops[len(ops) - 1].operation


def _get_last_writer(src, ci, ops):
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
    try:
        return operand_segments(o)[0] == 0 or all(
            not is_memref(v) for v in list(o.operands)[:operand_segments(o)[0]])
    except KeyError:
        return False


def _is_read_after(src, ci, ops):
    after = sorted(((t, op_of(u.owner), u) for u in src.uses
                    for t in [_get_top_index(ops, op_of(u.owner))]
                    if t is not None and t > ci), key=lambda r: r[0])
    if not after:
        return False
    _t, first, use = after[0]
    if reads_through_this_operand(first, use):
        return True
    return any(u.operand_number != use.operand_number
               for u in src.uses if op_of(u.owner) == first)


def _rewrite_i1_buffers_to_i8(func):
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
    from mlir.ir import MemRefType
    try:
        return MemRefType(v.type).element_type
    except (ValueError, TypeError):
        return None


def _apply_i8_elem_type(v, i8):
    from mlir.ir import MemRefType
    ty = MemRefType(v.type)
    v.set_type(MemRefType.get(ty.shape, i8, ty.layout, ty.memory_space))


def _apply_i8_to_every_holder(func, i1, i8):
    from mlir.dialects import scf
    moved = True
    while moved:
        moved = False
        for op in walk(func.operation):
            src = view_source(op)
            if src is not None:
                if _get_memref_elem(src) == i8 and _get_memref_elem(op.results[0]) == i1:
                    _apply_i8_elem_type(op.results[0], i8)
                    moved = True
                continue
            if not isinstance(op.opview, scf.ForOp):
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
    o = op_of(getattr(v, "owner", None))
    if o is None or name_of(o) != "arith.trunci":
        return None
    src = o.operands[0]
    return src if src.type == i8 and v.type == i1 else None


def _apply_spad_encoding(module):
    """Every tensor but a constant typed with the scratchpad's space as its encoding, which one-shot-bufferize
    gives each buffer it makes (use-encoding-for-memory-space); a constant stays a global."""
    from mlir.dialects import arith
    from mlir.ir import FunctionType, IntegerAttr, IntegerType, RankedTensorType, TypeAttr
    space = IntegerAttr.get(IntegerType.get_signless(64), SPAD_SPACE)

    def encoded(ty):
        try:
            t = RankedTensorType(ty)
        except ValueError:
            return ty
        return t if t.encoding is not None else RankedTensorType.get(list(t.shape), t.element_type, space)
    for op in walk(module.operation):
        for block in (b for r in op.regions for b in r.blocks):
            for a in block.arguments:
                a.set_type(encoded(a.type))
        if not isinstance(op.opview, arith.ConstantOp):
            for r in op.results:
                r.set_type(encoded(r.type))
    for func in funcs(module):
        ft = FunctionType(TypeAttr(func.operation.attributes["function_type"]).value)
        func.operation.attributes["function_type"] = TypeAttr.get(
            FunctionType.get([encoded(t) for t in ft.inputs], [encoded(t) for t in ft.results]))


def _place_allocs_at_entry(module):
    """Every buffer bufferize allocated at the top of its function's entry block, each one moved ahead of the
    last: the one storage each had as a global, in the order those had; every alias of one a view."""
    for func in funcs(module):
        entry = func.operation.regions[0].blocks[0]
        for op in _get_allocs(func):
            _refuse_a_hoist(op)
            first = entry.operations[0].operation
            if first != op:
                op.move_before(first)
            _check_aliases_are_views(op.results[0])


def _get_allocs(func):
    """Each op of `func` stating it allocates a memref (its effects, not its name)."""
    return [o for o in walk(func.operation)
            if any(is_memref(r) and "allocate" in effects_on(o, r) for r in o.results)]


def _refuse_a_hoist(op):
    """Moving an alloc to the entry is legal when nothing it needs is computed later (no dynamic
    size) and no op frees it yet: a free left in a loop would end the hoisted buffer on trip one."""
    if len(op.operands):
        raise _Fatal(f"dynamically sized alloc cannot be scratchpad: {op}")
    v = op.results[0]
    freed = [u.owner for u in v.uses if "free" in effects_on(op_of(u.owner), v)]
    if freed:
        raise _Fatal(f"an alloc is freed before a06 gives it its life, so moving it to the entry "
                     f"would leave that free inside its old scope: {op_of(freed[0])}")


def _get_symbols(module):
    from mlir.ir import StringAttr
    return {StringAttr(o.operation.attributes["sym_name"]).value
            for o in module.operation.regions[0].blocks[0].operations
            if "sym_name" in o.operation.attributes}


def _check_static_lifetimes(module, symbols):
    """Each scratchpad buffer is one static-shape alloc at its function's entry, freed once, by an
    unconditional dealloc in that same block; the pipeline left no runtime ownership (a helper
    function, a bufferization op) and no buffer outside the scratchpad."""
    from mlir.ir import MemRefType, ShapedType
    made = sorted(_get_symbols(module) - symbols)
    if made:
        raise _Fatal(f"the dealloc pipeline made @{made[0]}: a buffer's owner is decided at run "
                     f"time, and a static scratchpad layout cannot follow it")
    for func in funcs(module):
        entry = func.operation.regions[0].blocks[0]
        frees = {}
        for op in walk(func.operation):
            if op.name.split(".", 1)[0] == "bufferization":
                raise _Fatal(f"the dealloc pipeline left {op.name}, a life it did not decide: {op}")
            for r in op.results:
                if not is_memref(r) or "allocate" not in effects_on(op, r):
                    continue
                mt = MemRefType(r.type)
                if not in_scratchpad(r.type):
                    raise _Fatal(f"an alloc outside the scratchpad (memory space {mt.memory_space}) "
                                 f"survived a06; nothing places it: {op}")
                if any(d == ShapedType.get_dynamic_size() for d in mt.shape):
                    raise _Fatal(f"a scratchpad alloc of dynamic shape has no static slot: {op}")
                if op.operation.block != entry:
                    raise _Fatal(f"a scratchpad alloc stands outside its function's entry block: {op}")
                frees.setdefault(r, [])
            for v in op.operands:
                if not is_memref(v) or "free" not in effects_on(op, v):
                    continue
                if op.operation.block != entry:
                    host = op.operation.parent
                    raise _Fatal(
                        f"a scratchpad buffer is freed inside {host.name}, not where it is allocated: "
                        f"its life would depend on which way the program goes: {op}")
                if v not in frees:
                    raise _Fatal(f"a dealloc frees a value no alloc before it in its block makes, "
                                 f"so which buffer's life ends is not stated: {op}")
                frees[v].append(op)
        for r, ends in frees.items():
            if len(ends) != 1:
                raise _Fatal(f"a scratchpad buffer is freed {len(ends)} times, not once: "
                             f"{op_of(r.owner)}")


def _check_aliases_are_views(value):
    """Every memref an op makes from `value`, through loops and views, is a view of it."""
    from mlir.dialects import memref
    for use in list(value.uses):
        owner = op_of(use.owner)
        carried = _get_region_carried_values(owner, use)
        if carried:
            for v in carried:
                _check_aliases_are_views(v)
            continue
        mem_results = [r for r in owner.results if is_memref(r.type)]
        if not mem_results or isinstance(owner.opview, memref.MemorySpaceCastOp):
            continue
        if not is_view(owner):
            _refuse_not_a_view(owner)
        for r in mem_results:
            _check_aliases_are_views(r)


def _get_region_carried_values(owner, use):
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
    raise _Fatal(
        f"op '{owner.name}' produces a memref from a scratchpad "
        f"buffer but is not a view that keeps axis 0 on the lanes, so "
        f"the buffer's layout says nothing about it: {owner}")


def _build_copy_into(src, dst):
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


# The buffers only dots (as A or B) and reductions (as an input) read: what XLA's cross-program prefetch
# may hold, stamped for pytorchsim-plan-copy-node.

PREFETCH_OPERAND = "pytorchsim.prefetch_operand"
"""An i64 on the op that allocates a buffer only dots (as A or B) and reductions (as an input) read, the
elements their outputs hold together: what XLA's cross-program prefetch may hold, a parameter every use of
which is one of those, and its comparator's cumulative use size."""

PREFETCH_OPERAND_NOT = "pytorchsim.prefetch_operand_not"
"""Why a buffer a dot or a reduction reads is no PREFETCH_OPERAND: the other op that reads it."""


class OperandUnallocated(Exception):
    """A dot or a reduction reads an operand that views no buffer an op of the function allocates."""


def mark_prefetch_operands(module):
    """Stamp PREFETCH_OPERAND with its uses' output elements, or PREFETCH_OPERAND_NOT with the other reader,
    on the allocating op of every dot's A and B and every reduction's inputs; refuse one that has none."""
    from mlir.ir import BlockArgument, IntegerAttr, IntegerType, StringAttr
    from ...dialects.pytorchsim import view_root
    for func in funcs(module):
        ops = list(walk(func.operation))
        roots, uses = {}, {}
        for op in ops:
            for k in _read_operands(op):
                root = view_root(op.operands[k])
                if isinstance(root, BlockArgument):
                    raise OperandUnallocated(f"operand {k} of the {op.name} at {op.location} views a block "
                                             f"argument, no allocated buffer")
                roots.setdefault(root, None)
                uses[root] = uses.get(root, 0) + _output_elements(op)
        for op in ops:
            if is_view(op):
                continue
            mine = set(_read_operands(op))
            for k, v in enumerate(op.operands):
                if k in mine or not in_scratchpad(v.type) or roots.get(view_root(v), "") is not None:
                    continue
                if effects_on(op, v) & {"read", "unknown"}:
                    roots[view_root(v)] = f"{op.name} at {op.location} reads it too"
        for root, other in roots.items():
            if other is None:
                i64 = IntegerType.get_signless(64)
                root.owner.attributes[PREFETCH_OPERAND] = IntegerAttr.get(i64, uses[root])
            else:
                root.owner.attributes[PREFETCH_OPERAND_NOT] = StringAttr.get(other)
    return module


def _output_elements(op):
    """The elements `op` (a dot or a reduction) writes: its C, or its inits."""
    import math
    from mlir.ir import MemRefType
    from ...dialects.pytorchsim import DotOp
    outs = [op.operands[2]] if isinstance(op.opview, DotOp) else list(op.operands)[len(op.operands) // 2:]
    return sum(math.prod(MemRefType(v.type).shape) for v in outs)


def _read_operands(op):
    """The operand numbers of what `op` reads as a matmul operand or a reduction input: a dot's A and B, a
    linalg.reduce's inputs (its first half; the rest are its inits)."""
    from mlir.dialects import linalg
    from ...dialects.pytorchsim import DotOp
    if isinstance(op.opview, DotOp):
        return (0, 1)
    if isinstance(op.opview, linalg.ReduceOp):
        return range(len(op.operands) // 2)
    return ()
