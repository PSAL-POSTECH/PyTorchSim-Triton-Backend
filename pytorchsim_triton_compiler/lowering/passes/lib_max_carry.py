"""A loop that carries a running `arith.maximumf` / `arith.minimumf` (a max or min reduction's
strip loop) carries it as `maxnumf` / `minnumf` plus a mask of the elements that have seen a NaN,
and puts the NaN back once, after the loop.

`maximumf` propagates a NaN; RVV's vfmax does not, so llc spells every trip's maximumf as two
self-compares, two merges and the vfmax -- five instructions and two extra register groups, which
at LMUL 8 is one group more than the register file holds. Carried this way a trip is the vfmax, one
self-compare of the element read and one mask OR; the result is the same in every element: NaN
wherever an element of the init or of any trip was NaN, else the max (vfmax orders -0 below +0,
as maximumf does).

When every carry is such an extremum, its init the identity (-inf for max, +inf for min) and the
bounds constant, the first trip is peeled: the carry starts from what that trip reads. llc would
otherwise hoist the identity's splat out of every enclosing loop and hold it in a register group
across them -- at LMUL 8, the one group that spills."""

_IDENTITY = {"arith.maximumf": float("-inf"), "arith.minimumf": float("inf")}

_NUM_OF = {"arith.maximumf": "arith.maxnumf", "arith.minimumf": "arith.minnumf"}


def carry_nan_apart(module):
    from .lib_ir import walk
    for lp in [o for o in walk(module.operation) if o.name == "affine.for"]:
        found = _carried_extrema(lp)
        if found:
            _rebuild(lp, found)


def _carried_extrema(lp):
    """[(iter_arg index, the extremum op, the operand it reads besides the carry)] of `lp`."""
    body = lp.regions[0].blocks[0]
    if len(body.arguments) < 2:
        return []
    term = list(body.operations)[-1]
    found = []
    for k, (arg, yielded) in enumerate(zip(list(body.arguments)[1:], term.operands)):
        op = yielded.owner
        if (not hasattr(op, "name") or op.name not in _NUM_OF
                or not _is_vector(arg.type) or op.parent != term.parent):
            continue
        opnds = list(op.operands)
        if arg not in opnds or len([u for u in arg.uses]) != 1:
            continue
        other = opnds[1] if opnds[0] == arg else opnds[0]
        found.append((k, op, other))
    return found


def _is_vector(t):
    from mlir.ir import VectorType
    try:
        VectorType(t)
        return True
    except (TypeError, ValueError):
        return False


def _peel_first_trip(lp, found, inits):
    """(the carries' inits read from a clone of the first trip, the next trip's start) when the
    first trip can be peeled, else None."""
    from mlir.dialects import arith
    from mlir.ir import DenseElementsAttr, IndexType, InsertionPoint, IntegerAttr
    from .lib_ir import clone_ops, loop_bounds, op_of, producer
    if len(found) != len(inits):
        return None
    start, end, step = loop_bounds(lp) or (None, None, None)
    if None in (start, end, step) or end - start < 2 * step:
        return None
    for (k, op, _) in found:
        cst = producer(inits[k])
        if cst is None or cst.name != "arith.constant":
            return None
        try:
            got = DenseElementsAttr(cst.attributes["value"])
            if not got.is_splat or float(got.get_splat_value().value) != _IDENTITY[op.name]:
                return None
        except (TypeError, ValueError, AttributeError):
            return None
    body = lp.regions[0].blocks[0]
    iv, carries = body.arguments[0], set(list(body.arguments)[1:])
    inside = {op_of(o) for o in body.operations}
    need, work = [], [other for _, _, other in found]
    while work:  # the ops of the body each extremum's other operand is made of
        v = work.pop()
        o = producer(v)
        if v in carries:
            return None
        if o is None or op_of(o) not in inside or op_of(o) in need:
            continue
        if op_of(o).name.endswith("store") or op_of(o).regions:
            return None
        need.append(op_of(o))
        work.extend(op_of(o).operands)
    order = {o: i for i, o in enumerate(op_of(x) for x in body.operations)}
    need.sort(key=lambda o: order[o])
    with InsertionPoint(lp):
        first = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), start)).result
    mapping = {iv: first}
    clone_ops(need, mapping, InsertionPoint(lp))
    return [mapping.get(other, other) for _, _, other in found], start + step


def _rebuild(lp, found):
    lp = lp.opview
    from mlir.dialects import affine, arith
    from mlir.ir import DenseElementsAttr, FloatAttr, InsertionPoint, Operation, VectorType
    old_body = lp.regions[0].blocks[0]
    old_term = list(old_body.operations)[-1]
    inits = list(lp.operands)[-len(lp.results):] if len(lp.results) else []
    lb_map, start = lp.attributes["lowerBoundMap"].value, None
    peeled = _peel_first_trip(lp, found, inits)
    if peeled is not None:
        inits, start = peeled
    with InsertionPoint(lp):
        if start is not None:
            from mlir.ir import AffineMap
            lb_map = AffineMap.get_constant(start)
        flags = []
        for k, _, _ in found:
            v = inits[k]
            flags.append(arith.CmpFOp(arith.CmpFPredicate.UNO, v, v).result)
        new = affine.AffineForOp(
            lb_map, lp.attributes["upperBoundMap"].value,
            lp.attributes["step"], iter_args=inits + flags,
            lower_bound_operands=[] if start is not None else list(lp.lowerBoundOperands),
            upper_bound_operands=list(lp.upperBoundOperands))
    new_body = new.body
    for old_arg, new_arg in zip(old_body.arguments, list(new_body.arguments)[:len(old_body.arguments)]):
        old_arg.replace_all_uses_with(new_arg)
    with InsertionPoint(new_body):
        for op in list(old_body.operations)[:-1]:
            op.operation.detach_from_parent()
            InsertionPoint(new_body).insert(op.operation)
        seen = list(new_body.arguments)[len(old_body.arguments):]
        outs = list(old_term.operands)
        more = []
        for (k, op, other), flag in zip(found, seen):
            with InsertionPoint(op):
                num = Operation.create(_NUM_OF[op.name], results=[op.results[0].type],
                                       operands=list(op.operands)).results[0]
                nan_here = arith.CmpFOp(arith.CmpFPredicate.UNO, other, other).result
                more.append(arith.OrIOp(flag, nan_here).result)
            op.results[0].replace_all_uses_with(num)
            op.erase()
            outs[k] = num
        affine.AffineYieldOp(outs + more)
    old_term.erase()
    n = len(lp.results)
    ip = InsertionPoint(lp)  # after the new loop, which was built before the old one
    finals = list(new.results)[:n]
    pending = []
    for (k, _, _), seen_nan in zip(found, list(new.results)[n:]):
        vt = VectorType(finals[k].type)
        nan, nz = (arith.ConstantOp(vt, DenseElementsAttr.get_splat(
            vt, FloatAttr.get(vt.element_type, x)), ip=ip).result for x in (float("nan"), -0.0))
        pending.append((k, arith.SelectOp(seen_nan, nan, nz, ip=ip).result))
    for old_res, fin in zip(lp.results, finals):
        old_res.replace_all_uses_with(fin)
    lp.erase()
    for k, poison in pending:
        _carry_through_the_fold(finals[k], poison)


_SLICES = ("vector.extract_strided_slice", "vector.shuffle")


def _carry_through_the_fold(value, poison):
    """The fold after the loop (slices, shuffles and the same extremum) runs on `value` with the
    number form, and `poison` -- NaN where a NaN was seen, else -0.0 -- is folded beside it with
    addf, which carries a NaN through; where the fold ends, value + poison is the answer (x + -0.0
    is x for every x, a zero's sign included)."""
    from mlir.dialects import arith
    from mlir.ir import InsertionPoint, Operation
    from .lib_ir import op_of
    fold, work = [], [value]
    while work:  # every slice, shuffle or extremum downstream of `value`
        for use in list(work.pop().uses):
            op = op_of(use.owner)
            if op.name in _SLICES + tuple(_NUM_OF) and op not in fold:
                fold.append(op)
                work.extend(op.results)
    block = value.owner.block if hasattr(value.owner, "block") else None
    order = {o: i for i, o in enumerate(op_of(x) for x in value.owner.parent.regions[0].blocks[0].operations)} \
        if block is None else {o: i for i, o in enumerate(op_of(x) for x in block.operations)}
    fold.sort(key=lambda o: order[o])
    partner = {value: poison}

    def whole(v, at):
        with InsertionPoint(at):
            return arith.AddFOp(v, partner[v]).result

    for op in fold:
        ins = list(op.operands)
        if not all(x in partner for x in ins):
            for k, x in enumerate(ins):
                if x in partner:
                    op.operands[k] = whole(x, op)
            continue
        if op.name in _SLICES:
            with InsertionPoint(op):
                twin = Operation.create(op.name, results=[op.results[0].type],
                                        operands=[partner[x] for x in ins],
                                        attributes={a: op.attributes[a] for a in op.attributes})
            partner[op.results[0]] = twin.results[0]
        else:
            with InsertionPoint(op):
                num = Operation.create(_NUM_OF[op.name], results=[op.results[0].type],
                                       operands=ins).results[0]
                p = arith.AddFOp(partner[ins[0]], partner[ins[1]]).result
            op.results[0].replace_all_uses_with(num)
            op.erase()
            partner[num] = p
    for v in list(partner):  # where the fold ends: what it hands on, with its NaN back
        if v is poison or v in partner.values():
            continue
        for use in list(v.uses):
            owner = op_of(use.owner)
            if (owner.name in _SLICES + tuple(_NUM_OF.values())
                    and all(x in partner for x in owner.operands)):
                continue
            if owner.name == "arith.addf" and partner.get(v) in list(owner.operands):
                continue
            owner.operands[use.operand_number] = whole(v, owner)
