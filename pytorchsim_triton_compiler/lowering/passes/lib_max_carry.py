"""A loop that carries a running `arith.maximumf` / `arith.minimumf` (a max or min reduction's
strip loop) carries it as `maxnumf` / `minnumf` plus a mask of the elements that have seen a NaN,
and puts the NaN back once, after the loop.

`maximumf` propagates a NaN; RVV's vfmax does not, so llc spells every trip's maximumf as two
self-compares, two merges and the vfmax -- five instructions and two extra register groups, which
at LMUL 8 is one group more than the register file holds. Carried this way a trip is the vfmax, one
self-compare of the element read and one mask OR; the result is the same in every element: NaN
wherever an element of the init or of any trip was NaN, else the max (vfmax orders -0 below +0,
as maximumf does)."""

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


def _rebuild(lp, found):
    lp = lp.opview
    from mlir.dialects import affine, arith
    from mlir.ir import DenseElementsAttr, FloatAttr, InsertionPoint, Operation, VectorType
    old_body = lp.regions[0].blocks[0]
    old_term = list(old_body.operations)[-1]
    inits = list(lp.operands)[-len(lp.results):] if len(lp.results) else []
    with InsertionPoint(lp):
        flags = []
        for k, _, _ in found:
            v = inits[k]
            flags.append(arith.CmpFOp(arith.CmpFPredicate.UNO, v, v).result)
        new = affine.AffineForOp(
            lp.attributes["lowerBoundMap"].value, lp.attributes["upperBoundMap"].value,
            lp.attributes["step"], iter_args=inits + flags,
            lower_bound_operands=list(lp.lowerBoundOperands),
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
    for (k, _, _), seen_nan in zip(found, list(new.results)[n:]):
        vt = VectorType(finals[k].type)
        nan = arith.ConstantOp(vt, DenseElementsAttr.get_splat(
            vt, FloatAttr.get(vt.element_type, float("nan"))), ip=ip).result
        finals[k] = arith.SelectOp(seen_nan, nan, finals[k], ip=ip).result
    for old_res, fin in zip(lp.results, finals):
        old_res.replace_all_uses_with(fin)
    lp.erase()
