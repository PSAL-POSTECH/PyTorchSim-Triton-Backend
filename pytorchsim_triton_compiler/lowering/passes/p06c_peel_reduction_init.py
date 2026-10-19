"""torchsim-peel-reduction-init: a loop that carries a reduction from its combiner's identity starts from its
first trip's value instead. A trip computes a value and combines it into the carried one; when the carried value
starts as a splat of the combiner's identity, combine(identity, value) is the value, so the first trip's value is
the start and the loop runs from its second trip. The identity splat is then no longer live across the loop: at
the kernel's widest LMUL it is a register group of its own, which llc spilled in softmax's max loop.

A loop is peeled when its bounds are constants and it runs at least two trips, and every value it carries is
combined exactly once, by one of COMBINERS, with a value the carried ones do not feed, from a start that is
that combiner's identity, and is yielded as that combination alone. A loop that still reads that identity
inside (a masked reduction fills the lanes past its mask with it) is left: the splat stays live across the loop
either way, so peeling frees no register, and the first trip's read of it before the loop only lengthens its
life."""

NAME = "torchsim-peel-reduction-init"

# Each combiner and its identity: a float, or for an integer combiner a function of the bit width.
COMBINERS = {
    "arith.maximumf": float("-inf"),
    "arith.minimumf": float("inf"),
    "arith.addf": 0.0,
    "arith.mulf": 1.0,
    "arith.addi": lambda w: 0,
    "arith.muli": lambda w: 1,
    "arith.ori": lambda w: 0,
    "arith.xori": lambda w: 0,
    "arith.andi": lambda w: -1,
    "arith.maxsi": lambda w: -(1 << (w - 1)),
    "arith.minsi": lambda w: (1 << (w - 1)) - 1,
    "arith.maxui": lambda w: 0,
    "arith.minui": lambda w: -1,
}

# What a loop keeps of its own when it is rebuilt: everything but its bounds and step.
_BOUNDS = ("lowerBoundMap", "upperBoundMap", "step", "operandSegmentSizes")


def run(module, **_opts):
    """Every affine.for that qualifies, innermost first, peeled by one trip."""
    from .lib_ir import walk
    loops = [o for o in walk(module.operation) if o.name == "affine.for"]
    for loop in reversed(loops):
        _peel(loop)
    return module


def _is_identity(start, combiner):
    """Whether `start` is an arith.constant splat of `combiner`'s identity, read from the attribute."""
    import math
    import mlir.ir as ir
    from .lib_ir import producer
    p = producer(start)
    if p is None or p.name != "arith.constant":
        return False
    attr = p.attributes["value"]
    want = COMBINERS[combiner]
    try:
        dense = ir.DenseElementsAttr(attr)
        if not dense.is_splat:
            return False
        attr = dense.get_splat_value()
    except ValueError:
        pass
    if isinstance(want, float):
        try:
            got = ir.FloatAttr(attr).value
        except ValueError:
            return False
        return got == want if not math.isinf(want) else math.isinf(got) and (got > 0) == (want > 0)
    try:
        ia = ir.IntegerAttr(attr)
    except ValueError:
        return False
    width = ir.IntegerType(ia.type).width
    mask = (1 << width) - 1
    return (ia.value & mask) == (want(width) & mask)


def _plan(loop):
    """For each carried value, (its combiner op, the operand combined into it); None unless all qualify."""
    from .lib_ir import trip_count
    body = loop.regions[0].blocks[0]
    carried = list(body.arguments)[1:]
    if not carried or (trip_count(loop) or 0) < 2:
        return None
    ops = [o.operation for o in body.operations]
    yielded = list(ops[-1].operands)
    starts = list(loop.operands)[len(list(loop.operands)) - len(carried):]
    feeds = _fed_by(ops, carried)
    plan = []
    for arg, start, out in zip(carried, starts, yielded):
        uses = list(arg.uses)
        if len(uses) != 1:
            return None
        comb = uses[0].owner
        if comb.name not in COMBINERS or len(comb.results) != 1 or comb.results[0] != out:
            return None
        if len(list(out.uses)) != 1:
            return None
        others = [v for v in comb.operands if v != arg]
        if len(others) != 1 or others[0] in feeds or others[0] in carried:
            return None
        if not _is_identity(start, comb.name) or _read_inside(loop, start):
            return None
        plan.append((comb, others[0]))
    return plan


def _read_inside(loop, start):
    """Whether an op inside `loop` reads a constant equal to `start` (its value attribute and type)."""
    from .lib_ir import producer, walk
    want = producer(start)
    for o in walk(loop):
        for v in o.operands:
            p = producer(v)
            if p is not None and p.name == "arith.constant" and v.type == start.type \
                    and p.attributes["value"] == want.attributes["value"]:
                return True
    return False


def _fed_by(ops, carried):
    """The values of `ops` that a carried value reaches, through an op's operands or the ops inside it."""
    from .lib_ir import walk
    tainted = set(carried)
    for o in ops:
        if any(v in tainted for x in [o] + walk(o) for v in x.operands):
            tainted.update(o.results)
    return tainted


def _peel(loop):
    import mlir.ir as ir
    from mlir.dialects import affine
    from .lib_ir import clone_ops, const_index, loop_bounds
    plan = _plan(loop)
    if plan is None:
        return
    first, end, step = loop_bounds(loop)
    body = loop.regions[0].blocks[0]
    iv, carried = body.arguments[0], list(body.arguments)[1:]
    ops = [o.operation for o in body.operations][:-1]
    combs = {c for c, _ in plan}
    before = ir.InsertionPoint(loop)
    with before, loop.location:
        mapping = {iv: const_index(first)}
        clone_ops([o for o in ops if o not in combs], mapping, before)
        starts = [mapping[v] if v in mapping else v for _, v in plan]
        new = affine.AffineForOp(first + step, end, step, iter_args=starts)
    for i in range(len(loop.attributes)):
        name = loop.attributes[i].name
        if name not in _BOUNDS:
            new.operation.attributes[name] = loop.attributes[name]
    nbody = new.operation.regions[0].blocks[0]
    inner = ir.InsertionPoint(nbody)
    with inner, loop.location:
        m = {iv: nbody.arguments[0], **dict(zip(carried, list(nbody.arguments)[1:]))}
        clone_ops(ops, m, inner)
        affine.AffineYieldOp([m[c.results[0]] for c, _ in plan])
    old_starts = list(loop.operands)[len(list(loop.operands)) - len(carried):]
    for r, n in zip(loop.results, new.results):
        r.replace_all_uses_with(n)
    loop.erase()
    from .lib_ir import producer
    for s in old_starts:
        p = producer(s)
        if p is not None and not list(p.results[0].uses):
            p.erase()
