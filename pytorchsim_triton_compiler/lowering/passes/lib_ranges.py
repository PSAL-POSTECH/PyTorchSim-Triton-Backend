"""The integer range every element of a value takes, read through its producers: constants, casts,
reshapes, add / sub / mul, linalg.fill and a linalg.generic over linalg.index on tensors, and on
vectors vector.broadcast, vector.step and torchsim.vlane_idx (a lane's own number, below the
machine's lanes). `env` gives the ranges of values the caller knows, a loop's induction variable
over the trips it asks about; anything else is unknown (None), and a compare of an unknown is
never decided."""

from .lib_ir import indexing_maps, operand_segments, producer

_CASTS = ("arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui")
_VIEWS = ("tensor.collapse_shape", "tensor.expand_shape", "tensor.cast", "vector.shape_cast")
_LANE = "torchsim.vlane_idx"

# arith.cmpi predicates slt sle sgt sge, and ult ule ugt uge as their signed forms when nothing is
# negative: (always true, always false) given the operands' ranges
_DECIDE = {2: (lambda a, b: a[1] < b[0], lambda a, b: a[0] >= b[1]),
           3: (lambda a, b: a[1] <= b[0], lambda a, b: a[0] > b[1]),
           4: (lambda a, b: a[0] > b[1], lambda a, b: a[1] <= b[0]),
           5: (lambda a, b: a[0] >= b[1], lambda a, b: a[1] < b[0])}
_UNSIGNED = {6: 2, 7: 3, 8: 4, 9: 5}


def identity_maps(maps):
    """Every map of `maps` is the identity of its dims."""
    from mlir.ir import AffineMap
    return all(m == AffineMap.get_identity(m.n_dims) for m in maps)


def decide(pred, a, b):
    """True or False when every element of `a <pred> b` is that, given their ranges; else None."""
    if a is None or b is None:
        return None
    if pred in _UNSIGNED:
        if a[0] < 0 or b[0] < 0:
            return None
        pred = _UNSIGNED[pred]
    if pred not in _DECIDE:
        return None
    always, never = _DECIDE[pred]
    return True if always(a, b) else False if never(a, b) else None


def range_of(v, env, lanes=None, inner=None):
    """[lo, hi] of every element `v` takes; None where a producer is not one this reads.
    `inner` maps a linalg body's block arguments (and "dims") to what they read."""
    from mlir.ir import DenseIntElementsAttr, IntegerAttr, RankedTensorType, VectorType
    if inner is not None and v in inner:
        return inner[v]
    if v in env:
        return env[v]
    p = producer(v)
    if p is None:
        return None
    n = p.name
    if n == "arith.constant":
        a = p.attributes["value"]
        try:
            c = IntegerAttr(a).value
            return (c, c)
        except ValueError:
            pass
        try:
            d = DenseIntElementsAttr(a)
        except ValueError:
            return None
        if not d.is_splat:
            return None
        c = IntegerAttr(d.get_splat_value()).value
        return (c, c)
    if n in _CASTS or n in _VIEWS or n == "vector.broadcast" or n == "linalg.fill":
        return range_of(p.operands[0], env, lanes, inner)
    if n in ("arith.addi", "arith.subi", "arith.muli"):
        a, b = (range_of(x, env, lanes, inner) for x in p.operands[:2])
        if a is None or b is None:
            return None
        if n == "arith.addi":
            return (a[0] + b[0], a[1] + b[1])
        if n == "arith.subi":
            return (a[0] - b[1], a[1] - b[0])
        prods = [x * y for x in a for y in b]
        return (min(prods), max(prods))
    if n == "vector.step":
        width = VectorType(p.results[0].type).shape[0]
        return (0, width - 1)
    if n == _LANE:
        return None if not lanes else (0, int(lanes) - 1)
    if n == "linalg.index":
        dims = inner.get("dims") if inner else None
        d = IntegerAttr(p.attributes["dim"]).value
        return None if dims is None or dims[d] < 1 else (0, dims[d] - 1)
    if n == "linalg.generic" and len(p.results) == 1:
        maps = indexing_maps(p)
        if maps is None or not identity_maps(maps):
            return None
        ty = p.results[0].type
        if not isinstance(ty, RankedTensorType) or not ty.has_static_shape:
            return None
        blk = p.regions[0].blocks[0]
        args = {"dims": list(ty.shape)}
        for k in range(operand_segments(p)[0]):
            r = range_of(p.operands[k], env, lanes, inner)
            if r is not None:
                args[blk.arguments[k]] = r
        y = list(blk.operations)[-1]
        if y.name != "linalg.yield" or len(y.operands) != 1:
            return None
        return range_of(y.operands[0], env, lanes, args)
    return None


def reads(o, v):
    """`o` depends on `v` through its operands' producers (and their regions)."""
    from .lib_ir import op_of, walk
    seen, todo = [], list(op_of(o).operands)
    while todo:
        x = todo.pop()
        if x == v:
            return True
        p = producer(x)
        if p is None or any(p == q for q in seen):
            continue
        seen.append(p)
        todo.extend(p.operands)
        for inner in walk(p):
            todo.extend(inner.operands)
    return False


#: Stamped on a loop that keeps a compare on its induction variable: why its trips were not split.
MASK_KEPT = "torchsim.mask_kept"

_LOOPS = ("scf.for", "affine.for")


def split_at_decided_compares(module, lanes=None):
    """Each loop, outermost first, cut into the runs of its trips on which every compare that reads
    its induction variable is decided (true on all, or false on all) and the run between them; a
    decided run's compares are folded to their value. A loop with such a compare and no decided run,
    or whose bounds are not constants, is stamped with why (MASK_KEPT)."""
    from .lib_ir import funcs
    for func in funcs(module):
        _walk_blocks(func.operation, lanes)
    return module


def _walk_blocks(op, lanes):
    for region in op.regions:
        for block in region.blocks:
            for o in list(block.operations):
                for piece in (_split(o, lanes) if o.name in _LOOPS else [o]):
                    _walk_blocks(piece, lanes)


def _bounds(loop):
    """(lb, ub, step, first init operand) of a loop whose bounds are constants, else None."""
    from .lib_ir import loop_bounds
    b = loop_bounds(loop)
    if b is None or any(x is None for x in b) or b[2] <= 0:
        return None
    if loop.name == "scf.for":
        return b + (3,)
    seg = [int(x) for x in loop.attributes["operandSegmentSizes"]]
    return None if seg[0] or seg[1] else b + (0,)


def _set_bounds(loop, lb, ub):
    from mlir.ir import AffineMap, AffineMapAttr, InsertionPoint, IntegerAttr
    if loop.name == "affine.for":
        loop.attributes["lowerBoundMap"] = AffineMapAttr.get(AffineMap.get_constant(lb))
        loop.attributes["upperBoundMap"] = AffineMapAttr.get(AffineMap.get_constant(ub))
        return
    from mlir.dialects import arith
    ty = loop.regions[0].blocks[0].arguments[0].type
    with InsertionPoint(loop):
        loop.operands[0] = arith.ConstantOp(ty, IntegerAttr.get(ty, lb)).result
        loop.operands[1] = arith.ConstantOp(ty, IntegerAttr.get(ty, ub)).result


def _enclosing(loop):
    """The ranges of the induction variables of the loops around `loop` with constant bounds."""
    env, p = {}, loop.parent
    while p is not None and p.name != "func.func":
        if p.name in _LOOPS:
            b = _bounds(p)
            if b is not None:
                lb, ub, s, _ = b
                if ub > lb:
                    env[p.regions[0].blocks[0].arguments[0]] = (lb, lb + (ub - lb - 1) // s * s)
        p = p.parent
    return env


def compares_in(loop, lanes=None):
    """Every mask under `loop`, in walk order: a compare (an arith.cmpi outside a linalg body, or a
    linalg.generic whose body is one cmpi of its two inputs) of an index built from the loop's
    induction variable -- one side reads it and has a range from its producers once the loops'
    induction variables have one -- against a bound that does not read it. A compare of data, or of
    two values that both move with the loop, is not one: it is neither cut at nor stamped."""
    from .lib_ir import walk
    iv = loop.regions[0].blocks[0].arguments[0]
    ivs = {iv: (0, 0)}
    for o in walk(loop):
        if o.name in _LOOPS and o != loop:
            ivs[o.regions[0].blocks[0].arguments[0]] = (0, 0)
    p = loop.parent
    while p is not None and p.name != "func.func":
        if p.name in _LOOPS:
            ivs[p.regions[0].blocks[0].arguments[0]] = (0, 0)
        p = p.parent
    out = []
    for o in walk(loop):
        if o == loop:
            continue
        if not ((o.name == "arith.cmpi" and o.parent.name != "linalg.generic") or _generic_compare(o)):
            continue
        a, b = o.operands[0], o.operands[1]
        ra, rb = reads_value(a, iv), reads_value(b, iv)
        index = a if ra and not rb else b if rb and not ra else None
        if index is not None and range_of(index, ivs, lanes or 1) is not None:
            out.append(o)
    return out


def reads_value(v, iv):
    """`v` is `iv` or depends on it through its producers."""
    if v == iv:
        return True
    p = producer(v)
    return p is not None and reads(p, iv)


def _generic_compare(o):
    if o.name != "linalg.generic" or len(o.results) != 1:
        return False
    maps = indexing_maps(o)
    if maps is None or operand_segments(o)[0] != 2 or not identity_maps(maps):
        return False
    ops = list(o.regions[0].blocks[0].operations)
    args = list(o.regions[0].blocks[0].arguments)
    return (len(ops) == 2 and ops[0].name == "arith.cmpi" and ops[1].name == "linalg.yield"
            and list(ops[0].operands) == args[:2] and ops[1].operands[0] == ops[0].result)


def _decide_all(cmps, env, lanes):
    """The value of each of `cmps` over `env`, or None when one is not decided."""
    from mlir.ir import IntegerAttr
    got = []
    for o in cmps:
        cmp = list(o.regions[0].blocks[0].operations)[0] if o.name == "linalg.generic" else o
        pred = IntegerAttr(cmp.attributes["predicate"]).value
        d = decide(pred, range_of(o.operands[0], env, lanes), range_of(o.operands[1], env, lanes))
        if d is None:
            return None
        got.append(d)
    return got


def _inner_env(loop, cmps):
    """The ranges of the loops between `loop` and its compares, over all their trips."""
    env = {}
    for o in cmps:
        p = o.parent
        while p is not None and p != loop:
            if p.name in _LOOPS:
                b = _bounds(p)
                if b is not None and b[1] > b[0]:
                    lb, ub, s, _ = b
                    env[p.regions[0].blocks[0].arguments[0]] = (lb, lb + (ub - lb - 1) // s * s)
            p = p.parent
    return env


def _split(loop, lanes):
    """`loop` cut at its decided runs; the loops it became, in order."""
    from mlir.ir import InsertionPoint, StringAttr
    from .lib_ir import op_of
    cmps = compares_in(loop, lanes)
    if not cmps:
        return [loop]
    b = _bounds(loop)
    if b is None:
        loop.attributes[MASK_KEPT] = StringAttr.get("the loop's bounds are not constants")
        return [loop]
    lb, ub, s, first_init = b
    trips = max(0, -(-(ub - lb) // s))
    iv = loop.regions[0].blocks[0].arguments[0]
    env0 = _enclosing(loop)
    env0.update(_inner_env(loop, cmps))

    def window(i, j):          # trips i .. j-1
        env = dict(env0)
        env[iv] = (lb + i * s, lb + (j - 1) * s)
        return _decide_all(cmps, env, lanes)

    def longest(ok):           # the largest k in 0..trips with ok(k)
        lo, hi = 0, trips
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if ok(mid) else (lo, mid - 1)
        return lo

    head = longest(lambda k: k == 0 or window(0, k) is not None)
    tail = 0 if head == trips else longest(lambda k: k == 0 or window(trips - k, trips) is not None)
    runs = [(0, head), (head, trips - tail), (trips - tail, trips)]
    runs = [(i, j) for i, j in runs if j > i]
    if head == 0 and tail == 0:
        loop.attributes[MASK_KEPT] = StringAttr.get(
            "no run of trips decides its compares from the loop's bounds")
        return [loop]
    if MASK_KEPT in loop.attributes:
        del loop.attributes[MASK_KEPT]
    pieces, carry = [], None
    for i, j in runs:
        piece = op_of(loop.clone(InsertionPoint(loop)))
        _set_bounds(piece, lb + i * s, min(ub, lb + j * s))
        if carry is not None:
            for k, r in enumerate(carry):
                piece.operands[first_init + k] = r
        carry = list(piece.results)
        values = window(i, j)
        if values is None:
            piece.attributes[MASK_KEPT] = StringAttr.get(
                "the trips between the decided runs keep their compares")
        else:
            for o, v in zip(compares_in(piece, lanes), values):
                _fold(o, v)
            _drop_dead(piece)
        pieces.append(piece)
    for old, new in zip(loop.results, carry):
        old.replace_all_uses_with(new)
    loop.erase()
    return pieces


def _fold(o, value):
    """A compare known to be `value` everywhere: each select it drives takes that side, any other
    reader reads the constant (a fill of it, on tensors)."""
    from mlir.ir import BoolAttr, DenseElementsAttr, InsertionPoint, IntegerType, VectorType
    from mlir.dialects import arith, linalg
    res = o.results[0]
    for use in list(res.uses):
        user = use.owner if hasattr(use.owner, "name") else use.owner.operation
        side = _select_side(user, use.operand_number, value)
        if side is not None:
            user.results[0].replace_all_uses_with(side)
    if not list(res.uses):
        return
    i1 = IntegerType.get_signless(1)
    with InsertionPoint(o):
        if o.name == "linalg.generic":
            one = arith.ConstantOp(i1, BoolAttr.get(value)).result
            const = linalg.fill(one, outs=[o.operands[2]])
        elif isinstance(res.type, VectorType):
            const = arith.ConstantOp(res.type, DenseElementsAttr.get_splat(
                res.type, BoolAttr.get(value))).result
        else:
            const = arith.ConstantOp(i1, BoolAttr.get(value)).result
    res.replace_all_uses_with(const)


def _select_side(user, slot, value):
    """What `user` hands back when its condition at operand `slot` is `value`: the chosen operand of
    an arith.select, or the input a linalg.generic yielding one reads for it."""
    k = 1 if value else 2
    if user.name == "arith.select" and slot == 0 and user.parent.name != "linalg.generic":
        v = user.operands[k]
        return v if v.type == user.results[0].type else None
    if user.name != "linalg.generic" or len(user.results) != 1:
        return None
    maps = indexing_maps(user)
    n_in = operand_segments(user)[0]
    if maps is None or not identity_maps(maps) or slot >= n_in:
        return None
    blk = user.regions[0].blocks[0]
    ops = list(blk.operations)
    if len(ops) != 2 or ops[0].name != "arith.select" or ops[1].name != "linalg.yield" \
            or ops[1].operands[0] != ops[0].result:
        return None
    args = list(blk.arguments)
    chosen = ops[0].operands[k]
    if ops[0].operands[0] != args[slot] or chosen not in args[:n_in]:
        return None
    v = user.operands[args.index(chosen)]
    return v if v.type == user.results[0].type else None


_PURE = ("linalg.generic", "linalg.fill", "tensor.empty", "arith.constant", "arith.addi",
         "arith.subi", "arith.muli", "arith.cmpi", "arith.select", "vector.broadcast", "vector.step",
         _LANE) + _CASTS + _VIEWS


def _drop_dead(loop):
    """Ops under `loop` that compute values nothing reads any more, erased; a linalg op only when
    it works on tensors (on memrefs it writes)."""
    from mlir.ir import RankedTensorType
    from .lib_ir import walk
    changed = True
    while changed:
        changed = False
        for o in reversed(walk(loop)):
            if o == loop or o.name not in _PURE or not o.results \
                    or any(list(r.uses) for r in o.results):
                continue
            if o.name.startswith("linalg.") and not all(isinstance(r.type, RankedTensorType)
                                                        for r in o.results):
                continue
            o.erase()
            changed = True
