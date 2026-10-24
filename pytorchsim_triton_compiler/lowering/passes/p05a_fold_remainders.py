"""pytorchsim-fold-remainders: a remainder drops the terms of its dividend its divisor divides,
`(m*P + r) rem P == r rem P` (RoPE's `(base + x) % D` becomes the lane number's); a signed one folds only
when every term is non-negative."""

NAME = "pytorchsim-fold-remainders"

_CASTS = ("arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui")


def _const(v):
    """The integer `v` is everywhere (a scalar or a splat constant), or None."""
    from .lib_ir import const_int, producer
    c = const_int(v)
    if c is not None:
        return c
    o = producer(v)
    if o is not None and o.name == "vector.broadcast":
        return _const(o.operands[0])
    if o is not None and o.name == "arith.constant":
        from mlir.ir import DenseIntElementsAttr
        try:
            a = DenseIntElementsAttr(o.attributes["value"])
        except (TypeError, ValueError):
            return None
        if a.is_splat:
            return a.get_splat_value().value
    return None


def _program_ids(func):
    """The kernel's program ids: the last three integer arguments of its function."""
    import mlir.ir as ir
    args = [a for a in func.regions[0].blocks[0].arguments if isinstance(a.type, ir.IntegerType)]
    return list(args[-3:]) if len(args) >= 6 else []


def _nonneg(v, pids, depth=0):
    from mlir.ir import BlockArgument
    from .lib_ir import producer
    c = _const(v)
    if c is not None:
        return c >= 0
    if depth > 32:
        return False
    if any(v == p for p in pids):
        return True
    if isinstance(v, BlockArgument):
        owner = v.owner.owner
        if owner is not None and owner.name == "affine.for" and v.arg_number == 0:
            return _nonneg_lower_bound(owner)
        return False
    o = producer(v)
    if o is None:
        return False
    if o.name in ("pytorchsim.vlane_idx", "vector.step"):
        return True
    if o.name in _CASTS[:2] + ("arith.extui", "vector.broadcast"):
        return _nonneg(o.operands[0], pids, depth + 1)
    if o.name in ("arith.addi", "arith.muli"):
        return all(_nonneg(x, pids, depth + 1) for x in o.operands)
    return False


def _nonneg_lower_bound(loop):
    from .lib_ir import loop_bounds
    b = loop_bounds(loop)
    return b is not None and b[0] is not None and b[0] >= 0


def _multiple(v, p, depth=0):
    """Whether `v` is a multiple of `p` by its ops: a constant one, or a product or sum of them."""
    from .lib_ir import producer
    c = _const(v)
    if c is not None:
        return c % p == 0
    o = producer(v)
    if o is None or depth > 32:
        return False
    if o.name in _CASTS + ("vector.broadcast",):
        return _multiple(o.operands[0], p, depth + 1)
    if o.name == "arith.muli":
        return any(_multiple(x, p, depth + 1) for x in o.operands)
    if o.name == "arith.addi":
        return all(_multiple(x, p, depth + 1) for x in o.operands)
    return False


_UNSIGNED_CASTS = ("arith.index_castui", "arith.extui")


def _terms(v, unsigned=None, depth=0):
    """`v` as the (term, widened unsigned) its additions sum, through the casts between them; a
    cast of the other signedness than one already passed ends the descent, at that cast."""
    from .lib_ir import producer
    o = producer(v)
    if o is not None and depth < 32:
        if o.name == "arith.addi":
            return (_terms(o.operands[0], unsigned, depth + 1)
                    + _terms(o.operands[1], unsigned, depth + 1))
        if o.name in _CASTS:
            u = o.name in _UNSIGNED_CASTS
            if unsigned is None or unsigned == u:
                return _terms(o.operands[0], u, depth + 1)
    return [(v, bool(unsigned))]


def _cast_to(v, ty, unsigned=False):
    """`v` at `ty`, widened as it was peeled (`unsigned`); a lane number is read at `ty` from the
    hardware, not cast to it."""
    from mlir.dialects import arith
    from mlir.ir import IndexType, IntegerType, Operation, VectorType
    from .lib_ir import producer
    o = producer(v)
    if o is not None and o.name == "pytorchsim.vlane_idx":
        return Operation.create("pytorchsim.vlane_idx", results=[ty], operands=[]).results[0]
    src = VectorType(v.type).element_type if isinstance(v.type, VectorType) else v.type
    dst = VectorType(ty).element_type if isinstance(ty, VectorType) else ty
    if src == dst:
        return v
    if isinstance(src, IndexType) or isinstance(dst, IndexType):
        return (arith.IndexCastUIOp if unsigned else arith.IndexCastOp)(ty, v).result
    if IntegerType(src).width < IntegerType(dst).width:
        return (arith.ExtUIOp if unsigned else arith.ExtSIOp)(ty, v).result
    return arith.TruncIOp(ty, v).result


def _fold(rem, pids):
    from mlir.dialects import arith
    from mlir.ir import InsertionPoint, VectorType
    p = _const(rem.operands[1])
    if p is None or p <= 0:
        return False
    terms = _terms(rem.operands[0])
    rest = [(t, u) for t, u in terms if not _multiple(t, p)]
    if not rest or len(rest) == len(terms):
        return False
    if rem.name == "arith.remsi" and not all(_nonneg(t, pids) for t, _ in terms):
        return False
    ty = rem.results[0].type
    if any(isinstance(t.type, VectorType) != isinstance(ty, VectorType) for t, _ in rest):
        return False
    with InsertionPoint(rem), rem.location:
        got = _cast_to(rest[0][0], ty, rest[0][1])
        for t, u in rest[1:]:
            got = arith.AddIOp(got, _cast_to(t, ty, u)).result
        got = (arith.RemSIOp if rem.name == "arith.remsi" else arith.RemUIOp)(
            got, rem.operands[1]).result
    rem.results[0].replace_all_uses_with(got)
    rem.erase()
    return True


def _erase_dead(func):
    """Arithmetic and vector values nothing reads, the fold's leftovers, erased."""
    from .lib_ir import walk
    changed = True
    while changed:
        changed = False
        for o in reversed(list(walk(func.operation))):
            if (o.name.split(".")[0] in ("arith", "vector") and o.results
                    and not o.regions and all(not list(r.uses) for r in o.results)):
                o.erase()
                changed = True


def run(module, **_opts):
    from .lib_ir import funcs, walk
    for func in funcs(module):
        pids = _program_ids(func)
        rems = [o for o in walk(func.operation) if o.name in ("arith.remsi", "arith.remui")]
        if sum(_fold(r, pids) for r in rems):
            _erase_dead(func)
    return module
