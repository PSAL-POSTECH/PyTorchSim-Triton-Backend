"""tritonpytorchsim-fold-decided-compares: an integer compare whose operands' value ranges decide
the result is that constant. A range is read off IR objects, or off `arg_ranges` for a kernel
parameter; whatever is not bounded is unknown, and its compare stays."""

from .lib_ir import const_int, funcs, indexing_maps, loop_bounds, name_of, op_of, producer, walk

NAME = "tritonpytorchsim-fold-decided-compares"

LEAST_INDEX_BITS = 32

DEEPEST = 64


def run(module, arg_ranges=None, **_opts):
    """Fold every decided scalar `arith.cmpi`, then let canonicalize remove what read it.
    `arg_ranges` is {a kernel parameter's position counted from the last: (lo, hi)}."""
    from mlir.ir import Location
    from mlir.passmanager import PassManager
    folded = 0
    with Location.unknown():
        for f in funcs(module):
            known = _Ranges(f.operation, arg_ranges or {})
            for op in [o for o in walk(f.operation) if name_of(o) == "arith.cmpi"]:
                folded += _fold(op, known)
    if folded:
        PassManager.parse("builtin.module(func.func(canonicalize,cse))").run(module.operation)
    return module


def _bits(ty):
    """The width a range of `ty` must fit as a signed number, or None for a type this pass does
    not bound. An `index` is held to LEAST_INDEX_BITS: what fits there fits every wider index."""
    from mlir.ir import IndexType, IntegerType
    if isinstance(ty, IndexType):
        return LEAST_INDEX_BITS
    if isinstance(ty, IntegerType) and ty.is_signless and ty.width >= 2:
        return ty.width
    return None


def _fit(r, ty):
    """`r` when every value in it is a signed number of `ty`, so that no operation producing it
    wrapped; else None."""
    b = _bits(ty)
    if r is None or b is None or not r[0] <= r[1]:
        return None
    return tuple(r) if -(1 << (b - 1)) <= r[0] and r[1] < (1 << (b - 1)) else None


class _Ranges:
    """(lo, hi) holding the signed value an integer SSA value of one function has on every run,
    or None where nothing here bounds it. Asked once per value."""

    def __init__(self, func, arg_ranges):
        self.func, self.arg_ranges, self.seen = func, arg_ranges, {}

    def of(self, v, depth=0):
        if v in self.seen:
            return self.seen[v]
        if depth > DEEPEST or _bits(v.type) is None:
            return None
        self.seen[v] = _fit(self._read(v, depth), v.type)
        return self.seen[v]

    def _read(self, v, depth):
        from mlir.ir import BlockArgument
        if isinstance(v, BlockArgument):
            return self._of_block_argument(v)
        p = producer(v)
        if p.name == "arith.constant":
            c = const_int(v)
            return None if c is None else (c, c)
        if p.name == "linalg.index":
            return _of_linalg_index(p)
        rule = _ARITHMETIC.get(p.name)
        if rule is None:
            return None
        ins = [self.of(o, depth + 1) for o in p.operands]
        return None if None in ins else rule(*ins)

    def _of_block_argument(self, v):
        """An scf.for's induction variable is [lb, the last value lb + k*step below ub]; an entry
        argument of the function is what `arg_ranges` says of its position."""
        block = v.owner
        owner = op_of(block.owner)
        if owner.name == "scf.for" and v.arg_number == 0 and "unsignedCmp" not in owner.attributes:
            b = loop_bounds(owner)
            if b is None or None in b or b[2] <= 0 or b[1] <= b[0]:
                return None
            return b[0], b[0] + b[2] * ((b[1] - 1 - b[0]) // b[2])
        if owner == self.func and block == owner.regions[0].blocks[0]:
            return self.arg_ranges.get(v.arg_number - len(block.arguments))
        return None


def _of_linalg_index(op):
    """`linalg.index d` is [0, extent - 1], the extent being the static size of an operand
    dimension its op's indexing_maps name as exactly d."""
    from mlir.ir import AffineDimExpr, IntegerAttr
    d = IntegerAttr(op.attributes["dim"]).value
    over = op_of(op.parent)
    maps = indexing_maps(over)
    if maps is None or len(maps) != len(over.operands):
        return None
    for operand, m in zip(over.operands, maps):
        for pos, e in enumerate(m.results):
            if isinstance(e, AffineDimExpr) and e.position == d:
                if not operand.type.is_dynamic_dim(pos) and operand.type.shape[pos] > 0:
                    return 0, operand.type.shape[pos] - 1
    return None


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _mul(a, b):
    corners = [x * y for x in a for y in b]
    return min(corners), max(corners)


_ARITHMETIC = {"arith.addi": _add, "arith.muli": _mul, "arith.index_cast": lambda a: a}


def _decided(pred, a, b):
    """True or False when every value of `a` against every value of `b` answers `pred` alike, else
    None. An unsigned order is (is negative, value): the negatives sit above, at any width."""
    from mlir.dialects.arith import CmpIPredicate as P
    if pred in (P.eq, P.ne):
        same = a[0] == a[1] == b[0] == b[1]
        apart = a[1] < b[0] or b[1] < a[0]
        return (pred == P.eq) == same if same or apart else None
    if pred in (P.ult, P.ule, P.ugt, P.uge):
        if a[0] < 0 <= a[1] or b[0] < 0 <= b[1]:
            return None
        a, b = ([(x < 0, x) for x in r] for r in (a, b))
    always, never = {
        P.slt: (a[1] < b[0], a[0] >= b[1]), P.ult: (a[1] < b[0], a[0] >= b[1]),
        P.sle: (a[1] <= b[0], a[0] > b[1]), P.ule: (a[1] <= b[0], a[0] > b[1]),
        P.sgt: (a[0] > b[1], a[1] <= b[0]), P.ugt: (a[0] > b[1], a[1] <= b[0]),
        P.sge: (a[0] >= b[1], a[1] < b[0]), P.uge: (a[0] >= b[1], a[1] < b[0]),
    }[pred]
    return True if always else False if never else None


def _fold(op, known):
    from mlir.dialects import arith
    from mlir.ir import InsertionPoint, IntegerAttr
    a, b = (known.of(o) for o in op.operands)
    if a is None or b is None:
        return 0
    says = _decided(arith.CmpIPredicate(IntegerAttr(op.attributes["predicate"]).value), a, b)
    if says is None:
        return 0
    i1 = op.results[0].type
    with InsertionPoint(op):
        k = arith.ConstantOp(i1, IntegerAttr.get(i1, int(says))).result
    op.results[0].replace_all_uses_with(k)
    op.erase()
    return 1
