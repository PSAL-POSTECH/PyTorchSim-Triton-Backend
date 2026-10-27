"""pytorchsim-simplify-index-arithmetic: the kernel's integer index arithmetic, after legalize-vectors fixed
its vector widths, in three parts that read one fact, upstream's integer range analysis:
- no signed wrap: adds and multiplies carry `overflow<nsw>` by the premise kernel_spec.OFFSET_INDEX_BITS for
  i32 index arithmetic and by the argument extents (`arg_numels`), so the analysis bounds what they make;
- remainders: a remainder drops the terms of its dividend its divisor divides, `(m*P + r) rem P == r rem P`,
  when the analysis proves the dividend and every sum on the way to its terms non-negative (one left is
  logged);
- narrowing: arithmetic the analysis proves fits a narrower width the function reads its index vectors at is
  computed there (upstream's arith-int-range-narrowing, ArithIntRangeOpts.cpp), offered only the widths index
  vectors are cast to and lane numbers and steps are made at."""

import logging

from .. import pass_name

NAME = pass_name(__name__)
LOG = logging.getLogger(__name__)
NARROWING = "arith-int-range-narrowing"
"""Upstream's pass the narrowing part runs."""


# No signed wrap: nsw by the premise and by the transfers' extents.

def _classes():
    """(casts, unsigned casts, add, multiply): the binding classes the bounds pass through."""
    from mlir.dialects import arith, vector
    casts = (arith.IndexCastOp, arith.IndexCastUIOp, arith.ExtSIOp, arith.ExtUIOp, vector.BroadcastOp)
    return casts, (arith.IndexCastUIOp, arith.ExtUIOp), arith.AddIOp, arith.MulIOp


def _width(v):
    """The signed width the integer range analysis gives `v`'s elements (an index is 64 bits)."""
    from mlir.ir import IndexType, IntegerType, VectorType
    t = VectorType(v.type).element_type if isinstance(v.type, VectorType) else v.type
    if isinstance(t, IndexType):
        return 64
    return IntegerType(t).width if isinstance(t, IntegerType) else None


def _combine(op, ins):
    """The exact range of the add or multiply `op` of operands in the ranges `ins`."""
    from mlir.dialects import arith
    if isinstance(op.opview, arith.AddIOp):
        return ins[0][0] + ins[1][0], ins[0][1] + ins[1][1]
    corners = [x * y for x in ins[0] for y in ins[1]]
    return min(corners), max(corners)


def _fits(v, lo, hi):
    w = _width(v)
    return w is not None and -(1 << (w - 1)) <= lo and hi < (1 << (w - 1))


def _index_leaf(v):
    """True, False or None (look at its operands): whether `v` is an index by what makes it -- a program id,
    a lane number, a step, a loop counter or a constant is; a loaded value or an argument is data."""
    from mlir.dialects import arith, vector
    from mlir.ir import BlockArgument
    from ...dialect.pytorchsim import LaneIdOp, ProgramIdOp, is_memory_effect_free, loop_like
    from ..utils.ir_utils import producer
    if isinstance(v, BlockArgument):
        loop = loop_like(v.owner.owner) if v.owner.owner is not None else None
        return loop is not None and any(v == iv for iv in loop.ivs)
    o = producer(v)
    if isinstance(o.opview, (ProgramIdOp, LaneIdOp, vector.StepOp, arith.ConstantOp)):
        return True
    return None if not o.regions and is_memory_effect_free(o) else False


def _indices(values):
    """Which of `values` are indices: each is, when every integer operand of the memory-free ops that make it
    is, down to _index_leaf's answers."""
    from ..utils.ir_utils import producer
    known, todo = {}, list(values)
    while todo:
        v = todo[-1]
        if v in known:
            todo.pop()
            continue
        leaf = _index_leaf(v)
        if leaf is not None:
            known[v] = leaf
            todo.pop()
            continue
        ins = [x for x in producer(v).operands if _width(x) is not None]
        missing = [x for x in ins if x not in known]
        if missing:
            todo += missing
            continue
        known[v] = all(known[x] for x in ins)
        todo.pop()
    return known


def _index_arithmetic(func):
    """The i32 adds and multiplies of `func` that compute an index, by _indices; the premise
    kernel_spec.OFFSET_INDEX_BITS says none of them wraps."""
    from mlir.ir import IntegerType, VectorType
    from ...contract.kernel_spec import OFFSET_INDEX_BITS
    from ..utils.ir_utils import walk
    _, _, add, mul = _classes()
    ops = []
    for o in walk(func.operation):
        t = o.results[0].type if isinstance(o.opview, (add, mul)) else None
        t = VectorType(t).element_type if isinstance(t, VectorType) else t
        if isinstance(t, IntegerType) and t.width == OFFSET_INDEX_BITS:
            ops.append(o)
    known = _indices([o.results[0] for o in ops])
    return [o for o in ops if known[o.results[0]]]


class _Bounds:
    """The analysis's range of each integer of `func`, narrowed by what the transfers' extents bound."""

    def __init__(self, func, arg_numels, premised):
        from ...dialect.pytorchsim import signed_ranges
        from ..utils.ir_utils import values
        self.func = func
        ints = [v for v in values(func.operation) if _width(v) is not None]
        self.range = {v: r for v, r in zip(ints, signed_ranges(func, ints)) if r is not None}
        self.bound = {}
        self.premised = set(premised)
        self.marked = set()
        self.arg_numels = arg_numels

    def of(self, v):
        b, r = self.bound.get(v), self.range.get(v)
        if b is None or r is None:
            return b or r
        return max(b[0], r[0]), min(b[1], r[1])

    def narrow(self, v, lo, hi):
        """Bound `v` by [lo, hi]; whether that told anything new."""
        old = self.of(v)
        got = (lo, hi) if old is None else (max(lo, old[0]), min(hi, old[1]))
        if got == old:
            return False
        self.bound[v] = got
        return True

    def seeds(self):
        """The DRAM offset of every ungathered transfer whose first element it moves (each mask's
        constant clamp holds index 0): inside its argument."""
        from ...dialect.pytorchsim import TransferOp
        from ..utils.ir_utils import const_int, walk
        got = []
        for o in walk(self.func.operation):
            t = o.opview
            if not isinstance(t, TransferOp) or t.index is not None:
                continue
            clamp = [const_int(c) for c in t.clamp]
            if None in clamp or any(not lo <= 0 < hi for lo, hi in zip(clamp[::2], clamp[1::2])):
                continue
            numel = self.arg_numels.get(t.dram_arg.value)
            if numel and self.narrow(t.dram_idx, 0, int(numel) - 1):
                got.append(t.dram_idx)
        return got

    def down(self, v):
        """What `v`'s bound makes of the operands of the op that defines it, marking that op: where their
        ranges keep it from wrapping, or for an add of non-negatives, whose wrapped sum would be negative."""
        from ..utils.ir_utils import producer
        casts, _, add, mul = _classes()
        o, b = producer(v), self.of(v)
        if o is None or b is None:
            return []
        if isinstance(o.opview, casts):
            src = o.operands[0]
            return [src] if b[0] >= 0 and _fits(src, *b) and self.narrow(src, 0, b[1]) else []
        if not isinstance(o.opview, (add, mul)):
            return []
        ins = [self.of(x) for x in o.operands]
        if None in ins:
            return []
        exact = _fits(v, *_combine(o, ins)) or o in self.premised
        if not exact and not (isinstance(o.opview, add) and min(b[0], ins[0][0], ins[1][0]) >= 0):
            return []
        self.marked.add(o)
        got = []
        for x, y in zip(o.operands, reversed(ins)):
            if isinstance(o.opview, add):
                lo, hi = b[0] - y[1], b[1] - y[0]
            elif b[0] >= 0 and y[0] > 0:
                lo, hi = 0, b[1] // y[0]
            else:
                continue
            if self.narrow(x, lo, hi):
                got.append(x)
        return got

    def up(self, v):
        """The results of `v`'s users its bound keeps inside their width, marking each such add or
        multiply."""
        casts, unsigned, add, mul = _classes()
        got = []
        for use in v.uses:
            o = use.owner
            if not isinstance(o.opview, casts + (add, mul)) or not o.results or _width(o.results[0]) is None:
                continue
            ins = [self.of(x) for x in o.operands]
            if any(r is None for r in ins):
                continue
            lo, hi = ins[0] if isinstance(o.opview, casts) else _combine(o, ins)
            if lo < 0 and isinstance(o.opview, unsigned):
                continue
            r = o.results[0]
            if not _fits(r, lo, hi):
                continue
            if isinstance(o.opview, (add, mul)):
                self.marked.add(o)
            if self.narrow(r, lo, hi):
                got.append(r)
        return got


def mark(func, arg_numels):
    """Mark `func`'s adds and multiplies nsw by the premise and by the transfers' extents; (marked, left)."""
    from ...dialect.pytorchsim import mark_no_signed_wrap
    from ..utils.ir_utils import walk
    _, _, add, mul = _classes()
    premised = _index_arithmetic(func)
    for o in premised:
        mark_no_signed_wrap(o)
    b = _Bounds(func, arg_numels, premised)
    todo = b.seeds()
    while todo:
        v = todo.pop()
        todo += b.down(v) + b.up(v)
    marked = b.marked | b.premised
    for o in b.marked - b.premised:
        mark_no_signed_wrap(o)
    left = sum(1 for o in walk(func.operation) if isinstance(o.opview, (add, mul)) and o not in marked)
    return len(marked), left


# Remainders: the terms the divisor divides dropped where the ranges prove it keeps the value.

def _casts():
    """(casts the terms are peeled through, the unsigned ones among them), as binding classes."""
    from mlir.dialects import arith
    unsigned = (arith.IndexCastUIOp, arith.ExtUIOp)
    return (arith.IndexCastOp, arith.ExtSIOp) + unsigned, unsigned


def _const(v):
    """The integer `v` is everywhere (a scalar or a splat constant), or None."""
    from mlir.dialects import arith, vector
    from ..utils.ir_utils import const_int, producer
    c = const_int(v)
    if c is not None:
        return c
    o = producer(v)
    if o is not None and isinstance(o.opview, vector.BroadcastOp):
        return _const(o.operands[0])
    if o is not None and isinstance(o.opview, arith.ConstantOp):
        from mlir.ir import DenseIntElementsAttr
        try:
            a = DenseIntElementsAttr(o.opview.value)
        except (TypeError, ValueError):
            return None
        if a.is_splat:
            return a.get_splat_value().value
    return None


def _multiple(v, p, depth=0):
    """Whether `v` is a multiple of `p` by its ops: a constant one, or a product or sum of them."""
    from mlir.dialects import arith, vector
    from ..utils.ir_utils import producer
    c = _const(v)
    if c is not None:
        return c % p == 0
    o = producer(v)
    if o is None or depth > 32:
        return False
    if isinstance(o.opview, _casts()[0] + (vector.BroadcastOp,)):
        return _multiple(o.operands[0], p, depth + 1)
    if isinstance(o.opview, arith.MulIOp):
        return any(_multiple(x, p, depth + 1) for x in o.operands)
    if isinstance(o.opview, arith.AddIOp):
        return all(_multiple(x, p, depth + 1) for x in o.operands)
    return False


def _peel(o, unsigned):
    """The signedness `o` widens by when the descent from `unsigned` goes through it, else None."""
    casts, unsigned_casts = _casts()
    if not isinstance(o.opview, casts):
        return None
    u = isinstance(o.opview, unsigned_casts)
    return u if unsigned is None or unsigned == u else None


def _terms(v, unsigned=None, depth=0):
    """`v` as the (term, widened unsigned) its additions sum, through the casts between them; a
    cast of the other signedness than one already passed ends the descent, at that cast."""
    from mlir.dialects import arith
    from ..utils.ir_utils import producer
    o = producer(v)
    if o is not None and depth < 32:
        if isinstance(o.opview, arith.AddIOp):
            return (_terms(o.operands[0], unsigned, depth + 1)
                    + _terms(o.operands[1], unsigned, depth + 1))
        u = _peel(o, unsigned)
        if u is not None:
            return _terms(o.operands[0], u, depth + 1)
    return [(v, bool(unsigned))]


def _summed(v, unsigned=None, depth=0, cast=False):
    """The values _terms passes on its way to `v`'s terms, each sum and each cast with what they add;
    a term a cast takes is left out, as the cast says what it adds."""
    from mlir.dialects import arith
    from ..utils.ir_utils import producer
    o = producer(v)
    if o is not None and depth < 32:
        if isinstance(o.opview, arith.AddIOp):
            return [v] + [x for a in o.operands for x in _summed(a, unsigned, depth + 1)]
        u = _peel(o, unsigned)
        if u is not None:
            return [v] + _summed(o.operands[0], u, depth + 1, True)
    return [] if cast else [v]


def _cast_to(v, ty, unsigned=False):
    """`v` at `ty`, widened as it was peeled (`unsigned`); a lane number is read at `ty` from the
    hardware, not cast to it."""
    from mlir.dialects import arith
    from mlir.ir import IndexType, IntegerType, VectorType
    from ...dialect.pytorchsim import LaneIdOp
    from ..utils.ir_utils import producer
    o = producer(v)
    if o is not None and isinstance(o.opview, LaneIdOp):
        return LaneIdOp(ty, o.opview.lanes.value).result
    src = VectorType(v.type).element_type if isinstance(v.type, VectorType) else v.type
    dst = VectorType(ty).element_type if isinstance(ty, VectorType) else ty
    if src == dst:
        return v
    if isinstance(src, IndexType) or isinstance(dst, IndexType):
        return (arith.IndexCastUIOp if unsigned else arith.IndexCastOp)(ty, v).result
    if IntegerType(src).width < IntegerType(dst).width:
        return (arith.ExtUIOp if unsigned else arith.ExtSIOp)(ty, v).result
    return arith.TruncIOp(ty, v).result


def _kept(func, rem, why):
    """Log why `rem` stays as it is; False, as _fold answers then."""
    LOG.warning("%s: %s in %s kept, %s", NAME, rem.name, func.attributes["sym_name"], why)
    return False


def _fold(func, rem):
    """Fold `rem` when its terms allow it; whether it did."""
    from mlir.dialects import arith
    from mlir.ir import InsertionPoint, VectorType
    from ...dialect.pytorchsim import signed_ranges
    p = _const(rem.operands[1])
    if p is None or p <= 0:
        return False
    terms = _terms(rem.operands[0])
    rest = [(t, u) for t, u in terms if not _multiple(t, p)]
    if not rest or len(rest) == len(terms):
        return False
    ranges = signed_ranges(func, _summed(rem.operands[0]))
    if any(r is None or r[0] < 0 for r in ranges):
        return _kept(func, rem, "a sum of its dividend not proven non-negative")
    ty = rem.results[0].type
    if any(isinstance(t.type, VectorType) != isinstance(ty, VectorType) for t, _ in rest):
        return _kept(func, rem, "a term it keeps is a scalar and the remainder a vector, or the other way")
    with InsertionPoint(rem), rem.location:
        got = _cast_to(rest[0][0], ty, rest[0][1])
        for t, u in rest[1:]:
            got = arith.AddIOp(got, _cast_to(t, ty, u)).result
        got = type(rem.opview)(got, rem.operands[1]).result
    rem.results[0].replace_all_uses_with(got)
    rem.erase()
    return True


def _erase_dead(func):
    """Arithmetic and vector values nothing reads, the fold's leftovers, erased: results of ops of the
    arith and vector bindings."""
    from mlir.dialects import arith, vector
    from mlir.ir import OpView
    from ..utils.ir_utils import walk
    kinds = tuple(c for d in (arith, vector) for c in vars(d).values()
                  if isinstance(c, type) and issubclass(c, OpView) and c is not OpView)
    changed = True
    while changed:
        changed = False
        for o in reversed(list(walk(func.operation))):
            if (isinstance(o.opview, kinds) and o.results and not o.regions
                    and all(not list(r.uses) for r in o.results)):
                o.erase()
                changed = True


# Narrowing: upstream's pass, offered the widths the function reads its index vectors at.

def _vector_int_widths(func):
    """The widths, below the index width (XLEN), `func` reads its index vectors at: what an index vector
    is cast to, and the integer widths lane numbers and steps are made at."""
    from mlir.dialects import arith, vector
    from mlir.ir import IndexType, IntegerType, VectorType
    from ...contract.target import XLEN
    from ...dialect.pytorchsim import LaneIdOp
    from ..utils.ir_utils import walk
    got = set()
    for o in walk(func.operation):
        if not o.results or not isinstance(o.results[0].type, VectorType):
            continue
        made = isinstance(o.opview, (LaneIdOp, vector.StepOp))
        cast = (isinstance(o.opview, (arith.IndexCastOp, arith.IndexCastUIOp))
                and isinstance(VectorType(o.operands[0].type).element_type, IndexType))
        ty = VectorType(o.results[0].type).element_type
        if (made or cast) and isinstance(ty, IntegerType) and 1 < ty.width < XLEN:
            got.add(ty.width)
    return sorted(got)


def _narrow(module, func):
    from mlir.passmanager import PassManager
    widths = _vector_int_widths(func)
    if widths:
        PassManager.parse(f"func.func({NARROWING}{{int-bitwidths-supported={','.join(map(str, widths))}}})",
                          module.context).run(func.operation)


def run(module, arg_numels=None, **_opts):
    """The three parts over every function in order: nsw marked, remainders folded, arithmetic narrowed."""
    from mlir.dialects import arith
    from ..utils.ir_utils import funcs, walk
    if arg_numels is None:
        raise ValueError(f"{NAME} needs arg_numels: the kernel's argument extents bound its offsets")
    for func in funcs(module):
        marked, left = mark(func, arg_numels)
        if left:
            LOG.info("%s: %s: %d add/mul marked nsw, %d neither the premise nor an extent reaches",
                     NAME, func.attributes["sym_name"], marked, left)
        rems = [o for o in walk(func.operation) if isinstance(o.opview, (arith.RemSIOp, arith.RemUIOp))]
        if sum(_fold(func, r) for r in rems):
            _erase_dead(func)
        _narrow(module, func)
    return module
