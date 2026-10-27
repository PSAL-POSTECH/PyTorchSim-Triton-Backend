"""pytorchsim-verify-spad-containment: every access of a scratchpad buffer stays inside that
buffer's own slot, one lane's need_per_lane share; past it is a compile error whatever sits beside
it in .spad, and an access whose range is not static is counted unknown, never refused."""

from collections import namedtuple

from ...dialect.pytorchsim import StridedLoadOp, StridedStoreOp, effects_on, view_root
from ..utils.bank_layout import in_scratchpad, is_one_bank, lane_extent, need_per_lane
from ..utils.ir_utils import elem_bytes, funcs, op_of, producer, strides_and_offset, walk
from ..utils.liveness import interval
from .. import pass_name

NAME = pass_name(__name__)

Audit = namedtuple("Audit", "checked unknown outside")


class SpadOverrun(Exception):
    """An access reaches past its scratchpad buffer's own slot: a compile error, never a write into
    whichever buffer the placement happened to put next."""


def run(module, *, vectorlane, subtarget=None, log=None, **_):
    """Refuse the first access of `module` that leaves its buffer's slot; `log` gets how many were
    proven inside and how many are unknown."""
    got = audit(module, vectorlane, subtarget)
    if log is not None:
        log(f"[lower] {NAME}: {got.checked} access(es) inside their slot, "
            f"{len(got.unknown)} unknown, {len(got.outside)} outside")
    if got.outside:
        op, what, lo, hi, slot = got.outside[0]
        raise SpadOverrun(
            f"{len(got.outside)} access(es) leave their scratchpad buffer's slot. First: {what} "
            f"reaches bytes [{lo}, {hi}) of a slot of {slot} per lane (need_per_lane): {op}")
    return module


def audit(module, vectorlane, subtarget=None):
    """(checked, unknown, outside) over every access of a scratchpad buffer in `module`: how many are
    proven inside, (op, what) of each whose range is not static, (op, what, lo, hi, slot) of each
    proven to leave its slot."""
    footprints = _dma_footprints(subtarget)
    checked, unknown, outside = 0, [], []
    for func in funcs(module):
        known = _Ranges(func)
        for op in walk(func.operation):
            for what, v, span in _accesses(op, known, footprints, vectorlane):
                root = view_root(v)
                if span is None or not _is_allocated(root):
                    unknown.append((op, what))
                    continue
                slot = _slot_bytes(root, vectorlane)
                lo, hi = span[0] * elem_bytes(_elem(v)), span[1] * elem_bytes(_elem(v))
                if lo < 0 or hi > slot:
                    outside.append((op, what, lo, hi, slot))
                else:
                    checked += 1
    return Audit(checked, unknown, outside)


def _dma_footprints(subtarget):
    """The target's own statement of what a transfer's descriptor touches, which it must make as it
    states DMA_DIMS; None without a subtarget."""
    if subtarget is None:
        return None
    from ...targets import target
    return target(subtarget).sram_footprints


def _is_allocated(root):
    from mlir.ir import BlockArgument
    if isinstance(root, BlockArgument):
        return False
    return in_scratchpad(root.type) and "allocate" in effects_on(op_of(root.owner), root)


def _elem(v):
    from mlir.ir import MemRefType
    return MemRefType(v.type).element_type


def _slot_bytes(root, vectorlane):
    """One lane's share of `root` as .spad reserves it: need_per_lane, the one definition."""
    from mlir.ir import MemRefType
    mt = MemRefType(root.type)
    numel = 1
    for d in mt.shape:
        numel *= d
    return need_per_lane(numel, lane_extent(root), vectorlane) * elem_bytes(mt.element_type)


def _accesses(op, known, footprints, vectorlane):
    """(what, memref, [lo, hi) in its elements from its buffer's start, or None) of each access `op`
    makes of a scratchpad buffer: what its effects say it reads or writes, where its indices put it;
    an op that is no view and states nothing it does to the buffer lets its address go, unknown."""
    from ...dialect.pytorchsim import TransferOp
    spad = [(k, v) for k, v in enumerate(op.operands) if in_scratchpad(v.type)]
    if not spad:
        return []
    if op.name == TransferOp.OPERATION_NAME:
        if footprints is None:
            return [("transfer", v, None) for _, v in spad]
        return [("transfer", v, span) for v, span in footprints(op, vectorlane, known.of)]
    out = []
    for k, v in spad:
        eff = effects_on(op, v)
        if eff & {"read", "write"}:
            out.append((op.name, v, _indexed_span(op, k, v, known)))
        elif _is_a_view_of(op, v):
            if _is_reinterpreting_view(op, v):
                out.append(("view", op.results[0], _view_span(op.results[0])))
        elif not eff & _STATED:
            out.append((op.name, v, None))
    return out


_STATED = frozenset({"allocate", "free", "sync"})


def _is_a_view_of(op, v):
    """Whether `op` is a ViewLikeOpInterface view of `v` in its own memory space: one of its results
    is a step on view_root's walk back to the buffer `v` names."""
    root = view_root(v)
    return any(in_scratchpad(r.type) and view_root(r) == root and r != root for r in op.results)


def _is_reinterpreting_view(op, v):
    """A view whose result states a lane's whole run itself (one bank: no lane axis left to read):
    what it describes must lie in the slot even before anything reads it."""
    from mlir.ir import MemRefType
    if len(op.results) != 1 or not in_scratchpad(op.results[0].type):
        return False
    r = op.results[0]
    return view_root(r) == view_root(v) and r != view_root(r) \
        and is_one_bank(list(MemRefType(r.type).shape))


def _view_span(r):
    from mlir.ir import MemRefType
    so = strides_and_offset(r)
    if not _static(so):
        return None
    return _span(so, [(0, n - 1) for n in MemRefType(r.type).shape])


def _static(so):
    from mlir.ir import ShapedType
    dyn = ShapedType.get_dynamic_stride_or_offset()
    return so is not None and so[1] != dyn and dyn not in so[0]


def _span(so, reach):
    """[lo, hi) of the elements reached through strides and offset `so` by indices running over
    `reach`, one inclusive (first, last) per dim."""
    strides, offset = so
    lo = hi = offset
    for (a, b), s in zip(reach, strides):
        lo, hi = lo + min(a * s, b * s), hi + max(a * s, b * s)
    return lo, hi + 1


def _indexed_span(op, k, v, known):
    """Where an op reading or writing `v` (operand `k`) at the indices after it reaches: a vector runs
    along the trailing dims (a transfer's along its permutation map, cut at the dim's end where not in
    bounds), a strided one `count` steps of `stride`; None when not static or masked."""
    from mlir.ir import MemRefType
    so = strides_and_offset(v)
    if not _static(so) or _masked(op, k):
        return None
    shape = list(MemRefType(v.type).shape)
    got = _walk(op, k, len(shape), known)
    if got is None:
        return None
    idx, runs, cut, tail = got
    if any(i is None for i in idx):
        return None
    reach = [(a, min(b + n - 1, shape[d] - 1) if d in cut else b + n - 1)
             for d, ((a, b), n) in enumerate(zip(idx, runs))]
    lo, hi = _span(so, reach)
    return lo + min(0, tail), hi + max(0, tail)


def _masked(op, k):
    """Whether `op` takes a mask, an i1 vector operand, other than a constant all-true splat: which
    lanes it moves is then the mask's run-time value, not every lane its vector has."""
    from mlir.ir import DenseIntElementsAttr, IntegerType, VectorType
    for j, w in enumerate(op.operands):
        try:
            ety = VectorType(w.type).element_type
            bits = IntegerType(ety).width
        except (TypeError, ValueError):
            continue
        if j == k or bits != 1:
            continue
        p = producer(w)
        if p is None or p.name != "arith.constant":
            return True
        got = DenseIntElementsAttr(p.attributes["value"])
        if not got.is_splat or not got.get_splat_value().value:
            return True
    return False


def _walk(op, k, rank, known):
    """(index interval per dim, run per dim, dims masked at their end, linear reach past the last
    run) of an access whose indices follow its memref operand `k`, as its op's ODS definition reads
    them (_INDEXING: the bindings state no indexing interface); None for an op of no such model."""
    model = _INDEXING.get(op.name)
    if model is None:
        return None
    args = list(op.operands)[k + 1:]
    return model(op, args, _vector_shape(op, k), rank, known)


def _plain(op, args, vec, rank, known):
    """The `rank` index operands after the memref, the vector along the trailing dims."""
    return _trailing([known.of(i) for i in args[:rank]], vec, rank)


def _affine(op, args, vec, rank, known):
    """The index operands through the op's own `map`, the vector along the trailing dims."""
    from mlir.ir import AffineMapAttr
    m = AffineMapAttr(op.attributes["map"]).value
    got = [known.of(i) for i in args[:m.n_dims + m.n_symbols]]
    idx = [None if None in got else
           _affine_interval(e, got[:m.n_dims], got[m.n_dims:]) for e in m.results]
    return _trailing(idx, vec, rank)


def _strided(op, args, vec, rank, known):
    """The index operands, then `count` elements `stride` apart from there."""
    from mlir.ir import IntegerAttr
    count = min(IntegerAttr(op.attributes["count"]).value, vec[0])
    return ([known.of(i) for i in args[:rank]], [1] * rank, (),
            (count - 1) * IntegerAttr(op.attributes["stride"]).value)


def _vector_shape(op, k):
    """The shape of the vector `op` moves: its result's, or a vector operand's other than `k`."""
    from mlir.ir import VectorType
    for v in list(op.results) + [w for j, w in enumerate(op.operands) if j != k]:
        try:
            return list(VectorType(v.type).shape)
        except (TypeError, ValueError):
            continue
    return []


def _trailing(idx, vec, rank):
    if len(vec) > rank:
        return None
    return idx, [1] * (rank - len(vec)) + vec, (), 0


def _transfer(op, args, vec, rank, known):
    """The index operands, the vector along the op's own `permutation_map`, cut where not in_bounds."""
    from mlir.ir import AffineDimExpr, AffineMapAttr, ArrayAttr, BoolAttr
    idx = [known.of(i) for i in args[:rank]]
    m = AffineMapAttr(op.attributes["permutation_map"]).value
    bounds = ([BoolAttr(a).value for a in ArrayAttr(op.attributes["in_bounds"])]
              if "in_bounds" in op.attributes else [False] * len(m.results))
    runs, cut = [1] * rank, set()
    for j, e in enumerate(m.results):
        if isinstance(e, AffineDimExpr):
            runs[e.position] = vec[j]
            if not bounds[j]:
                cut.add(e.position)
    return idx, runs, cut, 0


_INDEXING = {
    **dict.fromkeys(("memref.load", "memref.store", "vector.load", "vector.store",
                     "vector.maskedload", "vector.maskedstore"), _plain),
    **dict.fromkeys(("affine.load", "affine.store", "affine.vector_load", "affine.vector_store"),
                    _affine),
    **dict.fromkeys(("vector.transfer_read", "vector.transfer_write"), _transfer),
    **dict.fromkeys((StridedLoadOp.OPERATION_NAME, StridedStoreOp.OPERATION_NAME), _strided),
}


def _affine_interval(e, dims, syms):
    """(low, high), inclusive, of affine expression `e` over operand intervals, or None."""
    import mlir.ir as ir
    if isinstance(e, ir.AffineConstantExpr):
        return e.value, e.value
    if isinstance(e, ir.AffineDimExpr):
        return dims[e.position]
    if isinstance(e, ir.AffineSymbolExpr):
        return syms[e.position]
    a = _affine_interval(e.lhs, dims, syms)
    b = _affine_interval(e.rhs, dims, syms)
    if a is None or b is None:
        return None
    if isinstance(e, ir.AffineAddExpr):
        return a[0] + b[0], a[1] + b[1]
    if isinstance(e, ir.AffineMulExpr):
        corners = [x * y for x in a for y in b]
        return min(corners), max(corners)
    if b[0] != b[1] or b[0] <= 0:
        return None
    c = b[0]
    if isinstance(e, ir.AffineFloorDivExpr):
        return a[0] // c, a[1] // c
    if isinstance(e, ir.AffineCeilDivExpr):
        return -(-a[0] // c), -(-a[1] // c)
    if isinstance(e, ir.AffineModExpr):
        if a[0] // c == a[1] // c:
            return a[0] % c, a[1] % c
        return 0, c - 1
    return None


class _Ranges:
    """What each index of a function can be: liveness.interval, told every loop counter's range and
    every affine.apply's, in program order so an inner bound reads an outer counter."""

    def __init__(self, func):
        self.known = []
        for op in walk(func.operation):
            read = _BOUNDS.get(op.name)
            if read is not None:
                got = read(self, op)
                if got is not None:
                    self.known.append((op.regions[0].blocks[0].arguments[0], got))
            elif op.name == "affine.apply":
                from mlir.ir import AffineMapAttr
                m = AffineMapAttr(op.attributes["map"]).value
                args = [self.of(v) for v in op.operands]
                got = _affine_interval(m.results[0], args[:m.n_dims], args[m.n_dims:])
                if got is not None:
                    self.known.append((op.results[0], got))

    def of(self, v):
        """(low, high), inclusive, of index `v`, or None when it is not static."""
        return interval(v, self.known)

    def scf_for(self, op):
        """scf.for's counter: its bounds and step are its first three operands."""
        ops = list(op.operands)
        lb, ub, step = self.of(ops[0]), self.of(ops[1]), self.of(ops[2])
        if lb is None or ub is None or step is None or step[0] <= 0:
            return None
        return _counter_range(lb, ub, step[0] if step[0] == step[1] else 1)

    def affine_for(self, op):
        """affine.for's counter: its bounds are its own maps over its operand segments."""
        from mlir.ir import AffineMapAttr, IntegerAttr
        ops = list(op.operands)
        seg = op.attributes["operandSegmentSizes"]
        n_lb, n_ub = int(seg[0]), int(seg[1])
        args_lb = [self.of(v) for v in ops[:n_lb]]
        args_ub = [self.of(v) for v in ops[n_lb:n_lb + n_ub]]
        lbm = AffineMapAttr(op.attributes["lowerBoundMap"]).value
        ubm = AffineMapAttr(op.attributes["upperBoundMap"]).value
        lbs = [_affine_interval(e, args_lb[:lbm.n_dims], args_lb[lbm.n_dims:]) for e in lbm.results]
        ubs = [_affine_interval(e, args_ub[:ubm.n_dims], args_ub[ubm.n_dims:]) for e in ubm.results]
        if not lbs or not ubs or None in lbs or None in ubs:
            return None
        lb = (max(x[0] for x in lbs), max(x[1] for x in lbs))
        ub = (min(x[0] for x in ubs), min(x[1] for x in ubs))
        return _counter_range(lb, ub, IntegerAttr(op.attributes["step"]).value)


_BOUNDS = {"scf.for": _Ranges.scf_for, "affine.for": _Ranges.affine_for}


def _counter_range(lb, ub, step):
    """A loop counter's (low, high) from its bounds' intervals; exact at constant bounds."""
    if ub[1] <= lb[0]:
        return None
    if lb[0] == lb[1] and ub[0] == ub[1]:
        return lb[0], lb[0] + (-(-(ub[1] - lb[0]) // step) - 1) * step
    return lb[0], ub[1] - 1

