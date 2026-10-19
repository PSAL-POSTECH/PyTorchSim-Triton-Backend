"""tritonpytorchsim-hoist-loop-invariants: what an affine loop computes the same on every trip is
computed once before it; a read only when nothing in the loop may write what it reads. The pass is
C++ (dialects/src/lib/HoistLoopInvariants.cpp), where mlir::AliasAnalysis answers that."""

from ...dialects import torchsim  # noqa: F401
from .lib_ir import walk

NAME = "tritonpytorchsim-hoist-loop-invariants"


def run(module, **_opts):
    """The pass over every function, then the accumulators the loops carry put in registers."""
    from mlir.ir import Location
    from mlir.passmanager import PassManager
    PassManager.parse(f"builtin.module(func.func({NAME}))", module.context).run(module.operation)
    with Location.unknown():
        for loop in [o for o in walk(module.operation) if o.name == "scf.for"]:
            _rewrite_carried_accumulator(loop)
    return module


_VIEWS = ("memref.collapse_shape", "memref.expand_shape", "memref.subview", "memref.cast")


def _rewrite_carried_accumulator(loop):
    """A loop whose every trip loads a vector from a place and stores one back there, the place
    touched by nothing else in the loop, carries that vector instead: one load before the loop, one
    store after it. A walk of one trip inside it is first made straight-line, so its vector is the
    loop's own. The accumulator of a chunked reduce then stays in registers across the chunks."""
    body = loop.regions[0].blocks[0]
    # a walk of one trip is made straight-line only when it holds the pair the loop will carry
    for inner in [o.operation for o in list(body.operations) if o.operation.name == "affine.for"]:
        if _is_one_trip(inner) and _find_pair(inner.regions[0].blocks[0], loop):
            _rewrite_one_trip_inline(inner, loop)
    pair = _find_pair(body, loop)
    if pair is None:
        return False
    _build_carry(loop, *pair)
    return True


def _find_pair(block, loop):
    """The (load, store) of one place in `block` that `loop` can carry: the same place and type, the
    load first, indices the loop does not change, the place touched by nothing else in it."""
    ops = [o.operation for o in block.operations]
    loads = [o for o in ops if o.name == "affine.vector_load"]
    stores = [o for o in ops if o.name == "affine.vector_store"]
    for ld in loads:
        st = next((s for s in stores if s.operands[1] == ld.operands[0]
                   and list(s.operands)[2:] == list(ld.operands)[1:]
                   and s.attributes["map"] == ld.attributes["map"]
                   and s.operands[0].type == ld.results[0].type
                   and ld.is_before_in_block(s)), None)
        if st is None or not _is_invariant(list(ld.operands)[1:], loop, block):
            continue
        place = ld.operands[0]
        root = _get_root(place)
        if not _is_invariant([root], loop) or _touched_besides(root, loop, (ld, st)):
            continue
        return ld, st
    return None


def _is_one_trip(inner):
    from mlir.ir import AffineMapAttr, IntegerAttr
    lo_m = AffineMapAttr(inner.attributes["lowerBoundMap"]).value
    hi_m = AffineMapAttr(inner.attributes["upperBoundMap"]).value
    if inner.operands or inner.results or len(lo_m.results) != 1 or len(hi_m.results) != 1:
        return False
    try:
        a, b = int(str(lo_m.results[0])), int(str(hi_m.results[0]))
    except ValueError:
        return False
    return b - a == IntegerAttr(inner.attributes["step"]).value


def _rewrite_one_trip_inline(inner, outer):
    """An affine.for of one trip (constant bounds, upper = lower + step) as its body, the induction
    variable its lower bound, a constant made before `outer`."""
    from mlir.dialects import arith
    from mlir.ir import AffineMapAttr, IndexType, InsertionPoint, IntegerAttr
    lo, hi = inner.attributes["lowerBoundMap"], inner.attributes["upperBoundMap"]
    lo_m, hi_m = AffineMapAttr(lo).value, AffineMapAttr(hi).value
    if inner.operands or len(lo_m.results) != 1 or len(hi_m.results) != 1:
        return False
    try:
        a, b = int(str(lo_m.results[0])), int(str(hi_m.results[0]))
    except ValueError:
        return False
    step = IntegerAttr(inner.attributes["step"]).value
    if b - a != step or inner.results:
        return False
    blk = inner.regions[0].blocks[0]
    with InsertionPoint(outer):
        iv = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), a)).result
    blk.arguments[0].replace_all_uses_with(iv)
    for o in [x.operation for x in blk.operations][:-1]:
        o.move_before(inner)
    inner.erase()
    return True


def _is_invariant(values, loop, block=None):
    """Is every value defined outside `loop` (a block argument of another block, or an op not in
    it)? The induction variable of a walk of one trip in `block` counts: it is made a constant."""
    for v in values:
        owner = v.owner
        o = getattr(owner, "operation", None)
        if o is None:                       # a block argument
            if owner == loop.regions[0].blocks[0]:
                return False
            if block is not None and owner == block and block != loop.regions[0].blocks[0]:
                continue
            continue
        p = o
        while p is not None and p != loop:
            p = p.parent
        if p is not None:
            # a view of an invariant buffer made inside the loop is invariant too
            if o.name in _VIEWS and _is_invariant([o.operands[0]], loop):
                continue
            return False
    return True


def _get_root(v):
    while getattr(getattr(v, "owner", None), "name", None) in _VIEWS:
        v = v.owner.operands[0]
    return v


def _touched_besides(root, loop, mine):
    """Does anything inside `loop` but `mine` use `root` or a view of it?"""
    work, seen = [root], set()
    while work:
        v = work.pop()
        for u in v.uses:
            o = u.owner
            if o.name in _VIEWS and u.operand_number == 0:
                if id(o) not in seen:
                    seen.add(id(o))
                    work.append(o.results[0])
                continue
            if any(o == m for m in mine):
                continue
            p = o
            while p is not None and p != loop:
                p = p.parent
            if p is not None:
                return True
    return False


def _build_carry(loop, ld, st):
    from mlir.dialects import scf
    from mlir.ir import InsertionPoint
    view = ld.operands[0]
    with InsertionPoint(loop):
        outer_view = _build_view_outside(view, loop)
        first = ld.clone()
    first.operands[0] = outer_view
    lb, ub, step = loop.operands[0], loop.operands[1], loop.operands[2]
    inits = list(loop.operands)[3:]
    with InsertionPoint(loop):
        new = scf.ForOp(lb, ub, step, inits + [first.results[0]])
    old = loop.regions[0].blocks[0]
    old.arguments[0].replace_all_uses_with(new.induction_variable)
    args = list(new.inner_iter_args)
    for a, b in zip(list(old.arguments)[1:], args[:-1]):
        a.replace_all_uses_with(b)
    ld.results[0].replace_all_uses_with(args[-1])
    carried = st.operands[0]
    ops = [o.operation for o in old.operations]
    old_yield = ops[-1]
    for o in ops[:-1]:
        if o == ld or o == st:
            continue
        new.body.append(o)
    with InsertionPoint(new.body):
        scf.YieldOp(list(old_yield.operands) + [carried])
    for a, b in zip(loop.results, list(new.results)[:-1]):
        a.replace_all_uses_with(b)
    with InsertionPoint(loop):
        last = st.clone()
    last.operands[0] = new.results[-1]
    last.operands[1] = outer_view
    last.move_after(new.operation)
    ld.erase()
    st.erase()
    loop.erase()


def _build_view_outside(view, loop):
    """`view` itself when it is made outside `loop`, else its view op cloned before the loop."""
    o = getattr(view.owner, "operation", None)
    p = o
    while p is not None and p != loop:
        p = p.parent
    if p is None:
        return view
    c = o.clone()
    c.operands[0] = _build_view_outside(o.operands[0], loop)
    return c.results[0]
