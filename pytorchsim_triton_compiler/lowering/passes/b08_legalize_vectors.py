"""pytorchsim-legalize-vectors: every vector cut to a width one instruction issues, the reductions folded at
that width, scalar transcendentals made vectors, and the lane loops gone -- the machine runs
them."""

from collections import namedtuple

from .lib_ir import (attrs_of, build_vector_body, elem_bits, enum_of, funcs, is_shaped, name_of,
                     op_of, walk, walk_ops)
from .lib_math import RVV_MATH
from .lib_bank_layout import build_a_scratchpad_alloc
from .b07_vectorize import LANE_SPLAT, REDUCE_LOOP, VLANE_LOOP
from .lib_vector_loops import REDUCE_KIND, VECTOR_LOOP

FORBIDS = (f"affine.for[{VLANE_LOOP}]", f"affine.for[{VECTOR_LOOP}]",
           f"affine.for[{REDUCE_LOOP}]", f"affine.*[{LANE_SPLAT}]", "pytorchsim.reduction")

_Width = namedtuple("_Width", "lp step widest v tail")

# The name a loop's location carries, its line the loop's number: llc's regalloc remarks name the
# loop a spill or reload is in by that location, and the retry narrows that loop (loop_lmul).
LOOP_FILE = "pytorchsim-loop"
# The loops numbered: a runtime loop (a Triton `for` over a non-constant range) is an scf one.
_LOOPS = ("affine.for", "scf.for", "scf.while")


class _Fatal(Exception):
    pass


def run(module, *, vlen, lmul, loop_lmul=None, **_opts):
    """Per function: each loop numbered, each marked loop's width chosen and applied, the
    reductions rewritten at it, scalar transcendentals reissued, then the lane loops and the marks
    taken off."""
    from mlir.ir import Location

    loc = Location.unknown()
    with loc:
        for func in funcs(module):
            _number_loops(func)
            for w in _choose_widths(func, vlen, lmul, loop_lmul or {}):
                _apply_width(w)
            _rewrite_reductions(func, loc)
            _rewrite_scalar_transcendentals(func, loc)
            _remove_vlane_loops(func)
            _remove_consumed_marks(func)
    return module


def _number_loops(func):
    """Every loop of `func` (_LOOPS) numbered in program order, the number its location's line: its
    own ops carry it, a nested loop's carry that loop's, so llc's remarks name the innermost loop
    a spill or reload is in. Numbers are the same on every run of the same IR."""
    from mlir.ir import Location
    for n, lp in enumerate([o for o in walk(func.operation) if o.name in _LOOPS], 1):
        at = Location.file(LOOP_FILE, n, 0)
        lp.location = at
        for region in lp.regions:
            for block in region.blocks:
                for inner in block.operations:
                    if inner.operation.name not in _LOOPS:
                        inner.operation.location = at


def _loop_number(op):
    """The number _number_loops gave loop `op`, read off its location; None for any other op."""
    from mlir.ir import FileLineColLoc
    try:
        at = FileLineColLoc(op.location)
    except (TypeError, ValueError):
        return None
    return at.start_line if at.filename == LOOP_FILE else None


def _loop_lmul(lp, lmul, loop_lmul):
    """`lp`'s LMUL: the kernel's, or less where a retry narrowed it, the target kind bounds it
    (TargetKind.max_lmul), or either did to a loop around it."""
    from .target_kind import target_kind
    bound = target_kind().max_lmul
    got, o = lmul, lp.operation
    while o is not None:
        if o.name in _LOOPS and _loop_number(o) is not None:
            got = min(got, loop_lmul.get(_loop_number(o), lmul))
        if bound(o) is not None:
            got = min(got, bound(o))
        parent = o.parent
        o = parent.operation if parent is not None else None
    return got


def _choose_widths(func, vlen, lmul, loop_lmul=None):
    """Each marked loop's width: the fewest trips at most its cap, a width dividing its step when one
    gives them, else the cap and the remainder one tail trip (_apply_width)."""
    from mlir.ir import IntegerAttr
    widths = []
    for lp in [o for o in walk(func.operation)
               if o.name == "affine.for" and VECTOR_LOOP in o.attributes]:
        body = lp.regions[0].blocks[0]
        step = IntegerAttr(lp.attributes["step"]).value
        widest = _get_widest_bits(body, step)
        if not widest:
            continue
        cap = max(1, (vlen * _loop_lmul(lp, lmul, loop_lmul or {})) // widest)
        v = max(d for d in range(1, min(step, cap) + 1) if step % d == 0)
        if -(-step // v) > -(-step // cap):
            v = cap
        widths.append(_Width(lp, step, widest, v, step % v))
    return widths


def _get_widest_bits(body, step):
    widest = 0
    for inner in walk_ops(body):
        for val in list(inner.operation.operands) + list(inner.operation.results):
            got = _get_vector(val)
            if got and got[0] == step:
                widest = max(widest, elem_bits(got[1]))
    return widest


def _apply_width(w):
    """`w`'s loop walked at width v; a remainder walked by a copy of it after it, one trip at the
    remainder's width, which carries on the loop's partial sums in their first elements."""
    from mlir.ir import InsertionPoint
    if w.v == w.step:
        return
    tail = w.lp.operation.clone(InsertionPoint(w.lp)).operation if w.tail else None
    _set_bounds(w.lp, 0, w.step - w.tail, w.v)
    _rewrite_crossers(w.lp, _apply_width_inside(w.lp, w.step, w.v), w.step, w.v)
    if tail is None:
        return
    tail.move_after(w.lp.operation)
    _set_bounds(tail, w.step - w.tail, w.step, w.tail)
    _rewrite_crossers(tail, _apply_width_inside(tail, w.step, w.tail), w.step, w.tail)
    _chain_tail(w.lp, tail, w.v, w.tail)


def _set_bounds(lp, lo, hi, step):
    from mlir.ir import AffineMap, AffineMapAttr, IntegerAttr
    lp.attributes["lowerBoundMap"] = AffineMapAttr.get(AffineMap.get_constant(lo))
    lp.attributes["upperBoundMap"] = AffineMapAttr.get(AffineMap.get_constant(hi))
    lp.attributes["step"] = IntegerAttr.get(lp.attributes["step"].type, step)


def _chain_tail(lp, tail, v, r):
    """Each partial `lp` carries out: its first r elements seed `tail`, and `tail`'s result put back
    over them is what `lp`'s readers read, by same-typed shuffles (a shufflevector each, not per element)."""
    from mlir.ir import InsertionPoint
    for i, res in enumerate(lp.results):
        uses = [(u.owner, u.operand_number) for u in res.uses]
        elem = _get_vector(res)[1]
        with InsertionPoint(tail):
            seed = _build_slice(res, 0, r, elem, tail.location)
        old = tail.operands[i]
        tail.operands[i] = seed
        _retire([old])
        with InsertionPoint(tail):
            wide = _build_shuffle(tail.results[i], elem, list(range(r)) + [0] * (v - r),
                                  tail.location)
            back = _build_two_shuffle(wide, res, elem, list(range(r)) + list(range(v + r, 2 * v)),
                                      tail.location)
        op_of(back.owner).move_after(tail)
        op_of(wide.owner).move_after(tail)
        for owner, k in uses:
            owner.operands[k] = back


def _apply_width_inside(lp, step, v):
    from mlir.ir import VectorType

    def resize(val):
        got = _get_vector(val)
        if got and got[0] == step:
            val.set_type(VectorType.get([v], got[1]))

    under = walk(lp)
    for inner in under:
        if LANE_SPLAT not in inner.attributes:
            for res in inner.results:
                resize(res)
    for owner in [lp] + under:
        for region in owner.regions:
            for blk in region.blocks:
                for arg in blk.arguments:
                    resize(arg)
    for res in lp.results:
        resize(res)
    return under


def _rewrite_crossers(lp, under, step, v):
    from mlir.ir import InsertionPoint
    inside = set(under)
    made = {}
    for op in [lp] + under:
        for i, val in enumerate(op.operands):
            got = _get_vector(val)
            if not got or got[0] != step or op_of(val.owner) in inside:
                continue
            key = (val, got[1])
            if key not in made:
                with InsertionPoint(lp):
                    made[key] = _build_uniform_at(val, v)
                if made[key] is None:
                    raise _Fatal(f"a vector<{step}> that is not uniform enters a loop cut "
                                 f"to {v}: {op}")
            op.operands[i] = made[key]
    _retire([val for val, _ in made])


def _rewrite_reductions(func, loc):
    """Each reduction folded to a vector<1> answer -- one lane reduction, or a halving tree down
    to width 1 -- combined with its accumulator there, read in place of its vector<1> broadcast."""
    from mlir.ir import InsertionPoint
    for red in [o for o in walk(func.operation)
                if o.name in ("vector.reduction", "pytorchsim.reduction")]:
        n = len(red.results)
        srcs, accs = list(red.operands)[:n], list(red.operands)[n:]
        width, elems = _get_vector(srcs[0])[0], [_get_vector(v)[1] for v in srcs]
        bcasts = [_get_reduction_broadcast(r, red) for r in red.results]
        combine = _get_combine(red, elems, loc)
        with InsertionPoint(bcasts[0]):
            folded = _build_lane_reduction(red, srcs[0], width, elems[0], loc)
            if width < 2:
                answers = list(srcs)
            elif folded is not None:
                answers = [folded]
            else:
                answers = _build_reduce_tree(srcs, width, elems, combine, loc)
            if accs:
                answers = combine(answers, [_build_one(a, e, loc) for a, e in zip(accs, elems)], 1)
        for bcast, answer in zip(bcasts, answers):
            bcast.results[0].replace_all_uses_with(answer)
            bcast.erase()
        red.erase()


def _build_one(val, elem, loc):
    """`val` as a vector<1>: itself when it already is one, a scalar broadcast, a wider vector's
    element 0."""
    from mlir.ir import Operation, VectorType
    one = VectorType.get([1], elem)
    if val.type == one:
        return val
    if _get_vector(val) is None:
        return Operation.create("vector.broadcast", results=[one], operands=[val],
                                loc=loc).results[0]
    return _build_element0(val, elem, loc)


def _build_lane_reduction(red, src, width, elem, loc):
    """`src` folded by one lane reduction to vector<1>; None when `red` is not a vector.reduction
    convert-pytorchsim-to-llvm has an RVV instruction for (lane_reduction_instruction)."""
    from mlir.ir import DenseI64ArrayAttr, Operation, VectorType
    from mlir.dialects import arith
    from .c24_convert_pytorchsim_to_llvm import lane_reduction_instruction
    if red.name != "vector.reduction" or width < 2:
        return None
    kind = enum_of(red.attributes["kind"], "#vector.kind<%s>", sorted(_IDENTITY))
    if kind is None or lane_reduction_instruction(kind, elem) is None:
        return None
    one_t = VectorType.get([1], elem)
    acc = arith.ConstantOp(one_t, _identity(kind, one_t)).result
    row = Operation.create("vector.shape_cast", results=[VectorType.get([1, width], elem)],
                           operands=[src], loc=loc).results[0]
    one = Operation.create("vector.multi_reduction", results=[one_t], operands=[row, acc],
                           attributes={"kind": red.attributes["kind"],
                                       "reduction_dims": DenseI64ArrayAttr.get([1])},
                           loc=loc).results[0]
    return one


_IDENTITY = {"add": (-0.0, 0), "maxnumf": (float("nan"), None), "minnumf": (float("nan"), None),
             "maxsi": (None, "smin"), "minsi": (None, "smax"), "maxui": (None, 0),
             "minui": (None, -1), "and": (None, -1), "or": (None, 0), "xor": (None, 0)}


def _identity(kind, vt):
    from mlir.ir import DenseElementsAttr, FloatAttr, IntegerAttr, IntegerType
    elem = vt.element_type
    f, i = _IDENTITY[kind]
    if not isinstance(elem, IntegerType):
        return DenseElementsAttr.get_splat(vt, FloatAttr.get(elem, f))
    w = elem.width
    i = {"smin": -(1 << (w - 1)), "smax": (1 << (w - 1)) - 1}.get(i, i)
    return DenseElementsAttr.get_splat(vt, IntegerAttr.get(elem, i))


def _get_reduction_broadcast(result, red):
    users = [op_of(u.owner) for u in result.uses]
    if (len(users) != 1 or users[0].name != "vector.broadcast"
            or _get_vector(users[0].results[0])[0] != 1):
        raise _Fatal(f"a reduction not broadcast straight back to vector<1>: {red}")
    return users[0]


def _get_combine(red, elems, loc):
    from mlir.ir import Operation, VectorType
    if red.name == "pytorchsim.reduction":
        blk = red.regions[0].blocks[0]
        return lambda his, los, w: build_vector_body(
            blk.operations, w, dict(zip(blk.arguments, his + los)), loc=loc)
    kind = enum_of(red.attributes["kind"], "#vector.kind<%s>",
                   sorted({k.lower() for k in REDUCE_KIND.values()}))
    if kind is None:
        raise _Fatal(f"a reduction kind REDUCE_KIND does not list: {red}")
    kind = kind.upper()
    ops = [o for o, k in REDUCE_KIND.items() if k == kind]
    comb = ops[0] if len(ops) == 1 else next(
        o for o in ops if o.endswith("f") == _is_float(red.operands[0]))
    return lambda his, los, w: [Operation.create(
        comb, results=[VectorType.get([w], elems[0])], operands=[his[0], los[0]],
        loc=loc).results[0]]


def _build_reduce_tree(vs, width, elems, combine, loc):
    from mlir.ir import BoolAttr, DenseElementsAttr, IntegerType, VectorType
    from mlir.dialects import arith
    if width & (width - 1):
        low = 1 << (width.bit_length() - 1)
        rest = width - low
        his = [_build_shuffle(v, e, [low + i if i < rest else 0 for i in range(low)], loc)
               for v, e in zip(vs, elems)]
        los = [_build_slice(v, 0, low, e, loc) for v, e in zip(vs, elems)]
        keep_ty = VectorType.get([low], IntegerType.get_signless(1))
        keep = arith.ConstantOp(keep_ty, DenseElementsAttr.get(
            [BoolAttr.get(i < rest) for i in range(low)], type=keep_ty)).result
        vs = [arith.SelectOp(keep, g, lo).result
              for g, lo in zip(combine(his, los, low), los)]
        width = low
    while width > 1:
        half = width // 2
        his, los = [], []
        for v, elem in zip(vs, elems):
            his.append(_build_slice(v, half, half, elem, loc))
            los.append(_build_slice(v, 0, half, elem, loc))
        vs = combine(his, los, half)
        width = half
    return vs


def _rewrite_scalar_transcendentals(func, loc):
    from mlir.ir import InsertionPoint, Operation, VectorType
    seeds = [o for o in walk(func.operation) if _is_scalar_transcendental(o)]
    if not seeds:
        return
    must, sinks, back = _find_chain_to_move(seeds)
    twinned = _find_ops_to_twin(must)
    twin = _build_scalar_twins(func, twinned, loc)
    for owner, i, v in back:
        with InsertionPoint(owner):
            owner.operands[i] = _build_scalar_out_of_lanes(func, twin[v], loc)
    for b in sinks:
        src, want = twin[b.operands[0]], b.results[0].type
        wide = src
        if want != src.type:
            with InsertionPoint(b):
                one = _build_element0(src, VectorType(src.type).element_type, loc)
                wide = Operation.create("vector.broadcast", results=[want], operands=[one],
                                        loc=loc).results[0]
        b.results[0].replace_all_uses_with(wide)
        b.erase()
    dropped = []
    for op in reversed([o for o in walk(func.operation) if o in twinned]):
        if not any(True for _ in op.results[0].uses):
            dropped.extend(o for o in op.operands if op_of(o.owner) not in twinned)
            op.erase()
    _retire(dropped)
    left = [o for o in walk(func.operation) if _is_scalar_transcendental(o)]
    if left:
        raise _Fatal(f"a scalar transcendental survived reissue: {left[0]}")


def _is_scalar_transcendental(op):
    """A scalar float math op the base ISA has no instruction for: only a vector unit, or an
    expansion over vectors, can compute it."""
    return (op.name.startswith("math.") and op.name not in RVV_MATH and len(op.results) == 1
            and not is_shaped(op.results[0]) and _is_float(op.results[0]))


def _find_chain_to_move(seeds):
    must, work, sinks, back = set(), [o.results[0] for o in seeds], [], []
    while work:
        v = work.pop()
        if v in must:
            continue
        must.add(v)
        for use in v.uses:
            owner = op_of(use.owner)
            if _is_scalar_elementwise(owner):
                work.append(owner.results[0])
            elif name_of(owner) == "vector.broadcast":
                sinks.append(owner)
            elif _is_float(v):
                back.append((owner, use.operand_number, v))
            else:
                raise _Fatal(f"a scalar transcendental feeds '{name_of(owner)}', which "
                             f"wants a non-float scalar back out of the lanes: {owner}")
    return must, sinks, back


def _is_scalar_elementwise(op):
    return (op.name.split(".", 1)[0] in ("arith", "math") and op.name != "arith.constant"
            and len(op.results) == 1 and not is_shaped(op.results[0])
            and not any(is_shaped(o) for o in op.operands))


def _is_float(v):
    got = _get_vector(v)
    return str(got[1] if got else v.type) in ("f16", "bf16", "f32", "f64")


def _find_ops_to_twin(must):
    twinned, frontier = set(), [op_of(v.owner) for v in must]
    while frontier:
        op = frontier.pop()
        if op in twinned:
            continue
        twinned.add(op)
        for o in op.operands:
            d = op_of(o.owner)
            if name_of(d) is not None and _is_scalar_elementwise(d):
                frontier.append(d)
    return twinned


def _build_scalar_twins(func, twinned, loc):
    """Each op of `twinned` reissued on vector<1>, its scalar operands broadcast: a vector
    register is what a transcendental with no base-ISA instruction needs, and one element is one."""
    from mlir.ir import InsertionPoint, Operation, VectorType
    twin = {}
    for op in walk(func.operation):
        if op not in twinned:
            continue
        with InsertionPoint(op):
            ops = [twin[o] if o in twin else _build_uniform_at(o, 1, loc)
                   for o in op.operands]
            for o, t in zip(op.operands, ops):
                twin[o] = t
            new = Operation.create(op.name, results=[VectorType.get([1], op.results[0].type)],
                                   operands=ops, attributes=attrs_of(op), loc=loc)
        twin[op.results[0]] = new.results[0]
    return twin


def _build_scalar_out_of_lanes(func, wide, loc):
    from mlir.ir import AffineMap, AffineMapAttr, IndexType, VectorType
    from mlir.dialects import affine, arith, memref
    elem = VectorType(wide.type).element_type
    buf = build_a_scratchpad_alloc(func, [1], elem)
    one = _build_element0(wide, elem, loc)
    zero = arith.ConstantOp(IndexType.get(), 0).result
    affine.AffineVectorStoreOp(one, buf, [zero], AffineMapAttr.get(AffineMap.get_identity(1)))
    return memref.LoadOp(buf, [zero]).result


def _remove_vlane_loops(func):
    """Unwrap each loop over the lanes used that b07 put around a nest: a vector op already runs
    in every lane at once, so the loop names the lanes and is no code."""
    for f in reversed([o for o in walk(func.operation)
                       if o.name == "affine.for" and VLANE_LOOP in o.attributes]):
        for inner in list(f.regions[0].blocks[0].operations)[:-1]:
            inner.operation.move_before(f)
        f.erase()


def _remove_consumed_marks(func):
    for op in walk(func.operation):
        for mark in (VECTOR_LOOP, REDUCE_LOOP, LANE_SPLAT):
            if mark in op.attributes:
                del op.attributes[mark]


def _get_vector(v):
    from mlir.ir import VectorType
    try:
        vt = VectorType(v.type)
    except (TypeError, ValueError):
        return None
    return (vt.shape[0], vt.element_type) if len(vt.shape) == 1 else None


def _build_slice(src, off, n, elem, loc):
    from mlir.ir import ArrayAttr, IntegerAttr, IntegerType, Operation, VectorType
    i64 = IntegerType.get_signless(64)
    return Operation.create(
        "vector.extract_strided_slice", results=[VectorType.get([n], elem)], operands=[src],
        attributes={"offsets": ArrayAttr.get([IntegerAttr.get(i64, off)]),
                    "sizes": ArrayAttr.get([IntegerAttr.get(i64, n)]),
                    "strides": ArrayAttr.get([IntegerAttr.get(i64, 1)])},
        loc=loc).results[0]


def _build_shuffle(v, elem, mask, loc):
    return _build_two_shuffle(v, v, elem, mask, loc)


def _build_two_shuffle(a, b, elem, mask, loc):
    from mlir.ir import DenseI64ArrayAttr, Operation, VectorType
    return Operation.create("vector.shuffle", results=[VectorType.get([len(mask)], elem)],
                            operands=[a, b], attributes={"mask": DenseI64ArrayAttr.get(mask)},
                            loc=loc).results[0]


def _build_element0(val, elem, loc):
    """Register element 0 of `val` as a vector<1>: a position in the register, which every lane
    holds its own of, not machine lane 0."""
    return _build_slice(val, 0, 1, elem, loc)


def _retire(vals):
    """Erase the producer of each of `vals` that nothing reads any more and that only names a
    value -- a constant or a broadcast -- and so on up through that producer's operands."""
    todo = list(vals)
    while todo:
        v = todo.pop()
        d = op_of(v.owner)
        if (name_of(d) not in ("arith.constant", "vector.broadcast")
                or any(True for _ in v.uses)):
            continue
        todo = [t for t in todo if t != v] + list(d.operands)
        d.erase()


def _build_uniform_at(val, width, loc=None):
    from mlir.ir import DenseElementsAttr, InsertionPoint, Operation, VectorType
    from mlir.dialects import arith
    got = _get_vector(val)
    vt = VectorType.get([width], got[1] if got else val.type)
    d = op_of(val.owner)
    name = name_of(d)
    if name == "arith.constant":
        value = d.attributes["value"]
        if got:
            dense = DenseElementsAttr(value)
            if not dense.is_splat:
                return None
            value = dense.get_splat_value()
        with InsertionPoint(d):
            return arith.ConstantOp(vt, DenseElementsAttr.get_splat(vt, value)).result
    if got and name != "vector.broadcast":
        return None
    return Operation.create("vector.broadcast", results=[vt],
                            operands=[d.operands[0] if got else val], loc=loc).results[0]
