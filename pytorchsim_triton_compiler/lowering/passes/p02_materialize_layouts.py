"""torchsim-materialize-layouts: the layouts TritonGPU decided made into shapes -- each tile laid
out as [lane, depth...] by its linear layout, a replica built where a broadcast widens it, a
torchsim.copy only where a conversion stands, and the encodings gone."""


from ...dialects import torchsim
from .lib_indexing_maps import get_kept_axes, maps_of
from .lib_ir import (FOR_ARGS, FOR_INITS, attrs_of, funcs, get_condition_forwarded, indexing_maps,
                     int_array, is_reshape_shape, iterator_types, name_of, op_of,
                     operand_segments, parallel_iterator, producer, set_indexing_maps,
                     unread_destinations, walk, writes_through_this_operand)
from .lib_linear_layout import (compose_reshape, digit_bounds, lane_bits_of_the_machine,
                               drop_dims, get_bases, lanes_on_one_dim, physical_perm, spell)
from .p03_fuse import MADE_ONCE

kHandled = ("tensor.empty", "bufferization.alloc_tensor", "linalg.generic",
            "linalg.fill", "linalg.transpose", "linalg.broadcast",
            "linalg.reduce", "torchsim.dot", "tensor.collapse_shape",
            "tensor.expand_shape", "tensor.reshape", "scf.for", "scf.yield", "scf.while",
            "scf.condition", "scf.if", "tensor.extract_slice",
            "tensor.insert_slice", "tensor.extract", "tensor.insert",
            "ttg.convert_layout", torchsim.TransferOp.OPERATION_NAME)
kPlainTransfer = torchsim.stated_attributes(torchsim.TransferOp.OPERATION_NAME)
kFresh = ("tensor.empty", "bufferization.alloc_tensor")
kReshape = ("tensor.collapse_shape", "tensor.expand_shape", "tensor.reshape")
kSlice = ("tensor.extract_slice", "tensor.insert_slice")
kElement = ("tensor.extract", "tensor.insert")
def run(module, **_):
    """The layouts are DECIDED ABOVE, in the TritonGPU encodings the types carry: this pass reads
    them -- the lane axis leads -- prepares the IR, and permutes every tile into its order with the
    copies the encodings' conversions ask for."""
    _rewrite_encoding_casts(module)
    _rewrite_replica_broadcasts(module)
    _spread_laps(module)
    _rewrite_reshape_pairs(module)
    _rewrite_into_digits(module)
    _rewrite_replica_broadcasts(module, digits=True)
    _fold_reshape_round_trips(module)
    for func in funcs(module):
        _rewrite_implicit_broadcasts(func)
    for func in funcs(module):
        for op in list(walk(func.operation)):
            _rewrite_unread_destinations(op)
    stuck = [w for w in (_get_unrewritable(f) for f in funcs(module)) if w]
    if stuck:
        raise NotImplementedError(
            "torchsim-materialize-layouts has no rewrite for %s" % stuck[0])
    plan = _read_plan_physical(module)
    if not isinstance(plan, str):
        plan = _rewrite_same_order_conversions(module, plan)
    if isinstance(plan, str):
        raise NotImplementedError(
            "torchsim-materialize-layouts found no plan: %s" % plan)
    _apply_plan(module, plan)
    _remove_encodings(module)
    return module


def _rewrite_implicit_broadcasts(func):
    from mlir.dialects import linalg, tensor
    from mlir.ir import (Operation, AffineMap, AffineMapAttr, ArrayAttr,
                         DenseI64ArrayAttr, InsertionPoint, ShapedType)
    for op in list(walk(func.operation)):
        maps = maps_of(op)
        if maps is None or not maps or maps[-1] is None:
            continue
        walked = [d for d in maps[-1] if d is not None]
        if len(walked) != len(maps[-1]):
            continue
        extents = _get_iteration_extents(op, maps)
        values = list(op.operands) + list(op.results)
        if op.attributes.get("indexing_maps") is None:
            continue
        for k in range(len(maps) - 1):
            dims = maps[k]
            if dims is None or k >= len(values) or not _has_shape(
                    values[k]):
                continue
            if dims and any(d is None for d in dims):
                collapsed = _build_collapsed_units(op, values[k], dims)
                if collapsed is None:
                    continue
                values[k] = collapsed
                dims = [d for d in dims if d is not None]
            if len(dims) != len(ShapedType(values[k].type).shape):
                continue
            missing = [d for d in walked if d not in dims]
            if not missing or any(d not in extents for d in missing):
                continue
            if (len(ShapedType(values[k].type).shape) + len(missing)
                    != len(walked)):
                continue
            shape = [extents[d] for d in walked]
            elt = ShapedType(values[k].type).element_type
            with InsertionPoint(op):
                empty = Operation.create("tensor.empty", results=[
                    _get_type_like(op.results[0], shape, elt)]).result
                wide = linalg.BroadcastOp(
                    [empty.type], values[k], empty,
                    DenseI64ArrayAttr.get(
                        sorted(walked.index(d) for d in missing)))
            blk = wide.regions[0].blocks.append(elt, elt)
            with InsertionPoint(blk):
                linalg.YieldOp([blk.arguments[0]])
            op.operands[k] = wide.results[0]
            got = list(op.attributes["indexing_maps"])
            n_dims = AffineMapAttr(got[-1]).value.n_dims
            ident = AffineMap.get_identity(n_dims)
            got[k] = AffineMapAttr.get(
                AffineMap.get(n_dims, 0, [ident.results[d] for d in walked]))
            op.attributes["indexing_maps"] = ArrayAttr.get(got)


def _get_iteration_extents(op, maps):
    from mlir.ir import ShapedType
    out = {}
    values = list(op.operands) + list(op.results)
    for k, dims in enumerate(maps):
        if dims is None or k >= len(values):
            continue
        try:
            shape = list(ShapedType(values[k].type).shape)
        except (TypeError, ValueError):
            continue
        for a, d in enumerate(dims):
            if d is not None and a < len(shape):
                out.setdefault(d, shape[a])
    return out


def _has_shape(v):
    from mlir.ir import ShapedType
    try:
        ShapedType(v.type).shape
        return True
    except (TypeError, ValueError):
        return False


def _rewrite_unread_destinations(op):
    from mlir.ir import InsertionPoint, Operation, RankedTensorType
    for slot in sorted(unread_destinations(op)):
        v = op.operands[slot]
        if not isinstance(v.type, RankedTensorType):
            continue
        made = producer(v)
        if (made is not None and name_of(made) == "tensor.empty"
                and len(list(v.uses)) == 1):
            continue
        with InsertionPoint(op):
            fresh = Operation.create("tensor.empty", results=[v.type])
        op.operands[slot] = fresh.results[0]


def _get_unrewritable(func):
    for raw in walk(func.operation):
        op = op_of(raw)
        values = list(op.operands) + list(op.results)
        tensors = [v for v in values if _is_tensor(v) and not is_reshape_shape(v)]
        if not tensors:
            continue
        nm = name_of(op)
        if nm not in kHandled and not _is_a_lone_tile(op, tensors):
            return nm + _get_why_layouts_differ(op, tensors)
        if not all(_is_static(v) for v in tensors):
            return nm + " (a dynamic shape)"
        if nm in kFresh and not _is_fresh(op) and _get_copied(op) is None:
            return nm + " (not fresh)"
        if nm == torchsim.TransferOp.OPERATION_NAME:
            extra = [getattr(a, "name", a) for a in op.attributes
                     if getattr(a, "name", a) not in kPlainTransfer]
            if extra:
                return nm + " (" + ", ".join(sorted(map(str, extra))) + ")"
    return None


def _is_a_lone_tile(op, tensors):
    """An op with no region whose only values are tiles of one type: no axis of one is related to
    another axis of any, so a layout permutes them all alike and their types are all it changes."""
    values = list(op.operands) + list(op.results)
    return (len(tensors) == len(values) and len({str(v.type) for v in values}) == 1
            and not len(op.operation.regions))


def _get_why_layouts_differ(op, tensors):
    """Why an op of tiles alone is still refused: its tiles are in more than one layout, and one op
    moves no tile from one layout to another -- the change belongs between ops."""
    values = list(op.operands) + list(op.results)
    if (len(tensors) != len(values) or len(op.operation.regions)
            or all(v.type == values[0].type for v in values)):
        return ""
    return (" (its tiles are in different layouts, and one op does not move a tile from one "
            "layout to another: issue it in one layout and change layouts between ops -- for "
            "vcix, an `iv` in the operand's layout and a `v.i` in the result's)")


def _is_static(v):
    from mlir.ir import ShapedType
    return all(d != ShapedType.get_dynamic_size() for d in v.type.shape)


def _finish_plan(module, perms, copies, reshape_plan):
    wanted = {(op_of(o).operation, k): w for o, k, w in copies}
    if not all(_is_carried_in_one_order(op_of(raw), perms, wanted)
               for func in funcs(module) for raw in walk(func.operation)
               if name_of(op_of(raw)) == "scf.for"):
        return "an scf.for whose carried tile takes two orders"
    for raw in [r for func in funcs(module) for r in walk(func.operation)
                if name_of(op_of(r)) == "scf.while"]:
        got = _get_while_carried_orders(op_of(raw), perms, wanted)
        if got is None:
            return "an scf.while whose carried tile takes two orders"
        perms.update(got)
    if not all(_is_if_carried_in_one_order(op_of(raw), perms, wanted)
               for func in funcs(module) for raw in walk(func.operation)
               if name_of(op_of(raw)) == "scf.if"):
        return "an scf.if whose arms and result take two orders"
    fixes = []
    for func in funcs(module):
        for raw in walk(func.operation):
            op = op_of(raw)
            if _get_named_attrs(op, perms, wanted) is False:
                return "a named op whose attributes cannot follow its axes"
            if name_of(op) in kSlice and not _is_slice_a_view(op, perms, wanted):
                if not _is_window_static(op):
                    return "a slice that is no view and has a dynamic window"
                fixes.append((op, ("copy", None)))
            if name_of(op) in kReshape:
                got = reshape_plan(op, perms, wanted)
                if got is None:
                    return "a reshape with a tile no layout was assigned to"
                if isinstance(got, str):
                    return got
                fixes.append((op, got))
    return perms, copies, fixes


def _read_plan_physical(module):
    from mlir.ir import RankedTensorType
    perms, relaid = {}, _fold_conversions_of_unlaid_reshapes(module)
    for value in _collect_tensor_values(module):
        ty = RankedTensorType(value.type)
        if ty.encoding is None:
            if ty.rank == 0 or is_reshape_shape(value):
                perms[value] = list(range(ty.rank))
                continue
            return "a tile that reached this pass with no layout: %s" % ty
        bases = get_bases(ty.encoding)
        if bases is None:
            return "a layout that is no linear one: %s" % ty.encoding
        got = physical_perm(bases, list(ty.shape))
        if isinstance(got, str) and got == "lanes over several dims":
            got = physical_perm(lanes_on_one_dim(bases), list(ty.shape))
            relaid.add(value)
        if isinstance(got, str):
            made = name_of(producer(value)) or "a block argument"
            read = sorted({name_of(op_of(u.owner)) for u in value.uses})
            src = producer(value)
            frm = str(src.operands[0].type)[:160] if src is not None and len(src.operands) else ""
            to = [str(op_of(u.owner).results[0].type)[:160] for u in value.uses if len(op_of(u.owner).results)]
            return "%s: %s, made by %s, read by %s; from %s to %s" % (got, ty, made, ", ".join(read), frm, to[:1])
        perms[value] = got
    plan = _finish_plan(module, perms, [], reshape_plan=lambda op, p, w: (
        ("copy", None) if op.operands[0] in relaid or op.results[0] in relaid
        else _get_reshape_plan_physical(op, p, w)))
    if isinstance(plan, str):
        return plan
    perms, copies, fixes = plan
    fixes += [(op_of(raw), ("copy", None)) for func in funcs(module)
              for raw in walk(func.operation) if name_of(op_of(raw)) == "ttg.convert_layout"]
    return perms, copies, fixes


def _fold_conversions_of_unlaid_reshapes(module):
    from mlir.ir import RankedTensorType
    moved = set()
    for func in funcs(module):
        for conv in [op_of(r) for r in walk(func.operation)
                     if name_of(op_of(r)) == "ttg.convert_layout"]:
            x = conv.operands[0]
            src = producer(x)
            if name_of(src) not in ("tensor.expand_shape", "tensor.collapse_shape",
                                    "tensor.reshape") or len(list(x.uses)) != 1:
                continue
            bases = get_bases(RankedTensorType(x.type).encoding)
            if bases is None or physical_perm(bases, list(x.type.shape)) != "lanes over several dims":
                continue
            x.set_type(conv.results[0].type)
            conv.results[0].replace_all_uses_with(x)
            conv.operation.erase()
            moved.add(x)
    return moved


def _rewrite_replica_broadcasts(module, digits=False):
    """Each replica read across the lanes built in each lane once, at the dims the lanes and the read
    walk; one whose tile also laps the lanes is made at the lane digit and waits for the digits,
    which cut its reader, marked `AT_LAPS`, at the laps."""
    from mlir.ir import (AffineDimExpr, AffineMap, Attribute, InsertionPoint, Operation,
                         RankedTensorType, UnitAttr)
    for func in funcs(module):
        for op in [op_of(r) for r in walk(func.operation)]:
            if name_of(op) == "linalg.broadcast" and not digits:
                _rewrite_replica_broadcast_op(op)
                continue
            if name_of(op) != "linalg.generic" or len(op.results) != 1:
                continue
            maps, n_in = indexing_maps(op), operand_segments(op)[0]
            out_ty = RankedTensorType(op.results[0].type)
            rank = out_ty.rank
            out_bases = get_bases(out_ty.encoding) if out_ty.encoding is not None else None
            lanes = {d for b in out_bases[1] for d, x in enumerate(b) if x} if out_bases else set()
            if not lanes or maps[-1] != AffineMap.get_identity(rank):
                continue
            changed = False
            for k in range(n_in):
                x = op.operands[k]
                if not _is_tensor(x) or not _is_linear_replica(x.type):
                    continue
                kept = _get_map_dims(maps[k])
                if kept is None:
                    continue
                elt = RankedTensorType(x.type).element_type
                if not lanes <= set(kept):
                    walked = sorted(lanes | {d for d in kept if d is not None})
                    at = [None if d is None else walked.index(d) for d in kept]
                    if digits:
                        lap = _lap_replica_of(x)
                        if lap is None:
                            continue
                        got = lap.operands[0]
                        if list(RankedTensorType(got.type).shape) != [out_ty.shape[d] for d in walked]:
                            raise DigitRefused("a lane replica made for other digits than its reader's")
                        if name_of(producer(got)) == "linalg.generic":
                            producer(got).operation.attributes[MADE_ONCE] = UnitAttr.get()
                    else:
                        got = _build_lane_replica(x, out_ty, out_bases, walked, at, op) \
                            if _laps_the_lanes(out_bases) else None
                        if got is not None:
                            # the reader is cut at the laps first, and reads it at its lane digit
                            op.operation.attributes[AT_LAPS] = UnitAttr.get()
                            with InsertionPoint(op):
                                op.operands[k] = Operation.create(
                                    LAP_REPLICA, results=[x.type], operands=[got]).results[0]
                            continue
                        ty = RankedTensorType.get([out_ty.shape[d] for d in walked], elt,
                                                  Attribute.parse(spell(_bases_on(out_bases, walked))))
                        got = _build_replica_in_shape(x, ty, at, op, {})
                        if got is None:
                            # no producer makes its copies (a loop carries it, say): it is spread
                            got = _build_spread_fold(x, ty, at, op)
                        if got is None:
                            raise DigitRefused("a replica read across the lanes whose producer, %s, "
                                               "cannot make its copies; innermost: %s" % (
                                                   name_of(producer(x)), _REPLICA_MISSES[0]
                                                   if _REPLICA_MISSES else "?"))
                    maps[k] = AffineMap.get(rank, 0, [AffineDimExpr.get(d) for d in walked])
                elif digits:
                    continue
                else:
                    proj = lambda b: [0 if d is None else b[d] for d in kept]
                    spread = (list(map(proj, out_bases[0])), list(map(proj, out_bases[1])))
                    ty = RankedTensorType.get(list(RankedTensorType(x.type).shape), elt,
                                              Attribute.parse(spell(spread)))
                    got = _build_replica_in_shape(x, ty, list(range(ty.rank)), op, {})
                    if got is None:
                        raise DigitRefused("a replica read along the lanes whose producer, %s, "
                                           "cannot lay it out so" % name_of(producer(x)))
                op.operands[k] = got
                changed = True
            if changed:
                set_indexing_maps(op, maps)
        if digits:
            for o in [op_of(r) for r in walk(func.operation)]:
                if AT_LAPS in o.operation.attributes:
                    del o.operation.attributes[AT_LAPS]
                if name_of(o) == LAP_REPLICA:
                    if list(o.results[0].uses):
                        raise DigitRefused("a lane replica whose reader was not cut at the laps")
                    o.operation.erase()
    _erase_unused(module)


_LAP_CARRIERS = ("linalg.generic", "linalg.reduce", "linalg.fill", "linalg.broadcast",
                 "linalg.transpose", "tensor.empty", "bufferization.alloc_tensor", "scf.for",
                 "scf.yield")


def _spread_laps(module):
    """The ops one cut at the laps computes with -- its producers and readers among the elementwise
    ops, reduces and the loops that carry them -- are cut so too, so no regroup splits the walk."""
    from mlir.ir import UnitAttr
    for func in funcs(module):
        work = [op_of(r) for r in walk(func.operation) if AT_LAPS in op_of(r).operation.attributes]
        while work:
            op = work.pop()
            near = [_carrier_of(v) for v in op.operands] + [op_of(u.owner) for r in op.results
                                                           for u in r.uses]
            if name_of(op) == "scf.for":
                body = op.regions[0].blocks[0]
                near += [op_of(u.owner) for a in list(body.arguments)[1:] for u in a.uses]
                near.append(op_of(list(body.operations)[-1]))
            if name_of(op) == "scf.yield" and name_of(op_of(op.operation.parent)) == "scf.for":
                near.append(op_of(op.operation.parent))
            for o in near:
                if o is not None and name_of(o) in _LAP_CARRIERS and _holds_laps(o) and \
                        AT_LAPS not in o.operation.attributes:
                    o.operation.attributes[AT_LAPS] = UnitAttr.get()
                    work.append(o)


def _holds_laps(op):
    """Does `op` make or read a tile that laps the lanes?"""
    from mlir.ir import RankedTensorType
    for v in list(op.operands) + list(op.results):
        if isinstance(v.type, RankedTensorType) and v.type.encoding is not None:
            b = get_bases(v.type.encoding)
            if b is not None and _laps_the_lanes(b):
                return True
    return False


def _carrier_of(v):
    """The op that makes `v`: its producer, or the loop whose iteration argument it is."""
    p = producer(v)
    if p is not None:
        return p
    owner = getattr(getattr(v, "owner", None), "owner", None)
    return op_of(owner) if owner is not None else None


LAP_REPLICA = "torchsim.lap_replica"


def _lap_replica_of(x):
    """The lap replica a reader's operand stands for, through the reshapes the digits added."""
    p = producer(x)
    while p is not None and name_of(p) in kReshape:
        p = producer(p.operands[0])
    return p if p is not None and name_of(p) == LAP_REPLICA else None


def _bases_on(bases, dims):
    """`bases` on `dims` only: the register bases that step elsewhere dropped."""
    on = lambda b: [b[d] for d in dims]
    return [r for r in map(on, bases[0]) if any(r)], list(map(on, bases[1]))


def _build_lane_replica(x, out_ty, out_bases, walked, at, op):
    """The replica `op` reads made at its lane digit -- the lanes and the dims the read walks, the
    laps left off -- or None when its producers cannot make it so, or it is a fold's spread, which
    has nothing to make in each lane."""
    from mlir.ir import Attribute, RankedTensorType
    p = producer(x)
    while p is not None and name_of(p) in kReshape + ("ttg.convert_layout",):
        p = producer(p.operands[0])
    if p is not None and name_of(p) == "linalg.reduce" and _folds_lanes(p):
        return None
    lanes = {d for b in out_bases[1] for d, v in enumerate(b) if v}
    regs = [r for r in out_bases[0] if not any(r[d] for d in lanes)]
    shape = [(1 << sum(1 for b in out_bases[1] if b[d])) if d in lanes else out_ty.shape[d]
             for d in walked]
    ty = RankedTensorType.get(shape, RankedTensorType(x.type).element_type, Attribute.parse(
        spell(_bases_on((regs, out_bases[1]), walked))))
    return _build_replica_in_shape(x, ty, at, op, {})


def _laps_the_lanes(bases):
    """Do the lanes walk one dim's lowest bits, and the registers step that dim above them, each
    lane holding laps of it?"""
    lanes = [b for b in bases[1] if any(b)]
    dims = {d for b in lanes for d, x in enumerate(b) if x}
    if len(dims) != 1:
        return False
    d = dims.pop()
    if any(sum(1 for x in b if x) != 1 for b in lanes) or [b[d] for b in lanes] != \
            [1 << i for i in range(len(lanes))]:
        return False
    return any(b[d] >= 1 << len(lanes) for b in bases[0])


class DigitRefused(NotImplementedError):
    """A tile this pass cannot cut into digits, and why."""


class _Form:

    def __init__(self, shape, groups, lead):
        self.shape, self.groups, self.lead = list(shape), [list(g) for g in groups], lead

    @property
    def digits(self):
        return ([1] if self.lead else []) + [x for g in self.groups for x in g]

    def is_plain(self):
        return not self.lead and all(len(g) == 1 for g in self.groups)

    def reassociation(self):
        """Each dim's digits, the lead joined to the first dim's (the unit stands before it)."""
        out, at = [], 1 if self.lead else 0
        for g in self.groups:
            out.append(list(range(at, at + len(g))))
            at += len(g)
        if self.lead and out:
            out[0] = [0] + out[0]
        return out

    def dims_of(self, d):
        """The digit dims dim `d` became."""
        return self.reassociation()[d] if not (self.lead and d == 0) else self.reassociation()[0][1:]

    def __eq__(self, other):
        return self.groups == other.groups and self.lead == other.lead


_CUT_LAPS = [False]
AT_LAPS = "torchsim.at_laps"


def _get_form(ty, laps=False):
    """The digits `ty` is cut into: none when its layout lays out as it is; with `laps` (or for an op
    cut at its laps), a lane axis longer than the lanes is cut at the lanes, its laps a digit."""
    from mlir.ir import RankedTensorType
    laps = laps or _CUT_LAPS[0]
    ty = RankedTensorType(ty)
    shape = list(ty.shape)
    bases = get_bases(ty.encoding) if ty.encoding is not None else None
    if bases is None or not shape or (isinstance(physical_perm(bases, shape), list) and not laps):
        return _Form(shape, [[n] for n in shape], False)
    bounds = digit_bounds(bases, shape)
    if bounds is None:
        raise DigitRefused("a basis that steps more than one bit: %s" % ty)
    groups = []
    for n, cuts in zip(shape, bounds):
        if cuts and n & (n - 1):
            raise DigitRefused("a dim no power of two that the layout cuts: %s" % ty)
        width = max(n - 1, 0).bit_length()
        edges = [0] + list(cuts) + [width]
        groups.append([1 << (edges[i + 1] - edges[i]) for i in reversed(range(len(edges) - 1))]
                      if width and cuts else [n])
    lead = not any(any(b) for b in bases[1]) and 1 not in shape
    return _Form(shape, groups, lead)


def _form_type(ty, form):
    from mlir.ir import Attribute, RankedTensorType
    ty = RankedTensorType(ty)
    bases = get_bases(ty.encoding) if ty.encoding is not None else None
    shape = list(ty.shape)
    enc = None
    if bases is None and form.lead and not shape:
        bases = ([], [[] for _ in range(lane_bits_of_the_machine())])
    if bases is not None:
        b = compose_reshape(bases, shape, [x for g in form.groups for x in g]) if shape else bases
        if form.lead:
            b = ([[0] + x for x in b[0]], [[0] + x for x in b[1]])
        enc = Attribute.parse(spell(b))
    return RankedTensorType.get(form.digits, ty.element_type, enc)


def _build_cut(v, form):
    from mlir.dialects import tensor
    if form.is_plain():
        return v
    p = producer(v)
    if name_of(p) == "tensor.collapse_shape" and p.operands[0].type == _form_type(v.type, form):
        return p.operands[0]
    ty = _form_type(v.type, form)
    return tensor.ExpandShapeOp(ty, v, form.reassociation(), [], form.digits).result


def _rewrite_reshape_pairs(module):
    from mlir.ir import InsertionPoint, RankedTensorType
    for func in funcs(module):
        for op in [op_of(r) for r in walk(func.operation) if name_of(op_of(r)) in kReshape]:
            mid = op.operands[0]
            first = producer(mid)
            if name_of(first) not in kReshape or len(list(mid.uses)) != 1:
                continue
            t = RankedTensorType(mid.type)
            b = get_bases(t.encoding) if t.encoding is not None else None
            if b is None or not isinstance(physical_perm(b, list(t.shape)), str):
                continue
            with InsertionPoint(op):
                got = _build_regrouped(first.operands[0], op.results[0].type)
            op.results[0].replace_all_uses_with(got)
            op.operation.erase()
            first.operation.erase()


def _rewrite_into_digits(module):
    for func in funcs(module):
        _cut_func_into_digits(func)
    _erase_unused(module)


def _cut_func_into_digits(func):
    from mlir.ir import InsertionPoint, RankedTensorType
    ops = [op_of(r) for r in walk(func.operation)
           if not (op_of(r).operation.parent.name or "").startswith("linalg.")]
    for op in ops:
        tiles = [v for v in list(op.operands) + list(op.results)
                 if isinstance(v.type, RankedTensorType)]
        at_laps = AT_LAPS in op.operation.attributes
        laps = (name_of(op) == "linalg.reduce" and _folds_lanes(op)) or at_laps
        if not tiles or all(_get_form(v.type, laps).is_plain() for v in tiles):
            continue
        nm = name_of(op)
        if nm == "scf.yield" or nm == LAP_REPLICA:
            continue        # its loop's rule cuts what it yields; the lap replica waits as it is
        rule = _DIGIT_RULES.get(nm)
        if rule is None:
            raise DigitRefused("%s on a tile cut into digits" % nm)
        _CUT_LAPS[0] = at_laps
        try:
            with InsertionPoint(op):
                got = rule(op)
        finally:
            _CUT_LAPS[0] = False
        for old, new in zip(op.results, got):
            with InsertionPoint(op):
                back = _build_regrouped(new, old.type) if (
                    isinstance(old.type, RankedTensorType) and new.type != old.type) else new
            old.replace_all_uses_with(back)
        op.operation.erase()


def _only_units_differ(a, b):
    """Are `a` and `b` the same tile up to unit dims, the same element and layout bases on the rest?"""
    from mlir.ir import RankedTensorType
    a, b = RankedTensorType(a), RankedTensorType(b)
    if a.element_type != b.element_type:
        return False
    keep = lambda t: [d for d in range(t.rank) if t.shape[d] != 1]
    ka, kb = keep(a), keep(b)
    if [a.shape[d] for d in ka] != [b.shape[d] for d in kb]:
        return False
    ba = get_bases(a.encoding) if a.encoding is not None else None
    bb = get_bases(b.encoding) if b.encoding is not None else None
    if ba is None or bb is None:
        return ba is bb
    proj = lambda bases, k: [[[x[d] for d in k] for x in part] for part in bases]
    return proj(ba, ka) == proj(bb, kb)


def _fold_reshape_round_trips(module):
    """A reshape that undoes the one it reads -- the digits regrouped back, then cut again -- is the
    tile it started from, or one reshape from it when the pair only adds or drops unit dims."""
    from mlir.ir import InsertionPoint
    pair = {"tensor.expand_shape": "tensor.collapse_shape",
            "tensor.collapse_shape": "tensor.expand_shape"}
    for func in funcs(module):
        for op in [op_of(r) for r in walk(func.operation)]:
            inner = producer(op.operands[0]) if name_of(op) in pair else None
            if inner is None or name_of(inner) != pair[name_of(op)]:
                continue
            if inner.operands[0].type == op.results[0].type:
                op.results[0].replace_all_uses_with(inner.operands[0])
            elif _only_units_differ(inner.operands[0].type, op.results[0].type):
                # the pair adds or drops unit dims only: one reshape from where it started
                with InsertionPoint(op):
                    op.results[0].replace_all_uses_with(
                        _build_regrouped(inner.operands[0], op.results[0].type))
    _erase_unused(module)


def _digit_index(new_dims, sizes):
    from mlir.dialects import arith
    from mlir.ir import IndexType, IntegerAttr, Operation
    idx = IndexType.get()
    total = None
    for k, d in enumerate(new_dims):
        i = Operation.create("linalg.index", results=[idx],
                             attributes={"dim": IntegerAttr.get(IntegerType_i64(), d)}).results[0]
        scale = 1
        for x in sizes[k + 1:]:
            scale *= x
        if scale != 1:
            i = arith.MulIOp(i, arith.ConstantOp(idx, IntegerAttr.get(idx, scale)).result).result
        total = i if total is None else arith.AddIOp(total, i).result
    return total


def IntegerType_i64():
    from mlir.ir import IntegerType
    return IntegerType.get_signless(64)


def _clone_body_into_digits(src_op, dst_op, form):
    from mlir.ir import InsertionPoint, IntegerAttr, Operation
    body = src_op.regions[0].blocks[0]
    blk = dst_op.regions[0].blocks.append(*[a.type for a in body.arguments])
    env = dict(zip(body.arguments, blk.arguments))
    with InsertionPoint(blk):
        for old in body.operations:
            old = old.operation
            if old.name == "linalg.index":
                d = IntegerAttr(attrs_of(old)["dim"]).value
                env[old.results[0]] = _digit_index(form.dims_of(d), form.groups[d])
                continue
            new = Operation.create(old.name, results=[r.type for r in old.results],
                                   operands=[env.get(x, x) for x in old.operands],
                                   attributes=attrs_of(old))
            env.update(zip(old.results, new.results))


def _cut_generic(op):
    from mlir.ir import (AffineConstantExpr, AffineDimExpr, AffineMap, AffineMapAttr, ArrayAttr,
                         Attribute, Operation, RankedTensorType)
    maps, n_in = indexing_maps(op), operand_segments(op)[0]
    form = _get_form(op.results[0].type)
    rank = len(form.digits)
    if any(m != AffineMap.get_identity(m.n_dims) for m in maps[n_in:]) or any(
            _get_form(r.type) != form for r in op.results):
        raise DigitRefused("a generic whose outputs are not one tile of digits")
    iters = iterator_types(op)
    new_iters = (["#linalg.iterator_type<parallel>"] if form.lead else []) + [
        iters[i] for i in range(len(form.groups)) for _ in form.groups[i]]
    operands, new_maps = [], []
    for x, m in zip(op.operands, maps):
        if not isinstance(x.type, RankedTensorType):
            operands.append(x)
            new_maps.append(AffineMap.get(rank, 0, []))
            continue
        dims = _get_map_dims(m)
        if dims is None:
            raise DigitRefused("a generic map that is no projection")
        shape = list(RankedTensorType(x.type).shape)
        groups = [[1] if d is None else form.groups[d] for d in dims]
        if [__import__("math").prod(g) for g in groups] != shape:
            raise DigitRefused("an operand whose dims are not its op's")
        xf = _Form(shape, groups, form.lead)
        operands.append(_build_cut(x, xf))
        exprs = [AffineDimExpr.get(0)] if form.lead else []
        for d in dims:
            exprs += ([AffineConstantExpr.get(0)] if d is None
                      else [AffineDimExpr.get(k) for k in form.dims_of(d)])
        new_maps.append(AffineMap.get(rank, 0, exprs))
    attrs = attrs_of(op)
    attrs["indexing_maps"] = ArrayAttr.get([AffineMapAttr.get(m) for m in new_maps])
    attrs["iterator_types"] = ArrayAttr.get([Attribute.parse(t) for t in new_iters])
    g = Operation.create("linalg.generic", results=[_form_type(r.type, form) for r in op.results],
                         operands=operands, attributes=attrs, regions=1)
    _clone_body_into_digits(op, g, form)
    return list(g.results)


def _cut_reduce(op):
    from mlir.ir import DenseI64ArrayAttr, Operation
    n = len(op.operands) // 2
    # a fold over a lane axis longer than the lanes reduces its laps in each lane first
    form = _get_form(op.operands[0].type, laps=_folds_lanes(op))
    # A reduce that folds the lanes away leaves a replica, which stands in one bank on a unit dim
    # before its own: the input takes that unit dim too, so the reduce keeps its dims.
    if not form.lead and any(_get_form(i.type).lead for i in list(op.operands)[n:]):
        form = _Form(form.shape, form.groups, True)
    dims = sorted(int_array(op.attributes["dimensions"]))
    kept = [d for d in range(len(form.groups)) if d not in dims]
    out_form = _Form([form.shape[d] for d in kept], [form.groups[d] for d in kept], form.lead)
    xs = [_build_cut(x, _Form(form.shape, form.groups, form.lead)) for x in list(op.operands)[:n]]
    inits = [_build_cut(i, out_form) for i in list(op.operands)[n:]]
    attrs = attrs_of(op)
    digits = sorted(k for d in dims for k in form.dims_of(d))
    xs, digits = _build_in_lane_first(op, xs, digits)
    attrs["dimensions"] = DenseI64ArrayAttr.get(digits)
    r = Operation.create("linalg.reduce", results=[i.type for i in inits], operands=xs + inits,
                         attributes=attrs, regions=1)
    _clone_body(op, r, list(range(len(form.digits))))
    return list(r.results)


_FOLD_IDENTITY = {"arith.maximumf": float("-inf"), "arith.maxnumf": float("-inf"), "arith.addf": 0.0,
                  "arith.minimumf": float("inf"), "arith.minnumf": float("inf")}


def _build_in_lane_first(op, xs, digits):
    """A fold over the lanes that also reduces digits each lane holds reduces those first, in each
    lane, into a tile that keeps the lanes: what folds across them is then one element a lane per
    kept position. (xs, digits) for the reduce that is left."""
    from mlir.dialects import arith, linalg, tensor
    from mlir.ir import (Attribute, DenseI64ArrayAttr, FloatAttr, Operation, RankedTensorType)
    if len(xs) != 1:
        return xs, digits
    xt = RankedTensorType(xs[0].type)
    bases = get_bases(xt.encoding) if xt.encoding is not None else None
    if bases is None:
        return xs, digits
    lanes = {d for b in bases[1] for d, x in enumerate(b) if x}
    inner = [d for d in digits if d not in lanes]
    comb = [o.operation.name for o in op.regions[0].blocks[0].operations][0]
    if not (lanes & set(digits)) or not inner or comb not in _FOLD_IDENTITY:
        return xs, digits
    x, rest = xs[0], list(digits)
    # one digit a reduce, the last first, as a lane's walk reduces one axis at a time
    for d in sorted(inner, reverse=True):
        xt = RankedTensorType(x.type)
        regs, lns = drop_dims(get_bases(xt.encoding), {d})
        shape = [n for k, n in enumerate(xt.shape) if k != d]
        enc = Attribute.parse(spell(([b for b in regs if any(b)], lns)))
        empty = tensor.EmptyOp(shape, xt.element_type, encoding=enc).result
        ident = arith.ConstantOp(xt.element_type,
                                 FloatAttr.get(xt.element_type, _FOLD_IDENTITY[comb])).result
        attrs = attrs_of(op)
        attrs["dimensions"] = DenseI64ArrayAttr.get([d])
        step = Operation.create("linalg.reduce", results=[empty.type],
                                operands=[x, linalg.fill(ident, outs=[empty])],
                                attributes=attrs, regions=1)
        _clone_body(op, step, list(range(xt.rank)))
        x = step.results[0]
        rest = [k - (k > d) for k in rest if k != d]
    return [x], rest


def _cut_transfer(op):
    from mlir.ir import ArrayAttr, IntegerAttr, IntegerType, Operation
    tile = op.operands[2]
    form = _get_form(op.results[0].type if op.results else tile.type)
    gather = IntegerAttr(op.attributes["indirect_dim"]).value if "indirect_dim" in op.attributes else None
    if gather is not None and len(form.groups[gather]) != 1:
        raise DigitRefused("a gathering DMA whose gathered dim is cut into digits")
    if "cut" in op.attributes:
        raise DigitRefused("a wrapping or stepping DMA onto a tile of digits")
    old = [IntegerAttr(a).value for a in ArrayAttr(op.attributes["dram_stride"])]
    strides = [0] if form.lead else []
    for st, g in zip(old, form.groups):
        for k in range(len(g)):
            below = 1
            for x in g[k + 1:]:
                below *= x
            strides.append(st * below)
    attrs = attrs_of(op)
    i64 = IntegerType.get_signless(64)
    attrs["dram_stride"] = ArrayAttr.get([IntegerAttr.get(i64, x) for x in strides])
    operands = list(op.operands)
    clamp = _get_clamp_bounds_on_digits(op, form)
    for key in ("masked_axes", "dyn_stride_axes"):
        if key in attrs:
            axes = [IntegerAttr(a).value for a in ArrayAttr(attrs[key])]
            if key == "dyn_stride_axes" and any(len(form.groups[a]) != 1 for a in axes):
                raise DigitRefused("a DMA %s on a dim cut into digits" % key)
            attrs[key] = ArrayAttr.get([IntegerAttr.get(i64, form.dims_of(a)[0]) for a in axes])
    if gather is not None:
        attrs["indirect_dim"] = IntegerAttr.get(i64, form.dims_of(gather)[0])
    if clamp is not None:
        at = sum(operand_segments(op)[:6])
        operands[at:at + len(clamp)] = clamp
    operands[2] = _build_cut(tile, form)
    if operand_segments(op)[5]:
        operands[5] = _build_cut(operands[5], _get_form(operands[5].type))
    t = Operation.create("torchsim.transfer", results=[operands[2].type] if op.results else [],
                         operands=operands, attributes=attrs)
    return list(t.results)


def _get_clamp_bounds_on_digits(op, form):
    """A masked dim cut into digits keeps its mask on its outermost digit: [0, high) of the dim is
    [0, high / inner) of that digit when high is a constant that the inner digits' extent divides --
    the mask then cuts whole inner runs. The clamp operands so rewritten, None when no masked dim is
    cut; any other masked cut is refused."""
    from mlir.dialects import arith
    from mlir.ir import ArrayAttr, IndexType, IntegerAttr
    if "masked_axes" not in op.attributes:
        return None
    axes = [IntegerAttr(a).value for a in ArrayAttr(op.attributes["masked_axes"])]
    if all(len(form.groups[a]) == 1 for a in axes):
        return None
    at = sum(operand_segments(op)[:6])
    clamp = list(op.operands)[at:at + 2 * len(axes)]
    out = []
    for k, a in enumerate(axes):
        lo, hi = clamp[2 * k], clamp[2 * k + 1]
        g = form.groups[a]
        if len(g) == 1:
            out += [lo, hi]
            continue
        inner = 1
        for x in g[1:]:
            inner *= x
        c_lo, c_hi = _const_int(lo), _const_int(hi)
        if c_lo != 0 or c_hi is None or c_hi % inner:
            raise DigitRefused("a DMA masked_axes on a dim cut into digits whose bound [%s, %s) does "
                               "not cut whole runs of %d" % (c_lo, c_hi, inner))
        out += [lo, arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), c_hi // inner)).result]
    return out


def _const_int(v):
    """The integer a constant op makes `v`, or None."""
    from mlir.ir import IntegerAttr
    o = producer(v)
    if name_of(o) != "arith.constant":
        return None
    try:
        return IntegerAttr(o.attributes["value"]).value
    except (TypeError, ValueError):
        return None


def _cut_fill(op):
    from mlir.ir import Operation
    form = _get_form(op.results[0].type)
    out = _build_cut(op.operands[1], form)
    f = Operation.create("linalg.fill", results=[out.type], operands=[op.operands[0], out],
                         attributes=attrs_of(op), regions=1)
    _clone_body(op, f, [])
    return list(f.results)


def _cut_fresh(op):
    from mlir.ir import Operation
    if len(op.operands):
        raise DigitRefused("%s with operands on a tile cut into digits" % name_of(op))
    form = _get_form(op.results[0].type)
    return list(Operation.create(name_of(op), results=[_form_type(op.results[0].type, form)],
                                 attributes=attrs_of(op)).results)


def _cut_reshape(op):
    src_form, dst_form = _get_form(op.operands[0].type), _get_form(op.results[0].type)
    src = _build_cut(op.operands[0], src_form)
    return [_build_regrouped(src, _form_type(op.results[0].type, dst_form))]


def _cut_transpose(op):
    from mlir.ir import DenseI64ArrayAttr, Operation
    form = _get_form(op.operands[0].type)
    perm = list(int_array(op.attributes["permutation"]))
    out_form = _Form([form.shape[p] for p in perm], [form.groups[p] for p in perm], form.lead)
    src = _build_cut(op.operands[0], form)
    init = _build_cut(op.operands[1], out_form)
    new = ([0] if form.lead else []) + [k for p in perm for k in form.dims_of(p)]
    attrs = attrs_of(op)
    attrs["permutation"] = DenseI64ArrayAttr.get(new)
    t = Operation.create("linalg.transpose", results=[init.type], operands=[src, init],
                         attributes=attrs, regions=1)
    _clone_body(op, t, [])
    return list(t.results)


def _cut_convert(op):
    from mlir.ir import Operation
    a, b = _get_form(op.operands[0].type), _get_form(op.results[0].type)
    both = _get_common_form(a, b)
    src = _build_cut(op.operands[0], a)
    if both != a:
        src = _build_regrouped(src, _form_type(op.operands[0].type, both))
    moved = Operation.create("ttg.convert_layout", results=[_form_type(op.results[0].type, both)],
                             operands=[src]).results[0]
    if both != b:
        moved = _build_regrouped(moved, _form_type(op.results[0].type, b))
    return [moved]


def _get_common_form(a, b):
    groups = []
    for ga, gb in zip(a.groups, b.groups):
        def cuts(g):
            out, at = set(), 0
            for x in reversed(g):
                at += x.bit_length() - 1
                out.add(at)
            return out
        width = sum(x.bit_length() - 1 for x in ga)
        edges = sorted(cuts(ga) | cuts(gb) | {0})
        groups.append([1 << (edges[i + 1] - edges[i]) for i in reversed(range(len(edges) - 1))]
                      if width else list(ga))
    return _Form(a.shape, groups, a.lead or b.lead)


def _build_regrouped(v, ty):
    from mlir.dialects import arith, tensor
    from mlir.ir import (Attribute, DenseElementsAttr, IntegerType, RankedTensorType)
    import numpy as np
    a, b = list(RankedTensorType(v.type).shape), list(RankedTensorType(ty).shape)
    if a == b and v.type == ty:
        return v
    mid = _get_collapsed_shape(a, b)
    sb = get_bases(RankedTensorType(v.type).encoding)
    mid_bases = compose_reshape(sb, a, mid) if sb is not None and mid else None
    whole = (mid not in (a, b) and (mid_bases is None or isinstance(
        physical_perm(mid_bases, mid), str)))
    if whole:
        i64 = IntegerType.get_signless(64)
        shape = arith.ConstantOp(RankedTensorType.get([len(b)], i64),
                                 DenseElementsAttr.get(np.array(b, dtype=np.int64),
                                                       type=RankedTensorType.get([len(b)], i64))).result
        return tensor.ReshapeOp(ty, v, shape).result
    got = v
    if mid != a:
        enc = Attribute.parse(spell(mid_bases)) if mid_bases is not None else None
        got = tensor.CollapseShapeOp(RankedTensorType.get(mid, RankedTensorType(ty).element_type, enc),
                                     got, _get_collapse_groups(a, mid)).result
    if mid != b:
        got = tensor.ExpandShapeOp(ty, got, _get_collapse_groups(b, mid), [], b).result
    return got


def _cut_alloc(op):
    from mlir.ir import Operation
    seg = operand_segments(op)
    if seg[0] or seg[2]:
        raise DigitRefused("an alloc with dynamic sizes on a tile cut into digits")
    if not seg[1]:
        return _cut_fresh(op)
    form = _get_form(op.results[0].type)
    src = _build_cut(op.operands[0], form)
    return list(Operation.create("bufferization.alloc_tensor",
                                 results=[_form_type(op.results[0].type, form)], operands=[src],
                                 attributes=attrs_of(op)).results)


def _cut_broadcast(op):
    from mlir.ir import DenseI64ArrayAttr, Operation, RankedTensorType
    form = _get_form(op.results[0].type)
    # a replica spread across the lanes stands on the unit dim it has in one bank: the result
    # takes that unit dim too, so the broadcast keeps its dims
    if not form.lead and _get_form(op.operands[0].type).lead:
        form = _Form(form.shape, form.groups, True)
    added = sorted(int_array(op.attributes["dimensions"]))
    kept = [d for d in range(len(form.groups)) if d not in added]
    shape = list(RankedTensorType(op.operands[0].type).shape)
    src = _build_cut(op.operands[0], _Form(shape, [form.groups[d] for d in kept], form.lead))
    init = _build_cut(op.operands[1], form)
    attrs = attrs_of(op)
    attrs["dimensions"] = DenseI64ArrayAttr.get(sorted(k for d in added for k in form.dims_of(d)))
    b = Operation.create("linalg.broadcast", results=[init.type], operands=[src, init],
                         attributes=attrs, regions=1)
    _clone_body(op, b, [])
    return list(b.results)


def _get_cut_window(form, offs, sizes, strides):
    o2, s2, t2 = ([0], [1], [1]) if form.lead else ([], [], [])
    for d, g in enumerate(form.groups):
        if len(g) == 1:
            o2.append(offs[d]); s2.append(sizes[d]); t2.append(strides[d])
            continue
        if strides[d] != 1:
            return None
        if offs[d] == 0 and sizes[d] == form.shape[d]:
            o2 += [0] * len(g); s2 += list(g); t2 += [1] * len(g)
        elif offs[d] % g[-1] + sizes[d] <= g[-1]:
            below = g[-1]
            rest = offs[d] // below
            digits = []
            for x in reversed(g[:-1]):
                digits.append(rest % x)
                rest //= x
            o2 += list(reversed(digits)) + [offs[d] % below]
            s2 += [1] * (len(g) - 1) + [sizes[d]]
            t2 += [1] * len(g)
        else:
            return None
    return o2, s2, t2


def _cut_slice(op):
    from mlir.dialects import tensor
    from mlir.ir import RankedTensorType
    nm = name_of(op)
    if any(operand_segments(op)[-3:]):
        raise DigitRefused("a window with dynamic bounds on a tile cut into digits")
    big = op.results[0] if nm == "tensor.insert_slice" else op.operands[0]
    small = op.operands[0] if nm == "tensor.insert_slice" else op.results[0]
    form = _get_form(big.type)
    offs, sizes, strides = (list(int_array(op.attributes[k]))
                            for k in ("static_offsets", "static_sizes", "static_strides"))
    got = _get_cut_window(form, offs, sizes, strides)
    if got is None:
        raise DigitRefused("a window across a dim cut into digits")
    o2, s2, t2 = got
    small_shape = list(RankedTensorType(small.type).shape)
    kept_dims = [d for d in range(len(sizes)) if sizes[d] != 1 or len(small_shape) == len(sizes)]
    groups = []
    for d in kept_dims:
        g = form.groups[d]
        groups.append(list(g) if len(g) > 1 and sizes[d] == form.shape[d] else [sizes[d]])
    small_form = _Form(small_shape, groups, form.lead and len(small_shape) == len(sizes))
    small_ty = _form_type(small.type, small_form)
    if nm == "tensor.insert_slice":
        dst = _build_cut(op.operands[1], form)
        src = _build_cut(op.operands[0], small_form)
        return [tensor.InsertSliceOp(src, dst, [], [], [], o2, s2, t2).result]
    src = _build_cut(op.operands[0], form)
    return [tensor.ExtractSliceOp(small_ty, src, [], [], [], o2, s2, t2).result]


def _cut_for(op):
    """A loop that carries a tile cut into digits carries the cut: the inits are cut, the body
    sees each carried tile regrouped back, and what it yields is cut again."""
    from mlir.dialects import scf
    from mlir.ir import InsertionPoint, RankedTensorType
    lb, ub, step = op.operands[0], op.operands[1], op.operands[2]
    inits = list(op.operands)[3:]
    forms = [_get_form(v.type) if isinstance(v.type, RankedTensorType) else None for v in inits]
    new_inits = [v if f is None else _build_cut(v, f) for v, f in zip(inits, forms)]
    loop = scf.ForOp(lb, ub, step, new_inits)
    old = op.regions[0].blocks[0]
    ops = list(old.operations)
    with InsertionPoint(loop.body):
        old.arguments[0].replace_all_uses_with(loop.induction_variable)
        for a, b in zip(list(old.arguments)[1:], loop.inner_iter_args):
            a.replace_all_uses_with(b if a.type == b.type else _build_regrouped(b, a.type))
    for o in ops:
        loop.body.append(o.operation)
    yld = ops[-1].operation
    with InsertionPoint(yld):
        for k, f in enumerate(forms):
            if f is not None:
                yld.operands[k] = _build_cut(yld.operands[k], f)
    return list(loop.results)


_DIGIT_RULES = {
    "linalg.generic": _cut_generic, "linalg.reduce": _cut_reduce,
    "torchsim.transfer": _cut_transfer, "linalg.fill": _cut_fill,
    "tensor.empty": _cut_fresh, "bufferization.alloc_tensor": _cut_alloc,
    "linalg.broadcast": _cut_broadcast, "tensor.insert_slice": _cut_slice,
    "tensor.extract_slice": _cut_slice, "tensor.collapse_shape": _cut_reshape,
    "tensor.expand_shape": _cut_reshape, "tensor.reshape": _cut_reshape,
    "linalg.transpose": _cut_transpose,
    "ttg.convert_layout": _cut_convert, "scf.for": _cut_for,
}


def _rewrite_replica_broadcast_op(op):
    from mlir.ir import RankedTensorType
    x, res = op.operands[0], op.results[0]
    out_ty = RankedTensorType(res.type)
    out_bases = get_bases(out_ty.encoding) if out_ty.encoding is not None else None
    if out_bases is None or not _is_linear_replica(x.type):
        return
    lanes = {d for b in out_bases[1] for d, v in enumerate(b) if v}
    added = set(int_array(op.attributes["dimensions"]))
    kept = [d for d in range(out_ty.rank) if d not in added]
    if not lanes or lanes <= set(kept):
        return
    ty = RankedTensorType.get(list(out_ty.shape), RankedTensorType(x.type).element_type,
                              out_ty.encoding)
    got = _build_replica_in_shape(x, ty, kept, op, {})
    if got is None:
        return              # left a broadcast, which the plan carries across the lanes
    res.replace_all_uses_with(got)
    op.operation.erase()


def _build_converted_wide(conv, ty, kept, at):
    from mlir.dialects import linalg, tensor
    from mlir.ir import Attribute, DenseI64ArrayAttr, InsertionPoint, RankedTensorType
    x = conv.operands[0]
    xt = RankedTensorType(x.type)
    real = [d for d in kept if d is not None]
    if real != sorted(real):
        return None
    units = [k for k, d in enumerate(kept) if d is None]
    with InsertionPoint(at):
        if units:
            groups, cur = [], []
            for k in range(xt.rank):
                cur.append(k)
                if kept[k] is not None:
                    groups.append(cur)
                    cur = []
            if cur:
                if not groups:
                    return None
                groups[-1] += cur
            bases = drop_dims(get_bases(xt.encoding), units)
            narrow = RankedTensorType.get([xt.shape[k] for k in range(xt.rank) if k not in units],
                                          xt.element_type, Attribute.parse(spell(bases)))
            x = tensor.CollapseShapeOp(narrow, x, groups).result
        init = tensor.EmptyOp(list(ty.shape), ty.element_type, encoding=ty.encoding).result
        wide = linalg.BroadcastOp([ty], x, init, DenseI64ArrayAttr.get(
            [d for d in range(ty.rank) if d not in real]))
    blk = wide.regions[0].blocks.append(xt.element_type, ty.element_type)
    with InsertionPoint(blk):
        linalg.YieldOp([blk.arguments[0]])
    return wide.results[0]


def _build_reduced_wide(red, ty, kept, at, memo):
    from mlir.ir import Attribute, DenseI64ArrayAttr, InsertionPoint, Operation, RankedTensorType
    if len(red.operands) != 2:
        return None
    x, init = red.operands[0], red.operands[1]
    xt = RankedTensorType(x.type)
    dims = sorted(int_array(red.attributes["dimensions"]))
    x_kept_dims = [d for d in range(xt.rank) if d not in dims]
    shape = list(ty.shape)
    wide_shape = shape + [xt.shape[d] for d in dims]
    x_kept = [None] * xt.rank
    for j, d in enumerate(x_kept_dims):
        x_kept[d] = kept[j]
    for j, d in enumerate(dims):
        x_kept[d] = len(shape) + j
    regs, lanes = get_bases(ty.encoding)
    pad = lambda b: list(b) + [0] * len(dims)
    new_regs = [pad(b) for b in regs]
    for j, d in enumerate(dims):
        bit = 1
        while bit < xt.shape[d]:
            new_regs.append([0] * len(shape) + [bit if k == j else 0 for k in range(len(dims))])
            bit *= 2
    wide_ty = RankedTensorType.get(wide_shape, xt.element_type,
                                   Attribute.parse(spell((new_regs, [pad(b) for b in lanes]))))
    wide_x = _build_replica_in_shape(x, wide_ty, x_kept, at, memo)
    wide_init = _build_replica_in_shape(init, RankedTensorType.get(
        shape, RankedTensorType(init.type).element_type, ty.encoding), kept, at, memo)
    if wide_x is None or wide_init is None:
        return None
    attrs = attrs_of(red)
    attrs["dimensions"] = DenseI64ArrayAttr.get(list(range(len(shape), len(wide_shape))))
    with InsertionPoint(at):
        r = Operation.create("linalg.reduce", results=[wide_init.type], operands=[wide_x, wide_init],
                             attributes=attrs, regions=1)
    _clone_body(red, r, list(range(len(wide_shape))))
    return r.results[0]

def _is_linear_replica(ty):
    from mlir.ir import RankedTensorType
    ty = RankedTensorType(ty)
    bases = get_bases(ty.encoding) if ty.encoding is not None else None
    return bases is not None and not any(any(b) for b in bases[1]) and ty.rank > 0


def _get_map_dims(m):
    from mlir.ir import AffineConstantExpr, AffineDimExpr
    out = []
    for e in m.results:
        e = e.maybe_downcast()
        if isinstance(e, AffineDimExpr):
            out.append(e.position)
        elif isinstance(e, AffineConstantExpr) and e.value == 0:
            out.append(None)
        else:
            return None
    return out


_REPLICA_MISSES = []


def _build_replica_in_shape(v, ty, kept, at, memo):
    key = (v, tuple(kept))
    if key not in memo:
        memo[key] = _build_replica_by_producer(v, ty, kept, at, memo)
        if memo[key] is None:
            o = producer(v)
            _REPLICA_MISSES.append(str(o).split("\n")[0][:300] if o is not None
                                   else "a block argument")
    return memo[key]


def _build_replica_by_producer(v, ty, kept, at, memo):
    from mlir.dialects import tensor
    from mlir.ir import AffineMap, AffineMapAttr, ArrayAttr, InsertionPoint, Operation, RankedTensorType
    o = producer(v)
    shape, enc = list(ty.shape), ty.encoding
    like = lambda x: RankedTensorType.get(shape, RankedTensorType(x.type).element_type, enc)
    nm = name_of(o)
    if nm == "bufferization.alloc_tensor" and operand_segments(o)[1]:
        from mlir.dialects import bufferization
        seg = operand_segments(o)
        if seg[0]:
            return None
        src = _build_replica_in_shape(o.operands[seg[0]], ty, kept, at, memo)
        if src is None:
            return None
        with InsertionPoint(at):
            return bufferization.AllocTensorOp(ty, [], copy=src).result
    if nm in ("tensor.empty", "bufferization.alloc_tensor"):
        with InsertionPoint(at):
            return tensor.EmptyOp(shape, ty.element_type, encoding=enc).result
    if nm == "linalg.fill":
        init = _build_replica_in_shape(o.operands[1], ty, kept, at, memo)
        if init is None:
            return None
        with InsertionPoint(at):
            got = Operation.create("linalg.fill", results=[ty], operands=[o.operands[0], init],
                                   attributes=attrs_of(o), regions=1)
        _clone_body(o, got, kept)
        return got.results[0]
    if nm == "tensor.collapse_shape":
        inner = _get_unit_expand_dims(o, through=o.operands[0])
        if inner is None:
            return _build_replica_through_collapse(o, ty, kept, at, memo)
        return _build_replica_in_shape(
            o.operands[0], ty, [kept[inner.index(d)] if d in inner else None
                                for d in range(len(o.operands[0].type.shape))], at, memo)
    if nm == "tensor.expand_shape":
        wide = _get_unit_expand_dims(o)
        if wide is None:
            return None
        return _build_replica_in_shape(o.operands[0], ty, [kept[d] for d in wide], at, memo)
    if nm == "linalg.transpose":
        perm = int_array(o.attributes["permutation"])
        return _build_replica_in_shape(o.operands[0], ty, [kept[perm.index(j)]
                                                           for j in range(len(perm))], at, memo)
    if nm == "linalg.generic":
        maps = indexing_maps(o)
        if len(o.results) != 1 or maps[-1] != AffineMap.get_identity(maps[-1].n_dims) or any(
                _get_map_dims(m) is None for m in maps) or not all(
                "parallel" in t for t in iterator_types(o)):
            return None
        operands = []
        for x, m in zip(o.operands, maps):
            md = _get_map_dims(m)
            y = _build_replica_in_shape(x, like(x), [None if d is None else kept[d] for d in md],
                                        at, memo) if _is_tensor(x) else x
            if y is None:
                return None
            operands.append(y)
        attrs = attrs_of(o)
        attrs["indexing_maps"] = ArrayAttr.get(
            [AffineMapAttr.get(AffineMap.get_identity(len(shape)))] * len(o.operands))
        attrs["iterator_types"] = ArrayAttr.get([parallel_iterator()] * len(shape))
        with InsertionPoint(at):
            g = Operation.create("linalg.generic", results=[ty], operands=operands,
                                 attributes=attrs, regions=1)
        _clone_body(o, g, kept)
        return g.results[0]
    if nm == "torchsim.transfer":
        return _build_replicated_load(o, ty, kept, at, memo)
    if nm == "ttg.convert_layout" and _is_linear_replica(o.operands[0].type):
        return _build_replica_in_shape(o.operands[0], ty, kept, at, memo)
    if nm == "ttg.convert_layout" and len(kept) == len(o.operands[0].type.shape):
        return _build_converted_wide(o, ty, kept, at)
    if nm == "linalg.reduce":
        # a reduce that folds the lanes is no per-lane work: rebuilt in each lane, every lane would
        # redo the whole input; its result is spread from the fold instead
        if _folds_lanes(o):
            return _build_spread_fold(v, ty, kept, at)
        return _build_reduced_wide(o, ty, kept, at, memo)
    return None


def _folds_lanes(red):
    """Does this linalg.reduce fold an axis its input lays across the lanes?"""
    from mlir.ir import RankedTensorType
    x = RankedTensorType(red.operands[0].type)
    bases = get_bases(x.encoding) if x.encoding is not None else None
    if bases is None:
        return False
    dims = set(int_array(red.attributes["dimensions"]))
    return any(b[d] for b in bases[1] for d in dims)


def _build_spread_fold(x, ty, kept, at):
    """A replica in `ty`'s shape: one linalg.broadcast, which the plan carries as a broadcast across
    the lanes, after a collapse of the unit dims `ty` does not walk (None in `kept`); None when
    `x`'s other dims do not stand in `ty`'s order."""
    from mlir.dialects import linalg, tensor
    from mlir.ir import Attribute, DenseI64ArrayAttr, InsertionPoint, RankedTensorType
    elt = ty.element_type
    if None in kept:
        xt = RankedTensorType(x.type)
        keep = [i for i, k in enumerate(kept) if k is not None]
        if any(xt.shape[i] != 1 for i, k in enumerate(kept) if k is None):
            return None
        groups, run = [], []
        for i, k in enumerate(kept):
            run.append(i)
            if k is not None:
                groups.append(run)
                run = []
        if run and groups:
            groups[-1] += run
        enc = None
        if keep:
            regs, lanes = get_bases(xt.encoding)
            proj = lambda b: [b[i] for i in keep]
            enc = Attribute.parse(spell((list(map(proj, regs)), list(map(proj, lanes)))))
        with InsertionPoint(at):
            x = tensor.CollapseShapeOp(RankedTensorType.get([xt.shape[i] for i in keep], elt, enc),
                                       x, groups).result
        kept = [kept[i] for i in keep]
    if kept != sorted(kept):
        return None
    with InsertionPoint(at):
        empty = tensor.EmptyOp(list(ty.shape), elt, encoding=ty.encoding).result
        wide = linalg.BroadcastOp([ty], x, empty, DenseI64ArrayAttr.get(
            [d for d in range(len(ty.shape)) if d not in kept]))
    blk = wide.regions[0].blocks.append(elt, elt)
    with InsertionPoint(blk):
        linalg.YieldOp([blk.arguments[0]])
    return wide.results[0]


def _build_replica_through_collapse(o, ty, kept, at, memo):
    """A replica of a collapse that merges dims: the source's replica in the wide shape with
    each kept dim split as the collapse splits it, collapsed back."""
    import math
    from mlir.ir import (Attribute, ArrayAttr, InsertionPoint, IntegerAttr, IntegerType, Operation,
                         RankedTensorType)
    src = o.operands[0]
    s_shape, shape = list(src.type.shape), list(ty.shape)
    groups = [[IntegerAttr(d).value for d in ArrayAttr(g)] for g in o.attributes["reassociation"]]
    bases = get_bases(ty.encoding)
    if bases is None:
        return None
    split = {}
    for j, g in enumerate(groups):
        if kept[j] is None:
            if any(s_shape[d] != 1 for d in g):
                return None
            continue
        if math.prod(s_shape[d] for d in g) != shape[kept[j]]:
            return None
        split[kept[j]] = g
    wide, where, regroup = [], {}, []
    for e, n in enumerate(shape):
        start = len(wide)
        for d in split.get(e, [None]):
            if d is not None:
                where[d] = len(wide)
            wide.append(n if d is None else s_shape[d])
        regroup.append(list(range(start, len(wide))))
    wide_ty = RankedTensorType.get(wide, ty.element_type,
                                   Attribute.parse(spell(compose_reshape(bases, shape, wide))))
    got = _build_replica_in_shape(src, wide_ty, [where.get(d) for d in range(len(s_shape))],
                                  at, memo)
    if got is None:
        return None
    i64 = IntegerType.get_signless(64)
    attrs = {"reassociation": ArrayAttr.get([ArrayAttr.get([IntegerAttr.get(i64, d) for d in g])
                                             for g in regroup])}
    with InsertionPoint(at):
        return Operation.create("tensor.collapse_shape", results=[ty], operands=[got],
                                attributes=attrs).results[0]


def _build_replicated_load(o, ty, kept, at, memo):
    from mlir.ir import ArrayAttr, DenseI64ArrayAttr, InsertionPoint, IntegerAttr, IntegerType, Operation
    seg = operand_segments(o)
    if len(o.results) != 1 or any(seg[7:]) or any(
            a in o.attributes for a in ("dyn_stride_axes", "indirect_dim", "subtile_size")):
        return None
    masked = [IntegerAttr(a).value for a in ArrayAttr(o.attributes["masked_axes"])] \
        if "masked_axes" in o.attributes else []
    if any(kept[a] is None for a in masked):
        return None
    cut = [list(int_array(c)) for c in ArrayAttr(o.attributes["cut"])] if "cut" in o.attributes else []
    if any(c and kept[k] is None for k, c in enumerate(cut)):
        return None
    init = _build_replica_in_shape(o.operands[2], ty, kept, at, memo)
    if init is None:
        return None
    i64 = IntegerType.get_signless(64)
    old = [IntegerAttr(a).value for a in ArrayAttr(o.attributes["dram_stride"])]
    strides = [0] * len(ty.shape)
    for k, d in enumerate(kept):
        if d is not None:
            strides[d] = old[k]
    attrs = attrs_of(o)
    attrs["dram_stride"] = ArrayAttr.get([IntegerAttr.get(i64, x) for x in strides])
    if masked:
        attrs["masked_axes"] = ArrayAttr.get([IntegerAttr.get(i64, kept[a]) for a in masked])
    if cut:
        moved = [[] for _ in ty.shape]
        for k, c in enumerate(cut):
            if c:
                moved[kept[k]] = c
        attrs["cut"] = ArrayAttr.get([DenseI64ArrayAttr.get(c) for c in moved])
    operands = list(o.operands)
    operands[2] = init
    if seg[5]:
        from mlir.ir import RankedTensorType
        index = o.operands[sum(seg[:5])]
        elt = RankedTensorType(index.type).element_type
        index = _build_replica_in_shape(
            index, RankedTensorType.get(list(ty.shape), elt, ty.encoding), kept, at, memo)
        if index is None:
            return None
        operands[sum(seg[:5])] = index
    with InsertionPoint(at):
        return Operation.create("torchsim.transfer", results=[ty], operands=operands,
                                attributes=attrs).results[0]


def _clone_body(src_op, dst_op, kept):
    from mlir.ir import IndexType, InsertionPoint, IntegerAttr, Operation
    body = src_op.regions[0].blocks[0]
    blk = dst_op.regions[0].blocks.append(*[a.type for a in body.arguments])
    env = dict(zip(body.arguments, blk.arguments))
    with InsertionPoint(blk):
        for old in body.operations:
            old = old.operation
            attrs = attrs_of(old)
            if old.name == "linalg.index":
                d = kept[IntegerAttr(attrs["dim"]).value]
                if d is None:
                    new = Operation.create("arith.constant", results=[IndexType.get()],
                                           attributes={"value": IntegerAttr.get(IndexType.get(), 0)})
                    env.update(zip(old.results, new.results))
                    continue
                attrs["dim"] = IntegerAttr.get(IntegerAttr(attrs["dim"]).type, d)
            new = Operation.create(old.name, results=[r.type for r in old.results],
                                   operands=[env.get(x, x) for x in old.operands], attributes=attrs)
            env.update(zip(old.results, new.results))


kPure = ("tensor.empty", "bufferization.alloc_tensor", "linalg.fill", "linalg.generic",
         "tensor.collapse_shape", "tensor.expand_shape", "tensor.reshape", "tensor.cast",
         "arith.constant", "ttg.convert_layout", "linalg.broadcast", "linalg.transpose",
         "linalg.reduce")


def _erase_unused(module):
    while True:
        dead = [op_of(r) for func in funcs(module) for r in walk(func.operation)
                if op_of(r).results and not any(list(x.uses) for x in op_of(r).results)
                and (name_of(op_of(r)) in kPure or (name_of(op_of(r)) == "torchsim.transfer"
                     and len(op_of(r).results) == 1))]
        if not dead:
            return
        for op in reversed(dead):
            op.operation.erase()


def _get_reshape_plan_physical(op, perms, wanted):
    from mlir.ir import RankedTensorType
    src, res = op.operands[0], op.results[0]
    ps, pr = _get_effective(op, 0, perms, wanted), perms.get(res)
    if ps is None or pr is None:
        return None
    s_ty, d_ty = RankedTensorType(src.type), RankedTensorType(res.type)
    if (s_ty.rank == 0 or d_ty.rank == 0) and all(x == 1 for x in list(s_ty.shape) + list(d_ty.shape)):
        return None, None
    have, want = get_bases(s_ty.encoding), get_bases(d_ty.encoding)
    if have is None or want is None:
        return "a reshape with an end that is no linear layout"
    if compose_reshape(have, list(s_ty.shape), list(d_ty.shape)) != want:
        return "a reshape whose result is not laid out as its source composed through it: %s" % (
            str(op).split("\n")[0][:600])
    return None, None


def _get_unit_expand_dims(op, through=None):
    from mlir.ir import ArrayAttr, IntegerAttr
    shape = list((op.results[0] if through is None else through).type.shape)
    kept = []
    for group in op.attributes["reassociation"]:
        dims = [IntegerAttr(d).value for d in ArrayAttr(group)]
        big = [d for d in dims if shape[d] != 1]
        if len(big) > 1:
            return None
        kept.append(big[0] if big else dims[-1])
    return kept


def _rewrite_type_up_the_chain(v, ty):
    from mlir.ir import OpResult
    while isinstance(v, OpResult):
        old = v.type
        v.set_type(ty)
        o = op_of(v.owner)
        if name_of(o) != "tensor.insert_slice" or o.operands[1].type != old:
            break
        v = o.operands[1]


def _collect_tensor_values(module):
    from mlir.ir import RankedTensorType
    out = []
    for func in funcs(module):
        for raw in walk(func.operation):
            op = op_of(raw)
            for r in op.results:
                if isinstance(r.type, RankedTensorType):
                    out.append(r)
            for region in op.regions:
                for block in region.blocks:
                    for a in block.arguments:
                        if isinstance(a.type, RankedTensorType):
                            out.append(a)
    return out


def _is_carried_in_one_order(op, perms, wanted):
    body = op.regions[0].blocks[0]
    yielded = op_of(list(body.operations)[-1])
    for j, res in enumerate(op.results):
        if not _is_tensor(res):
            continue
        orders = [_get_effective(op, FOR_INITS + j, perms, wanted),
                  perms.get(body.arguments[FOR_ARGS + j]),
                  _get_effective(yielded, j, perms, wanted), perms.get(res)]
        if any(o is None or o != orders[0] for o in orders):
            return False
    return True


def _get_while_carried_orders(op, perms, wanted):
    before, after = op.regions[0].blocks[0], op.regions[1].blocks[0]
    yielded = op_of(list(after.operations)[-1])
    got = {}
    for j, init in enumerate(op.operands):
        if not _is_tensor(init):
            continue
        order = _get_effective(op, j, perms, wanted)
        if order is None or _get_effective(yielded, j, perms, wanted) != order:
            return None
        got[before.arguments[j]] = order
    for k, fwd in enumerate(get_condition_forwarded(op)):
        if not _is_tensor(fwd):
            continue
        order = got[fwd] if fwd in got else perms.get(fwd)
        if order is None or perms.get(op.results[k], order) != order:
            return None
        got[after.arguments[k]] = order
        got[op.results[k]] = order
    return got


def _is_if_carried_in_one_order(op, perms, wanted):
    arms = [op_of(list(r.blocks[0].operations)[-1]) for r in op.regions
            if r.blocks and len(r.blocks[0].operations)]
    for j, res in enumerate(op.results):
        if not _is_tensor(res):
            continue
        orders = [_get_effective(y, j, perms, wanted) for y in arms]
        orders.append(perms.get(res))
        if any(o is None or o != orders[0] for o in orders):
            return False
    return True


def _is_slice_a_view(op, perms, wanted):
    extracting = name_of(op) == "tensor.extract_slice"
    big, small = _get_slice_ends(op)
    pb = _get_effective(op, 0, perms, wanted) if extracting else perms.get(big)
    ps = perms.get(small) if extracting else _get_effective(op, 0, perms, wanted)
    offs, sizes, strides = (list(int_array(op.attributes[n])) for n in (
        "static_offsets", "static_sizes", "static_strides"))
    kept = get_kept_axes(sizes, list(small.type.shape))
    if pb is None or ps is None or kept is None:
        return False
    shape = list(big.type.shape)
    lane = pb[0] if pb else None
    if lane is not None and shape[lane] != 1 and (
            offs[lane] != 0 or sizes[lane] != shape[lane] or strides[lane] != 1):
        return False
    return [kept.index(a) for a in pb if a in kept] == ps


def _is_window_static(op):
    from mlir.ir import ShapedType
    return all(v != ShapedType.get_dynamic_size() for n in (
        "static_offsets", "static_sizes", "static_strides") for v in int_array(op.attributes[n]))


def _apply_plan(module, plan):
    perms, copies, fixes = plan
    shapes = [op.operands[1] for op, _ in fixes if name_of(op) == "tensor.reshape"]
    _rewrite_copies(module, perms, copies, fixes)
    named = _find_named_attrs(module, perms)
    _rewrite_types(module, perms, named)
    _rewrite_reshapes(perms, fixes, shapes)
    _rewrite_axes(module, perms, named)


def _rewrite_copies(module, perms, copies, fixes):
    for op, slot, want in copies:
        _rewrite_operand_order(op_of(op), slot, perms, want)
    for op, (side, order) in fixes:
        if side == "copy":
            _rewrite_as_copy(op, perms)
    held = [op_of(raw) for func in funcs(module) for raw in walk(func.operation)
            if _get_copied(op_of(raw)) is not None]
    for op in held:
        _rewrite_copied(op, perms)
    for func in funcs(module):
        for raw in walk(func.operation):
            if len(op_of(raw).results):
                _rewrite_destination(op_of(raw), perms)


def _rewrite_operand_order(op, slot, perms, want):
    from mlir.dialects import linalg, tensor
    from mlir.ir import InsertionPoint
    src = op.operands[slot]
    have = perms[src]
    if have == want:
        return
    with InsertionPoint(op):
        init = tensor.EmptyOp(list(src.type.shape), src.type.element_type)
        moved = linalg.transpose(src, outs=[init.result],
                                 permutation=list(range(len(have))))
    perms[init.result] = want
    perms[moved.results[0]] = want
    op.operands[slot] = moved.results[0]


def _rewrite_as_copy(op, perms):
    from mlir.dialects import bufferization
    from mlir.ir import InsertionPoint
    src, res = op.operands[0], op.results[0]
    inserting = name_of(op) == "tensor.insert_slice"
    with InsertionPoint(op):
        dst = (op.operands[1] if inserting
               else bufferization.AllocTensorOp(res.type, []).result)
    window = {}
    if name_of(op) in kSlice:
        window = {n: op.attributes[n] for n in (
            "static_offsets", "static_sizes", "static_strides")}
    got = _build_copy_op(op, src, dst, perms[src], perms[res], into_window=inserting,
                         **window)
    perms[got] = perms.pop(res)
    perms.setdefault(dst, perms[got])
    res.replace_all_uses_with(got)
    op.operation.erase()


def _build_copy_op(op, src, dst, src_perm, dst_perm, **window):
    from mlir.ir import InsertionPoint
    with InsertionPoint(op):
        return torchsim.CopyOp(dst.type, src, dst, src_perm, dst_perm, **window).results[0]


def _rewrite_copied(op, perms):
    from mlir.dialects import bufferization, linalg
    from mlir.ir import InsertionPoint
    src, res = op.operands[0], op.results[0]
    want = perms.get(res)
    if want is None or perms.get(src) == want:
        return
    maker = op_of(src.owner) if hasattr(src, "result_number") else None
    if maker is not None and _has_no_tile_input(maker) and len(maker.results) == 1:
        with InsertionPoint(op):
            again = maker.clone().results[0]
        perms[again] = want
        del perms[res]
        res.replace_all_uses_with(again)
        op.operation.erase()
        return
    with InsertionPoint(op):
        init = bufferization.AllocTensorOp(res.type, [])
        moved = linalg.transpose(src, outs=[init.result],
                                 permutation=list(range(len(want))))
    perms[init.result] = want
    perms[moved.results[0]] = want
    del perms[res]
    res.replace_all_uses_with(moved.results[0])
    op.operation.erase()


def _has_no_tile_input(op):
    from mlir.ir import RankedTensorType
    if name_of(op) == "linalg.fill":
        return True
    if name_of(op) != "linalg.generic":
        return False
    return not any(isinstance(v.type, RankedTensorType)
                   for v in list(op.operands)[:operand_segments(op)[0]])


def _rewrite_destination(op, perms):
    from mlir.ir import InsertionPoint
    for r, res in enumerate(op.results):
        slot = _get_tied_init(op, r)
        want = perms.get(res)
        if slot is None or want is None:
            continue
        init = op.operands[slot]
        if perms.get(init) == want:
            continue
        maker = op_of(init.owner) if hasattr(init, "owner") else None
        if maker is not None and _is_fresh(maker):
            with InsertionPoint(op):
                fresh = maker.clone()
            perms[fresh.results[0]] = want
            op.operands[slot] = fresh.results[0]
        else:
            _rewrite_operand_order(op, slot, perms, want)


def _find_named_attrs(module, perms):
    named = []
    for func in funcs(module):
        for raw in walk(func.operation):
            got = _get_named_attrs(op_of(raw), perms, {})
            if got:
                named.append((op_of(raw), got))
    return named


def _rewrite_types(module, perms, named):
    for func in funcs(module):
        for raw in walk(func.operation):
            if name_of(op_of(raw)) == torchsim.TransferOp.OPERATION_NAME:
                _rewrite_transfer(op_of(raw), perms)
    for op, got in named:
        _rewrite_named_attrs(op, got)
    for value, p in perms.items():
        if p and p != list(range(len(p))):
            value.set_type(_get_permuted(value.type, p))
    for func in funcs(module):
        for raw in walk(func.operation):
            if name_of(op_of(raw)) in ("linalg.generic", "torchsim.dot"):
                _rewrite_maps(op_of(raw), perms)


def _rewrite_transfer(op, perms):
    from mlir.ir import ShapedType
    tx = op.operation.opview
    tile = op.results[0] if len(op.results) else tx.sram
    p = perms.get(tile)
    if not p:
        return
    torchsim.renumber_per_axis(op.operation, [[a] for a in p], list(ShapedType(tile.type).shape))


def _rewrite_named_attrs(op, got):
    from mlir.ir import DenseI64ArrayAttr
    for name, vals in got.items():
        op.attributes[name] = DenseI64ArrayAttr.get(list(vals))


def _get_permuted(ty, perm):
    from mlir.ir import RankedTensorType
    shape = list(ty.shape)
    return RankedTensorType.get([shape[a] for a in perm], ty.element_type)


def _rewrite_maps(op, perms):
    from mlir.ir import AffineMap
    maps = indexing_maps(op)
    if maps is None and name_of(op) == "torchsim.dot":
        maps = _get_matmul_maps()
    out = []
    for k, m in enumerate(maps):
        v = op.operands[k]
        p = perms.get(v) if _is_tensor(v) else None
        if not p or len(p) != len(m.results):
            out.append(m)
            continue
        res = list(m.results)
        out.append(AffineMap.get(m.n_dims, m.n_symbols, [res[a] for a in p]))
    set_indexing_maps(op, out)


def _get_matmul_maps():
    from mlir.ir import AffineDimExpr, AffineMap
    m, n, k = (AffineDimExpr.get(i) for i in range(3))
    return [AffineMap.get(3, 0, [m, k]), AffineMap.get(3, 0, [k, n]),
            AffineMap.get(3, 0, [m, n])]


def _rewrite_reshapes(perms, fixes, shapes):
    for op, (side, _order) in fixes:
        if side == "copy" or name_of(op) not in kReshape:
            continue
        if name_of(op) == "tensor.reshape":
            _rewrite_reshape_shape(op)
        else:
            _rewrite_reassociation(op, perms)
    seen = []
    for v in shapes:
        if any(v == w for w in seen):
            continue
        seen.append(v)
        if not list(v.uses) and name_of(op_of(v.owner)) == "arith.constant":
            op_of(v.owner).erase()


def _rewrite_reshape_shape(op):
    from mlir.dialects import arith
    from mlir.ir import Attribute, InsertionPoint
    dst = list(op.results[0].type.shape)
    ty = op.operands[1].type
    with InsertionPoint(op):
        op.operands[1] = arith.ConstantOp(ty, Attribute.parse(
            "dense<%s> : %s" % (dst, ty))).result


def _rewrite_reassociation(op, perms):
    from mlir.dialects import tensor
    from mlir.ir import InsertionPoint, RankedTensorType
    src, res = op.operands[0], op.results[0]
    s_shape, d_shape = list(src.type.shape), list(res.type.shape)
    mid = _get_collapsed_shape(s_shape, d_shape)
    got = src
    with InsertionPoint(op):
        if mid != s_shape:
            got = tensor.CollapseShapeOp(RankedTensorType.get(mid, res.type.element_type), got,
                                         _get_collapse_groups(s_shape, mid)).result
        if mid != d_shape:
            got = tensor.ExpandShapeOp(res.type, got, _get_collapse_groups(d_shape, mid), [],
                                       d_shape).result
    want = perms.pop(res, None)
    if want is not None and (mid != s_shape or mid != d_shape):
        perms[got] = want
    res.replace_all_uses_with(got)
    op.operation.erase()


def _get_collapsed_shape(a, b):
    if not a or not b:
        return []
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        while x != y and i < len(a) and j < len(b):
            if x < y:
                i += 1
                x *= a[i] if i < len(a) else 1
            else:
                j += 1
                y *= b[j] if j < len(b) else 1
        if x == y:
            out.append(x)
        i, j = i + 1, j + 1
    return out


def _get_collapse_groups(big, small):
    if not big or not small:
        return []
    groups, i = [[] for _ in small], 0
    for j, want in enumerate(small):
        if i == len(big):
            break
        n = big[i]
        while n < want:
            groups[j].append(i)
            i += 1
            n *= big[i]
        if n == want:
            groups[j].append(i)
            i += 1
            if j == len(small) - 1 or small[j + 1] != 1:
                while i < len(big) and big[i] == 1:
                    groups[j].append(i)
                    i += 1
    return groups


def _rewrite_axes(module, perms, named):
    for func in funcs(module):
        for raw in walk(func.operation):
            if name_of(op_of(raw)) in kSlice + kElement:
                _rewrite_window(op_of(raw), perms)
    for op, got in named:
        _absorb_identity_transpose(op, got)
    held = [op_of(raw) for func in funcs(module) for raw in walk(func.operation)
            if name_of(op_of(raw)) in ("linalg.transpose", "linalg.broadcast")]
    for op in held:
        if name_of(op) == "linalg.transpose":
            _rewrite_transpose(op)
        else:
            _rewrite_broadcast_across_lanes(op)


def _rewrite_window(op, perms):
    from mlir.ir import DenseI64ArrayAttr, ShapedType
    nm = name_of(op)
    if nm in kElement:
        tile = op.operands[0] if nm == "tensor.extract" else op.results[0]
        p, base = perms.get(tile), (1 if nm == "tensor.extract" else 2)
        if p:
            old = list(op.operands)[base:]
            for i, a in enumerate(p):
                op.operands[base + i] = old[a]
        return
    p = perms.get(_get_slice_ends(op)[0])
    if not p:
        return
    dyn = ShapedType.get_dynamic_size()
    cursor = 1 if nm == "tensor.extract_slice" else 2
    for name in ("static_offsets", "static_sizes", "static_strides"):
        vals = list(int_array(op.attributes[name]))
        n_dyn = sum(1 for v in vals if v == dyn)
        ops = list(op.operands)[cursor:cursor + n_dyn]
        entries, k = [], 0
        for v in vals:
            if v == dyn:
                entries.append((v, ops[k]))
                k += 1
            else:
                entries.append((v, None))
        entries = [entries[a] for a in p]
        op.attributes[name] = DenseI64ArrayAttr.get([v for v, _o in entries])
        for j, got in enumerate(o for _v, o in entries if o is not None):
            op.operands[cursor + j] = got
        cursor += n_dyn


def _absorb_identity_transpose(op, got):
    perm = got.get("permutation")
    if name_of(op) != "linalg.transpose" or perm != list(range(len(perm))):
        return
    op.results[0].replace_all_uses_with(op.operands[0])
    op.operation.erase()


def _rewrite_transpose(op):
    perm = list(int_array(op.attributes["permutation"]))
    src, init, res = op.operands[0], op.operands[1], op.results[0]
    s_shape, d_shape = list(src.type.shape), list(res.type.shape)
    if perm[0] == 0 or (s_shape[0] == 1 and d_shape[0] == 1):
        got = _build_in_bank_copy(op, perm, src, init)
    else:
        got = _build_copy_op(op, src, init, list(range(len(perm))), perm)
    res.replace_all_uses_with(got)
    op.operation.erase()


def _build_in_bank_copy(op, perm, src, init):
    from mlir.dialects import linalg
    from mlir.ir import (AffineMap, AffineMapAttr, ArrayAttr,
                         DenseI32ArrayAttr, InsertionPoint, Operation)
    inverse = [0] * len(perm)
    for d, a in enumerate(perm):
        inverse[a] = d
    maps = ArrayAttr.get([
        AffineMapAttr.get(AffineMap.get_permutation(inverse)),
        AffineMapAttr.get(AffineMap.get_identity(len(perm)))])
    with InsertionPoint(op):
        gen = Operation.create(
            "linalg.generic", results=[init.type], operands=[src, init],
            attributes={"indexing_maps": maps,
                        "iterator_types": ArrayAttr.get(
                            [parallel_iterator()] * len(perm)),
                        "operandSegmentSizes": DenseI32ArrayAttr.get([1, 1])},
            regions=1)
    blk = gen.regions[0].blocks.append(src.type.element_type,
                                       init.type.element_type)
    with InsertionPoint(blk):
        linalg.YieldOp([blk.arguments[0]])
    return gen.results[0]


def _rewrite_broadcast_across_lanes(op):
    added = set(int_array(op.attributes["dimensions"]))
    src, res = op.operands[0], op.results[0]
    d_shape, s_shape = list(res.type.shape), list(src.type.shape)
    if not d_shape or 0 not in added:
        return
    if d_shape[0] == 1 and (not s_shape or s_shape[0] == 1):
        return
    got = _build_again_in_shape(
        src, res.type, [d for d in range(len(d_shape)) if d not in added], op)
    if got is None:
        got = _build_broadcast_across_lanes(op, src, op.operands[1], added)
    res.replace_all_uses_with(got)
    op.operation.erase()


def _build_again_in_shape(v, ty, kept, at):
    from mlir.dialects import linalg, tensor
    from mlir.ir import (AffineMap, AffineMapAttr, ArrayAttr, DenseI32ArrayAttr,
                         InsertionPoint, IntegerAttr, Operation,
                         RankedTensorType)
    o = producer(v)
    shape = list(ty.shape)
    if name_of(o) == "tensor.collapse_shape":
        inner = _get_unit_expand_dims(o, through=o.operands[0])
        if inner is None:
            return None
        return _build_again_in_shape(
            o.operands[0], ty, [kept[inner.index(d)] if d in inner else None
                                for d in range(len(o.operands[0].type.shape))], at)
    if name_of(o) == "linalg.fill":
        with InsertionPoint(at):
            init = tensor.EmptyOp(shape, ty.element_type).result
            return linalg.fill(o.operands[0], outs=[init])
    if name_of(o) != "linalg.generic" or len(o.results) != 1 or not _is_cheap(o):
        return None
    rank, n_in = len(v.type.shape), operand_segments(o)[0]
    if (any(m != AffineMap.get_identity(rank) for m in indexing_maps(o))
            or any("parallel" not in t for t in iterator_types(o))):
        return None
    ins = []
    for x in list(o.operands)[:n_in]:
        got = None if not _is_tensor(x) else _build_again_in_shape(
            x, RankedTensorType.get(shape, x.type.element_type), kept, at)
        if got is None:
            return None
        ins.append(got)
    with InsertionPoint(at):
        init = tensor.EmptyOp(shape, ty.element_type).result
        gen = Operation.create(
            "linalg.generic", results=[init.type], operands=ins + [init],
            attributes={
                "indexing_maps": ArrayAttr.get(
                    [AffineMapAttr.get(AffineMap.get_identity(len(shape)))]
                    * (n_in + 1)),
                "iterator_types": ArrayAttr.get(
                    [parallel_iterator()] * len(shape)),
                "operandSegmentSizes": DenseI32ArrayAttr.get([n_in, 1])},
            regions=1)
    body = o.regions[0].blocks[0]
    blk = gen.regions[0].blocks.append(*[a.type for a in body.arguments])
    env = dict(zip(body.arguments, blk.arguments))
    with InsertionPoint(blk):
        for old in body.operations:
            old = old.operation
            attrs = attrs_of(old)
            if old.name == "linalg.index":
                dim = IntegerAttr(attrs["dim"])
                if kept[dim.value] is None:
                    attrs = {"value": IntegerAttr.get(old.results[0].type, 0)}
                    new = Operation.create("arith.constant",
                                           results=[old.results[0].type], attributes=attrs)
                    env.update(zip(old.results, new.results))
                    continue
                attrs["dim"] = IntegerAttr.get(dim.type, kept[dim.value])
            new = Operation.create(
                old.name, results=[r.type for r in old.results],
                operands=[env.get(x, x) for x in old.operands],
                attributes=attrs)
            env.update(zip(old.results, new.results))
    return gen.results[0]


def _is_cheap(op):
    """A body of arith, math and its own indices: rebuilt per lane, it reads nothing across lanes."""
    for o in op.regions[0].blocks[0].operations:
        nm = o.operation.name
        if not (nm.startswith(("arith.", "math."))
                or nm in ("linalg.index", "linalg.yield")):
            return False
    return True


def _build_broadcast_across_lanes(op, src, init, added):
    from mlir.dialects import linalg, tensor
    from mlir.ir import DenseI64ArrayAttr, InsertionPoint
    s_shape, elt = list(src.type.shape), init.type.element_type
    lane = init
    if added != {0}:
        with InsertionPoint(op):
            lane = tensor.EmptyOp([init.type.shape[0]] + s_shape, elt).result
    got = _build_copy_op(op, src, lane, list(range(len(s_shape))),
                         list(range(len(s_shape) + 1)), added_dims=[0])
    if added == {0}:
        return got
    with InsertionPoint(op):
        wide = linalg.BroadcastOp([init.type], got, init,
                                  DenseI64ArrayAttr.get(sorted(added - {0})))
    blk = wide.regions[0].blocks.append(elt, elt)
    with InsertionPoint(blk):
        linalg.YieldOp([blk.arguments[0]])
    return wide.results[0]


def _writes(op, k):
    if k >= len(op.operands):
        return True
    if name_of(op) in ("linalg.broadcast", "linalg.transpose", "linalg.copy"):
        return k == 1
    if name_of(op) in kSlice + kReshape:
        return False
    for use in op.operands[k].uses:
        if use.owner == op.operation and use.operand_number == k:
            return writes_through_this_operand(op, use)
    return False


def _get_tied_init(op, r):
    from mlir.ir import RankedTensorType
    nm = name_of(op)
    if nm in ("tensor.insert_slice", "tensor.insert", torchsim.CopyOp.OPERATION_NAME):
        return 1 if r == 0 else None
    if nm == torchsim.TransferOp.OPERATION_NAME:
        return torchsim.operand_number(op, "sram") if r == 0 else None
    if nm == torchsim.DotOp.OPERATION_NAME:
        return torchsim.operand_number(op, "init") if r == 0 else None
    if nm.startswith("linalg."):
        outs = [k for k in range(len(op.operands))
                if isinstance(op.operands[k].type, RankedTensorType)
                and _writes(op, k)]
        return outs[r] if r < len(outs) else None
    return None


def _is_tensor(v):
    from mlir.ir import RankedTensorType
    return isinstance(v.type, RankedTensorType)


def _is_fresh(op):
    return name_of(op) in kFresh and not len(op.operands)


def _get_copied(op):
    if name_of(op) != "bufferization.alloc_tensor" or len(op.operands) != 1:
        return None
    return op.operands[0] if _is_tensor(op.operands[0]) else None


def _get_named_attrs(op, perms, wanted):
    nm = name_of(op)
    if nm not in ("linalg.transpose", "linalg.broadcast", "linalg.reduce"):
        return None
    sides = _get_named_sides(op, perms, wanted)
    if sides is None:
        return False
    pa, pb = sides
    if nm == "linalg.transpose":
        perm = int_array(op.attributes["permutation"])
        return {"permutation": [pa.index(perm[pb[i]]) for i in range(len(pb))]}
    dims = set(int_array(op.attributes["dimensions"]))
    if _get_kept_need(nm, pa, pb, dims) != (pa if nm == "linalg.broadcast"
                                            else pb):
        return False
    wide = pb if nm == "linalg.broadcast" else pa
    return {"dimensions": [i for i in range(len(wide)) if wide[i] in dims]}


def _get_named_sides(op, perms, wanted):
    half = len(op.operands) // 2 if name_of(op) == "linalg.reduce" else 1
    pas = [_get_effective(op, k, perms, wanted) for k in range(half)]
    pbs = [perms.get(r) for r in op.results] or [
        _get_effective(op, k, perms, wanted)
        for k in range(half, len(op.operands))]
    if any(p is None for p in pas + pbs) or len({tuple(p) for p in pas}) > 1 \
            or len({tuple(p) for p in pbs}) > 1:
        return None
    return pas[0], pbs[0]


def _get_effective(op, slot, perms, wanted):
    got = wanted.get((op.operation, slot))
    return got if got is not None else perms.get(op.operands[slot])


def _get_kept_need(nm, pa, pb, dims):
    wide = pb if nm == "linalg.broadcast" else pa
    kept = [a for a in range(len(wide)) if a not in dims]
    return [kept.index(wide[i]) for i in range(len(wide)) if wide[i] not in dims]


def _get_slice_ends(op):
    if name_of(op) == "tensor.extract_slice":
        return op.operands[0], op.results[0]
    return op.results[0], op.operands[0]


def _get_type_like(value, shape, elem):
    from mlir.ir import RankedTensorType
    ty = RankedTensorType(value.type)
    enc = ty.encoding if list(ty.shape) == list(shape) else None
    return RankedTensorType.get(list(shape), elem, enc)


def _remove_encodings(module):
    from mlir.ir import DenseElementsAttr, RankedTensorType
    for v in _collect_tensor_values(module):
        ty = RankedTensorType(v.type)
        if ty.encoding is not None:
            v.set_type(RankedTensorType.get(list(ty.shape), ty.element_type))
    for func in funcs(module):
        for raw in walk(func.operation):
            op = op_of(raw)
            if (name_of(op) != "arith.constant"
                    or not isinstance(op.results[0].type, RankedTensorType)):
                continue
            attr = DenseElementsAttr(op.attributes["value"])
            ty = RankedTensorType(attr.type)
            if ty.encoding is None:
                continue
            if not attr.is_splat:
                raise NotImplementedError(
                    "torchsim-materialize-layouts: a non-splat constant carrying a layout")
            op.attributes["value"] = DenseElementsAttr.get_splat(
                RankedTensorType.get(list(ty.shape), ty.element_type),
                attr.get_splat_value())


def _rewrite_encoding_casts(module):
    from mlir.ir import InsertionPoint, Operation, OpResult, RankedTensorType
    for func in funcs(module):
        for raw in list(walk(func.operation)):
            op = op_of(raw)
            if name_of(op) != "tensor.cast":
                continue
            src, res = op.operands[0], op.results[0]
            a, b = RankedTensorType(src.type), RankedTensorType(res.type)
            if list(a.shape) != list(b.shape) or a.element_type != b.element_type:
                continue
            if a.encoding is None and b.encoding is not None and isinstance(src, OpResult):
                _rewrite_type_up_the_chain(src, res.type)
                a = RankedTensorType(src.type)
            if a.encoding is None:
                continue
            if b.encoding is not None and a.encoding != b.encoding:
                with InsertionPoint(op):
                    keep = Operation.create("ttg.convert_layout", results=[res.type],
                                            operands=[src]).results[0]
            else:
                keep = src
            res.replace_all_uses_with(keep)
            op.operation.erase()


def _build_collapsed_units(op, v, dims):
    from mlir.dialects import tensor
    from mlir.ir import Attribute, InsertionPoint, RankedTensorType
    ty = RankedTensorType(v.type)
    shape = list(ty.shape)
    units = [i for i, d in enumerate(dims) if d is None]
    if not units or any(shape[u] != 1 for u in units):
        return None
    kept = [i for i in range(len(shape)) if i not in units]
    groups = [[i] for i in kept]
    for u in units:
        if groups:
            groups[0 if u < kept[0] else max(j for j, i in enumerate(kept) if i < u)].append(u)
    groups = [sorted(g) for g in groups]
    bases = get_bases(ty.encoding) if ty.encoding is not None else None
    enc = Attribute.parse(spell(drop_dims(bases, units))) if bases is not None and kept else None
    with InsertionPoint(op):
        return tensor.CollapseShapeOp(
            RankedTensorType.get([shape[i] for i in kept], ty.element_type, enc), v, groups).result


def _rewrite_same_order_conversions(module, plan):
    perms, copies, fixes = plan
    kept = []
    for op, how in fixes:
        if name_of(op) == "ttg.convert_layout" and perms.get(op.operands[0]) == perms.get(op.results[0]):
            perms.pop(op.results[0], None)
            op.results[0].replace_all_uses_with(op.operands[0])
            op.operation.erase()
            continue
        kept.append((op, how))
    return perms, copies, kept
