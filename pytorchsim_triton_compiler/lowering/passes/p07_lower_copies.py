"""torchsim-lower-copies: every `torchsim.copy` onto a unit this machine has -- one move of the
cross-lane unit where one of its ops carries it, else a round trip through a DRAM image -- then
every transfer async with its wait where the tile is first needed, which the trace reads."""

import itertools
import math

from ...dialects import torchsim
from .lib_bank_layout import row_major
from .lib_ir import (funcs, int_array, is_memref, name_of, op_of, reads_through_this_operand, walk,
                     writes_through_this_operand)

FORBIDS = (torchsim.CopyOp.OPERATION_NAME,)

_LOOP_OPS = ("scf.for", "affine.for", "scf.while")


def run(module, lanes, **_opts):
    """Each copy, in the order it stands, onto its carrier; then the transfers made async."""
    for f in funcs(module):
        for op in [op_of(o) for o in walk(f.operation)
                   if name_of(o) == torchsim.CopyOp.OPERATION_NAME]:
            _rewrite_copy(op, lanes)
    _apply_async_transfers(module)
    return module


def _rewrite_copy(op, lanes):
    """One copy onto the XLU op of fewest rounds that carries every element's (from, to), else
    its kind's DRAM round trip, and the copy gone. WRITING INTO A WINDOW keeps the rest of the
    tile, so only a permute will do; a window past the machine's lanes goes through DRAM."""
    from .lib_cross_lane import XLU_PERMUTE, choose_xlu_op
    kind = _get_copy_kind(op)
    src, dst = op.operands[0], op.operands[1]
    s_shape, d_shape = list(src.type.shape), list(dst.type.shape)
    chosen = None
    if kind != "window" or not any(s and s[0] > lanes for s in (s_shape, d_shape)):
        kinds = (XLU_PERMUTE,) if "into_window" in op.attributes else None
        chosen = choose_xlu_op(_get_pairs(op, kind), s_shape, d_shape, lanes, kinds=kinds)
    if chosen is not None:
        _build_move(op, chosen, src, dst)
    else:
        _TRIPS[kind](op)
    op.operation.erase()


def _get_copy_kind(op):
    """What the copy moves: a broadcast along axes it adds, a window, a reshape where the two
    ends' logical shapes differ, else a relayout into another lane axis."""
    if "added_dims" in op.attributes:
        return "broadcast"
    if "static_offsets" in op.attributes:
        return "window"
    src, dst = op.operands[0], op.operands[1]
    if (_get_logical(list(src.type.shape), _get_perm(op, "src_perm"))
            != _get_logical(list(dst.type.shape), _get_perm(op, "dst_perm"))):
        return "reshape"
    return "relayout"


def _get_pairs(op, kind):
    """`(from, to)` bank positions of every element the copy writes, each end stored in its own
    order and banked on its leading axis."""
    src, dst = op.operands[0], op.operands[1]
    ps, pd = _get_perm(op, "src_perm"), _get_perm(op, "dst_perm")
    ss, ds = _get_logical(list(src.type.shape), ps), _get_logical(list(dst.type.shape), pd)
    at_s, at_d = _get_positions(list(src.type.shape)), _get_positions(list(dst.type.shape))
    return [(at_s[_get_flat(ss, ps, a)], at_d[_get_flat(ds, pd, b)])
            for a, b in _get_elements(op, kind, ss, ds)]


def _get_elements(op, kind, ss, ds):
    """(source, destination) logical coordinates of every element the copy writes: a broadcast
    repeats the kept axes, a window steps from its offsets, a reshape keeps row-major order."""
    from .lib_indexing_maps import get_kept_axes
    if kind == "broadcast":
        added = set(int_array(op.attributes["added_dims"]))
        kept = [d for d in range(len(ds)) if d not in added]
        return [([y[d] for d in kept], y) for y in _get_all_coords(ds)]
    if kind == "window":
        into = "into_window" in op.attributes
        offs, sizes, steps = _get_window(op)
        small, big = (ss, ds) if into else (ds, ss)
        kept = get_kept_axes(sizes, small)
        out = []
        for y in _get_all_coords(small):
            x = list(offs)
            for j, a in enumerate(kept):
                x[a] += steps[a] * y[j]
            out.append((y, x) if into else (x, y))
        return out
    if kind == "reshape":
        return [(_get_coords(f, ss), _get_coords(f, ds)) for f in range(math.prod(ss))]
    return [(y, y) for y in _get_all_coords(ds)]


def _get_positions(shape):
    """Every element's `(lane, depth)` in a tile of `shape` banked on axis 0, by its row-major
    position; a rank-0 tile is one element in lane 0."""
    if not shape:
        return [(0, 0)]
    rows = row_major(shape)
    return [(x[0], sum(x[i] * rows[i] for i in range(1, len(shape))))
            for x in _get_all_coords(shape)]


def _build_move(op, chosen, src, dst):
    """ONE MOVE of the cross-lane unit from `src` into `dst`, in front of `op`, carrying its
    XLU op and the plan it was priced at."""
    from mlir.ir import ArrayAttr, DenseI64ArrayAttr, InsertionPoint, IntegerAttr, IntegerType, Operation
    from .lib_cross_lane import CROSS_LANE_ATTR, CROSS_LANE_OP, CROSS_LANE_ROUNDS
    xlu_op, plan = chosen
    with InsertionPoint(op):
        move = Operation.create(CROSS_LANE_OP, operands=[src, dst])
    move.attributes[CROSS_LANE_ATTR] = IntegerAttr.get(IntegerType.get_signless(64), xlu_op)
    move.attributes[CROSS_LANE_ROUNDS] = ArrayAttr.get(
        [DenseI64ArrayAttr.get(list(r)) for r in plan])
    return move


def _build_broadcast_trip(op):
    """The source's DRAM image read back with a zero stride along the axes the copy adds."""
    added = set(int_array(op.attributes["added_dims"]))
    src, dst = op.operands[0], op.operands[1]
    s_shape, d_shape = list(src.type.shape), list(dst.type.shape)
    rows = row_major(s_shape)
    kept = [d for d in range(len(d_shape)) if d not in added]
    strides = [0] * len(d_shape)
    for k, d in enumerate(kept):
        strides[d] = rows[k]
    staging = _build_staging(op, s_shape or [1], dst.type.element_type)
    _build_trip(op, staging, "MVOUT", src, rows)
    _build_trip(op, staging, "MVIN", dst, strides)


def _build_window_trip(op):
    """Through the big tile's DRAM image, where the window's start is a DRAM index and no leg
    names a bank; writing into the window writes the whole tile back."""
    from .lib_indexing_maps import get_kept_axes
    src, dst = op.operands[0], op.operands[1]
    into = "into_window" in op.attributes
    p_src, p_dst = _get_perm(op, "src_perm"), _get_perm(op, "dst_perm")
    big, small, pb, ps = (dst, src, p_dst, p_src) if into else (src, dst, p_src, p_dst)
    bshape = _get_logical(list(big.type.shape), pb)
    sshape = _get_logical(list(small.type.shape), ps)
    offs, sizes, steps = _get_window(op)
    rm = row_major(bshape)
    start = sum(o * r for o, r in zip(offs, rm))
    walked = [rm[a] * steps[a] for a in get_kept_axes(sizes, sshape)]
    staging = _build_staging(op, bshape, big.type.element_type)
    image = _get_stored(rm, pb)
    _build_trip(op, staging, "MVOUT", big, image)
    if into:
        _build_trip(op, staging, "MVOUT", src, _get_stored(walked, ps), start)
        _build_trip(op, staging, "MVIN", dst, image)
    else:
        _build_trip(op, staging, "MVIN", dst, _get_stored(walked, ps), start)


def _build_reshape_trip(op):
    """The source's DRAM image read back in the destination's shape -- row-major order is what
    a reshape keeps."""
    src, dst = op.operands[0], op.operands[1]
    ps, pr = _get_perm(op, "src_perm"), _get_perm(op, "dst_perm")
    sshape = _get_logical(list(src.type.shape), ps)
    dshape = _get_logical(list(dst.type.shape), pr)
    staging = _build_staging(op, sshape, src.type.element_type)
    _build_trip(op, staging, "MVOUT", src, _get_stored(row_major(sshape), ps))
    _build_trip(op, staging, "MVIN", dst, _get_stored(row_major(dshape), pr))


def _build_relayout_trip(op):
    """A DRAM round trip whose MVOUT scatters the source into the destination's row-major
    image."""
    src, dst = op.operands[0], op.operands[1]
    p_src, p_dst = _get_perm(op, "src_perm"), _get_perm(op, "dst_perm")
    perm = [p_src.index(a) for a in p_dst]
    d_shape = list(dst.type.shape)
    rows = row_major(d_shape)
    inverse = [0] * len(perm)
    for d, a in enumerate(perm):
        inverse[a] = d
    staging = _build_staging(op, d_shape, dst.type.element_type)
    _build_trip(op, staging, "MVOUT", src, [rows[inverse[a]] for a in range(len(perm))])
    _build_trip(op, staging, "MVIN", dst, rows)


_TRIPS = {"broadcast": _build_broadcast_trip, "window": _build_window_trip,
          "reshape": _build_reshape_trip, "relayout": _build_relayout_trip}


def _build_staging(op, shape, elt):
    """A flat DRAM buffer of `shape`'s element count at the module's top level, under the first
    `graphtripN_dram` name the module does not hold. ROW-MAJOR AT BOTH ENDS."""
    from mlir.ir import InsertionPoint, MemRefType, StringAttr, TypeAttr
    from mlir.dialects import memref
    top = op.operation
    while top.parent is not None:
        top = top.parent
    block = top.regions[0].blocks[0]
    taken = {o.attributes["sym_name"].value for o in block.operations
             if o.operation.name == "memref.global"}
    name = next(n for n in (f"graphtrip{i}_dram" for i in itertools.count(1)) if n not in taken)
    with InsertionPoint(block):
        g = memref.GlobalOp(StringAttr.get(name),
                            TypeAttr.get(MemRefType.get([math.prod(shape)], elt)),
                            sym_visibility=None, initial_value=None,
                            constant=False, alignment=None)
    return g.operation


def _build_trip(op, staging, kind, tile, strides, dram_at=0):
    """One transfer between `tile` and the staging image, in front of `op`, its DRAM strides
    listed in the tile's stored-axis order."""
    from mlir.dialects import arith, memref
    from mlir.ir import FlatSymbolRefAttr, IndexType, InsertionPoint, IntegerAttr

    def index(n):
        return arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), n)).result

    with InsertionPoint(op):
        image = memref.GetGlobalOp(staging.attributes["type"].value,
                                   FlatSymbolRefAttr.get(staging.attributes["sym_name"].value)).result
        return torchsim.TransferOp(
            None, image, index(dram_at), tile, index(0), index(0),
            [], [], kind, torchsim.DRAM_ARG_ROUND_TRIP, [int(s) for s in strides]).operation


def _apply_async_transfers(module):
    """Every transfer but a round trip async on its function's next tag slot, and a wait on that
    slot in front of the op that first needs the tile: reads it after an MVIN, writes it after an
    MVOUT (a loop's earlier writer when none follows). The ELF drops them; the trace times them."""
    from mlir.dialects import arith
    from mlir.ir import BoolAttr, IndexType, InsertionPoint, IntegerAttr, MemRefType
    order, transfers = {}, []
    for i, op in enumerate(walk(module.operation)):
        order[op] = i
        if name_of(op) == torchsim.TransferOp.OPERATION_NAME:
            transfers.append((i, op))
    slots, per_func = [], {}
    for _i, op in transfers:
        key = str(_get_func(op).attributes["sym_name"])
        slots.append(per_func.get(key, 0))
        per_func[key] = slots[-1] + 1

    def index(n):
        with InsertionPoint.at_block_begin(entry):
            return arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), n)).result

    for (at, op), slot in zip(transfers, slots):
        tx = op.operation.opview
        if IntegerAttr(tx.dram_arg).value == torchsim.DRAM_ARG_ROUND_TRIP:
            continue
        entry = _get_func(op).regions[0].blocks[0]
        tx.async_ = BoolAttr.get(True)
        slot_v = index(slot)
        op.operands[torchsim.operand_number(op, "tag_idx")] = slot_v
        mvin = "MVIN" in tx.dma_kind.value
        anchor = next((user for _i, user, use in _find_tile_users(tx.sram, at, order)
                       if (reads_through_this_operand if mvin
                           else writes_through_this_operand)(user, use)), None)
        if anchor is None and not mvin and _get_loop(op) is not None:
            anchor = _find_earlier_writer(tx.sram, at, order, _get_loop(op))
        if anchor is None:
            continue
        num = index(math.prod(MemRefType(tx.sram.type).shape))
        with InsertionPoint(anchor):
            torchsim.WaitOp(slot_v, num)


def _find_tile_users(tile, after, order):
    """`(position, op, use)` of every op past position `after` that uses `tile` or a view of it,
    earliest first."""
    seen, front, hits = [], [tile], []
    while front:
        for use in front.pop().uses:
            o = op_of(use.owner)
            if o.results and all(is_memref(r) for r in o.results):
                for r in o.results:
                    if not any(r == s for s in seen):
                        seen.append(r)
                        front.append(r)
                continue
            i = order.get(o)
            if i is not None and i > after:
                hits.append((i, o, use))
    return sorted(hits, key=lambda h: h[0])


def _find_earlier_writer(tile, at, order, loop):
    """The earliest op before position `at` inside `loop` that writes `tile`: the next trip's
    first writer."""
    got = [(i, user) for i, user, use in _find_tile_users(tile, -1, order)
           if i < at and _is_inside(user, loop) and writes_through_this_operand(user, use)]
    return min(got, key=lambda g: g[0])[1] if got else None


def _get_func(op):
    """The func.func `op` is in."""
    f = op.operation
    while name_of(f) != "func.func":
        f = f.parent
    return f


def _get_loop(op):
    """The nearest loop `op` sits in, or None."""
    p = op.operation.parent
    while p is not None and name_of(p) not in _LOOP_OPS:
        p = p.parent
    return p


def _is_inside(op, loop):
    """Is `op` in `loop`'s body?"""
    p = op
    while p is not None and p != loop:
        p = p.parent
    return p is not None


def _get_perm(op, name):
    """The logical axis each stored axis of one end is."""
    return list(int_array(op.attributes[name]))


def _get_logical(shape, perm):
    """The logical shape of a tile stored as `shape`, stored axis k being logical perm[k]."""
    out = [0] * len(shape)
    for k, a in enumerate(perm):
        out[a] = shape[k]
    return out


def _get_stored(per_logical, perm):
    """Per-logical-axis numbers listed in stored-axis order."""
    return [per_logical[a] for a in perm]


def _get_window(op):
    """(offsets, sizes, steps) of a window copy, in logical axes."""
    return tuple(list(int_array(op.attributes[n]))
                 for n in ("static_offsets", "static_sizes", "static_strides"))


def _get_coords(f, shape):
    """The coordinates of row-major element `f` of `shape`."""
    x = [0] * len(shape)
    for i in reversed(range(len(shape))):
        x[i], f = f % shape[i], f // shape[i]
    return x


def _get_all_coords(shape):
    """Every coordinate of `shape`, in row-major order."""
    return [list(x) for x in itertools.product(*[range(n) for n in shape])]


def _get_flat(shape, perm, x):
    """Where logical coordinates `x` of a tile of logical `shape` fall in row-major order once
    it is stored with axis k holding logical axis perm[k]."""
    n = 0
    for a in perm:
        n = n * shape[a] + x[a]
    return n
