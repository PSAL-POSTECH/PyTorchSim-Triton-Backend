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
    from .lib_cross_lane import XLU_PERMUTE, choose_xlu_op
    kind = _get_copy_kind(op)
    src, dst = op.operands[0], op.operands[1]
    s_shape, d_shape = list(src.type.shape), list(dst.type.shape)
    chosen = None
    if kind != "window" or not any(s and s[0] > lanes for s in (s_shape, d_shape)):
        kinds = (XLU_PERMUTE,) if "into_window" in op.attributes else None
        chosen = choose_xlu_op(_get_pairs(op, kind), s_shape, d_shape, lanes, kinds=kinds)
    if chosen is None:
        raise NotImplementedError(f"a {kind} copy no XLU op carries: {op}")
    _build_move(op, chosen, src, dst)
    op.operation.erase()


def _get_copy_kind(op):
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
    src, dst = op.operands[0], op.operands[1]
    ps, pd = _get_perm(op, "src_perm"), _get_perm(op, "dst_perm")
    ss, ds = _get_logical(list(src.type.shape), ps), _get_logical(list(dst.type.shape), pd)
    at_s, at_d = _get_positions(list(src.type.shape)), _get_positions(list(dst.type.shape))
    return [(at_s[_get_flat(ss, ps, a)], at_d[_get_flat(ds, pd, b)])
            for a, b in _get_elements(op, kind, ss, ds)]


def _get_elements(op, kind, ss, ds):
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
    if not shape:
        return [(0, 0)]
    rows = row_major(shape)
    return [(x[0], sum(x[i] * rows[i] for i in range(1, len(shape))))
            for x in _get_all_coords(shape)]


def _build_move(op, chosen, src, dst):
    from mlir.ir import ArrayAttr, DenseI64ArrayAttr, InsertionPoint, IntegerAttr, IntegerType, Operation
    from .lib_cross_lane import CROSS_LANE_ATTR, CROSS_LANE_OP, CROSS_LANE_ROUNDS
    xlu_op, plan = chosen
    with InsertionPoint(op):
        move = Operation.create(CROSS_LANE_OP, operands=[src, dst])
    move.attributes[CROSS_LANE_ATTR] = IntegerAttr.get(IntegerType.get_signless(64), xlu_op)
    move.attributes[CROSS_LANE_ROUNDS] = ArrayAttr.get(
        [DenseI64ArrayAttr.get(list(r)) for r in plan])
    return move


def _apply_async_transfers(module):
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
    got = [(i, user) for i, user, use in _find_tile_users(tile, -1, order)
           if i < at and _is_inside(user, loop) and writes_through_this_operand(user, use)]
    return min(got, key=lambda g: g[0])[1] if got else None


def _get_func(op):
    f = op.operation
    while name_of(f) != "func.func":
        f = f.parent
    return f


def _get_loop(op):
    p = op.operation.parent
    while p is not None and name_of(p) not in _LOOP_OPS:
        p = p.parent
    return p


def _is_inside(op, loop):
    p = op
    while p is not None and p != loop:
        p = p.parent
    return p is not None


def _get_perm(op, name):
    return list(int_array(op.attributes[name]))


def _get_logical(shape, perm):
    out = [0] * len(shape)
    for k, a in enumerate(perm):
        out[a] = shape[k]
    return out


def _get_window(op):
    return tuple(list(int_array(op.attributes[n]))
                 for n in ("static_offsets", "static_sizes", "static_strides"))


def _get_coords(f, shape):
    x = [0] * len(shape)
    for i in reversed(range(len(shape))):
        x[i], f = f % shape[i], f // shape[i]
    return x


def _get_all_coords(shape):
    return [list(x) for x in itertools.product(*[range(n) for n in shape])]


def _get_flat(shape, perm, x):
    n = 0
    for a in perm:
        n = n * shape[a] + x[a]
    return n
