"""pytorchsim-place-transfer-waits: every transfer, async since convert-tts-to-pytorchsim made it, takes its
semaphore and its wait where the tile is first needed (whole-tile loads started together: where the first is),
among the final readers (once the target's ops are instructions, before the descriptor cuts a transfer);
a store in flight is waited before its free."""

import math

from ...dialect import pytorchsim
from ..utils.ir_utils import ends_lifetime, is_memref, name_of, op_of, reads_through_this_operand, trip_count
from ..utils.ir_utils import walk, writes_through_this_operand
from .. import pass_name

NAME = pass_name(__name__)

MARKERS = (pytorchsim.TransferOp.OPERATION_NAME,)

_LOOP_OPS = ("scf.for", "affine.for", "scf.while")


def run(module, **_):
    """Each transfer's semaphore, and its wait where the tile is first needed (or, for a store, before its
    buffer's free), landing in the buffer the transfer's tile is a view of."""
    from mlir.dialects import arith
    from mlir.ir import IndexType, InsertionPoint, IntegerAttr
    order, transfers = {}, []
    for i, op in enumerate(walk(module.operation)):
        order[op] = i
        if name_of(op) == pytorchsim.TransferOp.OPERATION_NAME:
            transfers.append((i, op))
    slots, per_func = [], {}
    for _i, op in transfers:
        key = str(_get_func(op).attributes["sym_name"])
        slots.append(per_func.get(key, 0))
        per_func[key] = slots[-1] + 1

    def index(n):
        with InsertionPoint.at_block_begin(entry):
            return arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), n)).result

    users = {}
    anchors = [_first_need(op.operation.opview, at, order, users) for at, op in transfers]
    anchors = _join_whole_loads(transfers, anchors, order)
    for (at, op), slot, anchor in zip(transfers, slots, anchors):
        tx = op.operation.opview
        entry = _get_func(op).regions[0].blocks[0]
        slot_v = index(slot)
        op.operands[pytorchsim.operand_number(op, "sem")] = slot_v
        moved = math.prod(pytorchsim.moved_shape(tx))
        if anchor is None and not pytorchsim.is_load(tx):
            _wait_before_the_free(op, tx, at, order, slot_v, moved, index)
            continue
        if anchor is None:
            continue
        _wait_at(op, anchor, slot_v, moved, index, tx)


def _first_need(tx, at, order, users):
    """The first op after the transfer at `at` that reads its tile (a load) or writes it (a store); `users`
    holds each tile's users in program order, walked once for every transfer of that tile."""
    mvin = pytorchsim.is_load(tx)
    found = users.get(tx.sram)
    if found is None:
        found = users.setdefault(tx.sram, _find_tile_users(tx.sram, -1, order))
    return next((user for i, user, use in found if i > at
                 and (reads_through_this_operand if mvin else writes_through_this_operand)(user, use)), None)


def _join_whole_loads(transfers, anchors, order):
    """Loads of whole tiles (no subtile_size) started in one block before any of them is read are one step's
    operands: each is waited where the first of them is needed, so compute never overlaps a step's own loads;
    only a subtiled load overlaps the compute, as a pipeline does. A load that is itself that first read (a
    gather reading another's indices) starts after it, so it is no operand of that step."""
    out, group = list(anchors), []

    def close():
        if group:
            first = min((anchors[i] for i in group), key=lambda a: order[a])
            for i in group:
                out[i] = first
        group.clear()

    for i, (at, op) in enumerate(transfers):
        tx = op.operation.opview
        if not pytorchsim.is_load(tx) or tx.subtile_size is not None or anchors[i] is None:
            continue
        if group and (op.operation.parent != transfers[group[0]][1].operation.parent
                      or at >= min(order[anchors[j]] for j in group)):
            close()
        group.append(i)
    close()
    return out


def _wait_at(op, anchor, slot_v, moved, index, tx):
    from mlir.ir import InsertionPoint
    where, trips = _place_wait(op, anchor)
    if trips:
        num = index(moved * trips)
        with InsertionPoint(where):
            pytorchsim.WaitOp(slot_v, num, [pytorchsim.view_root(tx.sram)])


def _wait_before_the_free(op, tx, at, order, slot_v, moved, index):
    """A store no later writer waits for: the next trip's writer waits for the trip before, and a wait
    before its buffer's free takes what is left in flight; past a loop of no constant trip count, each
    trip's store is waited at the end of that loop's body instead, as a load's wait is (_place_wait)."""
    from mlir.ir import InsertionPoint
    root = pytorchsim.view_root(tx.sram)
    free = next((op_of(u.owner) for u in root.uses if ends_lifetime(op_of(u.owner), root)), None)
    if free is not None and _trips_until(op, free) is None:
        _wait_at(op, free, slot_v, moved, index, tx)
        return
    loop = _get_loop(op)
    war = _find_earlier_writer(tx.sram, at, order, loop) if loop is not None else None
    if war is not None:
        _wait_at(op, war, slot_v, moved, index, tx)
    if free is None:
        return
    trips = _trips_until(op, free)
    if trips:
        num = index(moved if war is not None else moved * trips)
        with InsertionPoint(free):
            pytorchsim.WaitOp(slot_v, num, [root])


def _trips_until(op, free):
    """How many times `op` runs before `free`: the product of the trip counts of the loops around
    it and not around `free`; None where one of them is not a constant."""
    trips, loop = 1, _get_loop(op)
    while loop is not None and not _is_inside(free, loop):
        n = trip_count(loop)
        if n is None:
            return None
        trips *= n
        loop = _get_loop(loop)
    return trips


def _place_wait(op, anchor):
    """Where the wait for `op` stands and how many of its trips it covers: before `anchor` at `op`'s own loop
    depth when every loop around `op` and not around `anchor` is counted, else at the end of the innermost
    uncounted one's body -- the wait farthest out whose amount is a constant."""
    trips, loop = 1, _get_loop(op)
    while loop is not None and not _is_inside(anchor, loop):
        n = trip_count(loop)
        if n is None:
            return _get_body_end(loop, op), trips
        trips *= n
        loop = _get_loop(loop)
    return _out_of_loops_without(anchor, op), trips


def _out_of_loops_without(where, op):
    """`where` moved out of every loop around it that is not around `op`: the latest point before it at
    the depth `op` runs at, so a tile moved once is not waited for each trip of a loop that reads it."""
    p = where.parent
    while p is not None and name_of(p) != "func.func":
        if name_of(p) in _LOOP_OPS and not _is_inside(op.operation, p):
            where = p
        p = p.parent
    return where


def _get_body_end(loop, op):
    o = op.operation
    while o.parent != loop:
        o = o.parent
    return list(o.block.operations)[-1]


def _find_tile_users(tile, after, order):
    seen, front, hits = set(), [tile], []
    while front:
        for use in front.pop().uses:
            o = op_of(use.owner)
            if o.results and all(is_memref(r) for r in o.results):
                for r in o.results:
                    if r not in seen:
                        seen.add(r)
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
