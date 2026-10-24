"""pytorchsim-make-transfers-async: a transfer cut into subtiles becomes async with its wait
where the tile is first needed (among the final readers, so after lower-copies), and a store still
in flight there also waited before its buffer is freed; any other transfer stays synchronous."""

import math

from ...dialects import torchsim
from .lib_ir import ends_lifetime, is_memref, name_of, op_of, reads_through_this_operand, trip_count
from .lib_ir import walk, writes_through_this_operand

NAME = "pytorchsim-make-transfers-async"

_LOOP_OPS = ("scf.for", "affine.for", "scf.while")


class TailWaitRefused(Exception):
    """A store still in flight when its buffer is freed whose batch is not a constant: a loop
    between them whose trip count is not one."""


def run(module, **_):
    """Each transfer cut into subtiles made async, its wait where the tile is first needed (or, for a
    store, before its buffer's free), landing in the buffer the transfer's tile is a view of."""
    from mlir.dialects import arith
    from mlir.ir import BoolAttr, IndexType, InsertionPoint, IntegerAttr
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
        if tx.subtile_size is None:
            continue
        entry = _get_func(op).regions[0].blocks[0]
        tx.async_ = BoolAttr.get(True)
        slot_v = index(slot)
        op.operands[torchsim.operand_number(op, "sem")] = slot_v
        mvin = torchsim.is_load(tx)
        anchor = next((user for _i, user, use in _find_tile_users(tx.sram, at, order)
                       if (reads_through_this_operand if mvin
                           else writes_through_this_operand)(user, use)), None)
        moved = math.prod(torchsim.moved_shape(tx))
        if anchor is None and not mvin:
            _wait_before_the_free(op, tx, at, order, slot_v, moved, index)
            continue
        if anchor is None:
            continue
        _wait_at(op, anchor, slot_v, moved, index, tx)


def _wait_at(op, anchor, slot_v, moved, index, tx):
    from mlir.ir import InsertionPoint
    where, trips = _place_wait(op, anchor)
    if trips:
        num = index(moved * trips)
        with InsertionPoint(where):
            torchsim.WaitOp(slot_v, num, [torchsim.view_root(tx.sram)])


def _wait_before_the_free(op, tx, at, order, slot_v, moved, index):
    """A store no later writer waits for: in a loop, the next trip's writer waits for the trip
    before (if one writes the tile there), and before its buffer's free a wait takes what is left
    in flight -- the last trip then, else every trip."""
    from mlir.ir import InsertionPoint
    loop = _get_loop(op)
    war = _find_earlier_writer(tx.sram, at, order, loop) if loop is not None else None
    if war is not None:
        _wait_at(op, war, slot_v, moved, index, tx)
    root = torchsim.view_root(tx.sram)
    free = next((op_of(u.owner) for u in root.uses if ends_lifetime(op_of(u.owner), root)), None)
    if free is None:
        return
    trips = _trips_until(op, free)
    if trips:
        num = index(moved if war is not None else moved * trips)
        with InsertionPoint(free):
            torchsim.WaitOp(slot_v, num, [root])


def _trips_until(op, free):
    """How many times `op` runs before `free`: the product of the trip counts of the loops around
    it and not around `free`; refused where one of them is not a constant."""
    trips, loop = 1, _get_loop(op)
    while loop is not None and not _is_inside(free, loop):
        n = trip_count(loop)
        if n is None:
            raise TailWaitRefused(
                f"{NAME}: a store in flight when its buffer is freed sits in a {name_of(loop)} "
                f"whose trip count is not a constant, so its wait's batch is not one: {op}")
        trips *= n
        loop = _get_loop(loop)
    return trips


def _place_wait(op, anchor):
    """Where the wait for `op` stands and how many of its trips it covers: at `anchor` when every
    loop around `op` and not around `anchor` is counted, else at the end of the innermost uncounted
    one's body -- the wait farthest out whose amount is a constant."""
    trips, loop = 1, _get_loop(op)
    while loop is not None and not _is_inside(anchor, loop):
        n = trip_count(loop)
        if n is None:
            return _get_body_end(loop, op), trips
        trips *= n
        loop = _get_loop(loop)
    return anchor, trips


def _get_body_end(loop, op):
    o = op.operation
    while o.parent != loop:
        o = o.parent
    return list(o.block.operations)[-1]


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
