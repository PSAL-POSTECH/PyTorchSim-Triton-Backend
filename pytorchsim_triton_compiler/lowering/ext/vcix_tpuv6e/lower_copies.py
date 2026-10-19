"""tritonpytorchsim-lower-copies-tpuv6e: every `torchsim.copy` onto one move of vcix_tpuv6e's
cross-lane unit (tpu::Xlu), stamped with the XLU op, the rounds that carry it and the kind of move
it is; a copy no XLU op carries is refused -- this model has no other carrier for it."""

import itertools
import math

from ....dialects import torchsim
from ...passes.lib_bank_layout import row_major
from ...passes.lib_ir import funcs, int_array, name_of, op_of, walk
from ...passes.transform import LOWER_COPIES

NAME = "tritonpytorchsim-lower-copies-tpuv6e"

REQUIRES = ("vcix_tpuv6e",)

SLOT = LOWER_COPIES.name

FORBIDS = LOWER_COPIES.lowers


def run(module, lanes, **_opts):
    """Each copy, in the order it stands, onto its carrier."""
    for f in funcs(module):
        for op in [op_of(o) for o in walk(f.operation)
                   if name_of(o) == torchsim.CopyOp.OPERATION_NAME]:
            _rewrite_copy(op, lanes)
    return module


def _rewrite_copy(op, lanes):
    from .lib_xlu import XLU_PERMUTE, choose_xlu_op
    kind = _get_copy_kind(op)
    src, dst = op.operands[0], op.operands[1]
    s_shape, d_shape = list(src.type.shape), list(dst.type.shape)
    if kind == "broadcast" and _is_a_fold_result(op, src):
        rows = _get_in_lane_rows(_get_pairs(op, kind), s_shape, d_shape, lanes)
        if rows is not None:
            why = _build_register_reads(op, src, dst, rows, lanes)
            if why is not None:
                _build_in_lane_spread(op, src, dst, rows, lanes, why)
            op.operation.erase()
            return
    kinds = (XLU_PERMUTE,) if "into_window" in op.attributes else None
    chosen = choose_xlu_op(_get_pairs(op, kind), s_shape, d_shape, lanes, kinds=kinds)
    if chosen is None:
        raise NotImplementedError(f"a {kind} copy no XLU op carries: {op}")
    _build_move(op, chosen, src, dst)
    op.operation.erase()


_VIEWS = ("memref.collapse_shape", "memref.expand_shape", "memref.subview", "memref.cast",
          "memref.reinterpret_cast")


def _is_a_fold_result(op, src):
    """Is `src` (or the buffer it views) one a fold across the lanes wrote before `op`, and nothing
    but fills and folds write? A fold's pop gives every lane the result and every lane stores it in
    its own bank, so each lane already holds all of it."""
    from ...passes.lib_cross_lane import CROSS_LANE_KIND, CROSS_LANE_OP, FOLD
    from mlir.ir import StringAttr
    root = src
    while name_of(op_of(getattr(root, "owner", None))) in _VIEWS:
        root = op_of(root.owner).operands[0]
    folded, seen, work = False, set(), [root]
    while work:
        v = work.pop()
        for u in v.uses:
            o = op_of(u.owner)
            if o is None or o.operation == op.operation:
                continue
            nm = name_of(o)
            if nm in _VIEWS and u.operand_number == 0:
                if id(o) not in seen:
                    seen.add(id(o))
                    work.append(o.results[0])
            elif nm == CROSS_LANE_OP and u.operand_number == 1:
                if CROSS_LANE_KIND not in o.attributes or \
                        StringAttr(o.attributes[CROSS_LANE_KIND]).value != FOLD:
                    return False
                folded = folded or o.operation.is_before_in_block(op.operation)
            elif nm == "torchsim.transfer" or nm == torchsim.CopyOp.OPERATION_NAME and u.operand_number == 1:
                return False
    return folded


def _get_in_lane_rows(pairs, s_shape, d_shape, lanes):
    """Each destination row (its offset in a lane's bank) and the source row it takes, or None when
    a destination row does not take the same source row in every lane."""
    from .lib_xlu import fold_onto_lanes
    rows = {}
    for (_sl, sd), (_tl, td) in fold_onto_lanes(pairs, s_shape, d_shape, lanes):
        if rows.setdefault(td, sd) != sd:
            return None
    return rows


_READS = ("affine.vector_load", "vector.transfer_read")

SPREAD_WHY = "torchsim.spread_because"


def _build_register_reads(op, src, dst, rows, lanes):
    """A broadcast of what every lane holds, read by vector loads that each take the same source
    rows at every iteration, needs no destination: each load becomes those rows read once from the
    source before the copy and repeated in the vector register, the layout the load walks (the v6e
    uses a lane fold's pop in place). None when built; else why not, building nothing: every load
    the copy's result reaches must be one."""
    from ...passes.lib_bank_layout import bank_view
    from mlir.ir import InsertionPoint
    loads = _get_loads_of_this_copy(op, dst)
    if isinstance(loads, str):
        return loads
    plans = []
    for ld in loads:
        seq = _get_rows_every_iteration(ld, rows)
        if isinstance(seq, str):
            return seq
        period = _get_period(seq)
        if period is None:
            return f"a load takes rows that repeat no run of a step of one: {name_of(ld)}"
        plans.append((ld, seq[0], period, len(seq)))
    made = {}
    with InsertionPoint(op):
        s_bank = bank_view(src, lanes)
        for ld, first, period, n in plans:
            if (first, period, n) not in made:
                made[(first, period, n)] = _build_repeated_rows(s_bank, first, period, n)
            ld.results[0].replace_all_uses_with(made[(first, period, n)])
            ld.operation.erase()
    return None


def _get_loads_of_this_copy(op, dst):
    """The vector loads that read what `op` writes into `dst`'s buffer: every read of the buffer
    whose last writer before it is `op`, in `op`'s block or nested in an op after it there; why not
    when the buffer is read any other way, or before `op` inside a loop around `op`."""
    from ...passes.lib_ir import reads_through_this_operand, writes_through_this_operand
    root = dst
    while name_of(op_of(getattr(root, "owner", None))) in _VIEWS + ("torchsim.bank_view",):
        root = op_of(root.owner).operands[0]
    uses, work = [], [root]
    while work:
        v = work.pop()
        for u in v.uses:
            o = op_of(u.owner)
            if name_of(o) in _VIEWS + ("torchsim.bank_view",) and u.operand_number == 0:
                work.append(o.results[0])
            else:
                uses.append((o, u))
    func = op.operation
    while name_of(func) != "func.func":
        func = func.parent
    ops = [op_of(x).operation for x in walk(func)]

    def order(o):
        return next(k for k, x in enumerate(ops) if x == o.operation)
    at = order(op)
    writes = sorted(order(o) for o, u in uses if writes_through_this_operand(o, u))
    upto = next((k for k in writes if k > at), len(ops))
    loops = _get_enclosing_loops(op.operation)
    out = []
    for o, u in uses:
        k = order(o)
        if not reads_through_this_operand(o, u) or o.operation == op.operation:
            continue
        if k < at and any(_is_inside(o.operation, lp) for lp in loops):
            return f"the result is read before the copy in a loop around it: {name_of(o)}"
        if not at < k < upto:
            continue
        if name_of(o) not in _READS:
            return f"the result is read by other than a vector load: {name_of(o)}"
        if _get_ancestor_in(o.operation, op.operation.parent) is None:
            return f"the result is read outside the copy's block: {name_of(o)}"
        out.append(o)
    return out or "nothing reads the result"


def _get_enclosing_loops(o):
    out, p = [], o.parent
    while p is not None and name_of(p) != "func.func":
        if name_of(p) in ("affine.for", "scf.for", "scf.while", "affine.parallel", "scf.parallel"):
            out.append(p)
        p = p.parent
    return out


def _is_inside(o, anc):
    p = o.parent
    while p is not None:
        if p == anc:
            return True
        if name_of(p) == "func.func":
            return False
        p = p.parent
    return False


def _get_ancestor_in(o, block_owner):
    """`o` or the op around it whose parent is `block_owner`."""
    while o is not None and o.parent != block_owner:
        if name_of(o) == "func.func":
            return None
        o = o.parent
    return o


def _get_rows_every_iteration(ld, rows):
    """The source rows a 1-D vector load takes, when they are the same at every iteration of the
    loops around it; else why not."""
    import itertools
    from mlir.ir import AffineMapAttr, BlockArgument, VectorType
    from ...passes.lib_bank_layout import bank_layout
    from ...passes.lib_ir import const_int, eval_affine, loop_bounds
    vt = VectorType(ld.results[0].type)
    if len(vt.shape) != 1:
        return f"a load of more than one dim: {name_of(ld)}"
    mem = ld.operands[0]
    if name_of(ld) == "affine.vector_load":
        amap, idx = AffineMapAttr(ld.attributes["map"]).value, list(ld.operands[1:])
    else:
        from mlir.ir import AffineMap, ArrayAttr, BoolAttr, MemRefType
        rank = len(MemRefType(mem.type).shape)
        if AffineMapAttr(ld.attributes["permutation_map"]).value != \
                AffineMap.get_minor_identity(rank, 1) or \
                not all(BoolAttr(b).value for b in ArrayAttr(ld.attributes["in_bounds"])) or \
                len(ld.operands) != rank + 2:
            return f"a transfer that is not a 1-D, in-bounds, unmasked read: {name_of(ld)}"
        amap, idx = None, list(ld.operands[1:1 + rank])
    strides, base = bank_layout(mem)
    ivs, ranges, which = [], [], []
    for v in idx:
        if const_int(v) is not None:
            which.append(None)
            continue
        if v not in ivs:
            loop = op_of(v.owner.owner) if isinstance(v, BlockArgument) else None
            b = loop_bounds(loop) if loop is not None else None
            if b is None or None in b or list(loop.regions[0].blocks[0].arguments)[0] != v:
                return ("an index that is neither a constant nor a counter of constant "
                        f"bounds: {name_of(ld)}")
            ivs.append(v)
            ranges.append(range(*b))
        which.append(ivs.index(v))
    seq = None
    for point in itertools.product(*ranges):
        vals = [const_int(v) if w is None else point[w] for v, w in zip(idx, which)]
        if amap is not None:
            n = amap.n_dims
            vals = [eval_affine(e, vals[:n], vals[n:]) for e in amap.results]
        start = base + sum(a * s for a, s in zip(vals, strides))
        got = [rows.get(start + k) for k in range(vt.shape[0])]
        if None in got:
            return f"a load reads rows the copy does not write: {name_of(ld)}"
        if seq is not None and got != seq:
            return f"a load takes other rows at another iteration: {name_of(ld)}"
        seq = got
    return seq


def _get_period(seq):
    """The shortest run `seq` repeats, when that run is one row or rows at a step of one."""
    n = len(seq)
    for p in range(1, n + 1):
        if n % p or any(seq[k] != seq[k % p] for k in range(n)):
            continue
        run = seq[:p]
        return p if all(run[k + 1] == run[k] + 1 for k in range(p - 1)) else None
    return None


def _build_repeated_rows(s_bank, first, period, n):
    """A vector of `n` that is the `period` source rows from `first`, repeated: one row broadcast in
    the vector register it is read into, a run read once and joined with itself."""
    from ...passes.lib_ir import const_index, transfer_read, zero_of
    from mlir.dialects import vector
    from mlir.ir import MemRefType, VectorType
    elt = MemRefType(s_bank.type).element_type
    v = transfer_read(VectorType.get([period], elt), s_bank, [const_index(0), const_index(first)],
                      zero_of(elt), True)
    if period == 1:
        return vector.BroadcastOp(VectorType.get([n], elt), v).result if n > 1 else v
    if n == period:
        return v
    return vector.ShuffleOp(v, v, [k % period for k in range(n)]).result


def _build_in_lane_spread(op, src, dst, rows, lanes, why):
    """A broadcast of what every lane holds needs no lane to send anything: each lane copies its
    own bank's rows into the destination's: a run of a step of one a vector at a time, one row
    repeated broadcast in a register. Each write carries SPREAD_WHY, why the readers do not hold
    the rows in registers instead."""
    from mlir.ir import StringAttr
    from ...passes.lib_bank_layout import bank_view
    from ...passes.lib_ir import const_index, transfer_read, transfer_write, zero_of
    from mlir.dialects import vector
    from mlir.ir import InsertionPoint, MemRefType, VectorType
    runs, at = [], sorted(rows)
    i = 0
    while i < len(at):
        j, step = i + 1, None
        while j < len(at) and at[j] == at[j - 1] + 1:
            d = rows[at[j]] - rows[at[j - 1]]
            if step is None and d in (0, 1):
                step = d
            if d != step or j - i >= 64:
                break
            j += 1
        runs.append((at[i], rows[at[i]], j - i, step or 0))
        i = j
    elt = MemRefType(src.type).element_type
    with InsertionPoint(op):
        s_bank, d_bank = bank_view(src, lanes), bank_view(dst, lanes)
        for td, sd, n, step in runs:
            ty, at = VectorType.get([n], elt), [const_index(0), const_index(sd)]
            if step == 1 or n == 1:
                v = transfer_read(ty, s_bank, at, zero_of(elt), True)
            else:
                # one row repeated is broadcast in the vector register it is read into: a lane's
                # value never passes a scalar register
                one = transfer_read(VectorType.get([1], elt), s_bank, at, zero_of(elt), True)
                v = vector.BroadcastOp(ty, one).result
            w = transfer_write(v, d_bank, [const_index(0), const_index(td)], True)
            w.attributes[SPREAD_WHY] = StringAttr.get(why)


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
    from ...passes.lib_indexing_maps import get_kept_axes
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
    """The move, stamped with its XLU op, its rounds and, for whoever counts moves without
    knowing this unit, the kind of move that op is."""
    from mlir.ir import (ArrayAttr, DenseI64ArrayAttr, InsertionPoint, IntegerAttr, IntegerType,
                         Operation, StringAttr)
    from ...passes.lib_cross_lane import (CROSS_LANE_ATTR, CROSS_LANE_KIND, CROSS_LANE_OP,
                                          CROSS_LANE_ROUNDS)
    from .lib_xlu import emitter_of
    xlu_op, plan = chosen
    with InsertionPoint(op):
        move = Operation.create(CROSS_LANE_OP, operands=[src, dst])
    move.attributes[CROSS_LANE_ATTR] = IntegerAttr.get(IntegerType.get_signless(64), xlu_op)
    move.attributes[CROSS_LANE_KIND] = StringAttr.get(emitter_of(xlu_op))
    move.attributes[CROSS_LANE_ROUNDS] = ArrayAttr.get(
        [DenseI64ArrayAttr.get(list(r)) for r in plan])
    return move


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
