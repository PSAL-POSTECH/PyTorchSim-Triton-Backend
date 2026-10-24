"""lower_copies: every `torchsim.copy` onto one move of the cross-lane unit
(tpu::Xlu), or into the DMA load it copies when that load's pieces cost less; a copy no XLU op carries is
refused -- the model has no other carrier for it."""

import itertools
import math

from ....dialects import torchsim
from ...passes.lib_bank_layout import is_uniform, row_major, view_lanes
from ...passes.lib_ir import funcs, int_array, name_of, op_of, walk
from ...passes.transform import LOWER_COPIES

NAME = "pytorchsim-lower-copies"

REQUIRES = ("pytorchsim_tpu_family",)

SLOT = LOWER_COPIES.name

FORBIDS = LOWER_COPIES.lowers


def run(module, lanes, **_opts):
    """Each copy, in the order it stands, onto its carrier."""
    from ...passes.lib_ir import LANES_ATTR, lanes_of_module
    per_cycle = lanes_of_module(module) if LANES_ATTR in module.operation.attributes else None
    for f in funcs(module):
        for op in [op_of(o) for o in walk(f.operation)
                   if name_of(o) == torchsim.CopyOp.OPERATION_NAME]:
            _rewrite_copy(op, lanes, per_cycle)
    return module


def _rewrite_copy(op, lanes, per_cycle):
    """A copy from a UNIFORM buffer, which every lane holds, is each lane's copy from its own bank
    when it writes each destination row it writes in every lane the destination holds there; any
    other copy is one move of the cross-lane unit."""
    from .lib_xlu import XLU_PERMUTE, choose_xlu_op, reaches_every_lane
    kind = _get_copy_kind(op)
    src, dst = op.operands[0], op.operands[1]
    s_shape, d_shape = list(src.type.shape), list(dst.type.shape)
    pairs = _get_pairs(op, kind, lanes)
    src_uniform = is_uniform(src)
    if src_uniform:
        rows = _get_in_lane_rows(pairs, _get_held(dst, lanes), s_shape, d_shape, lanes)
        if rows is not None:
            why = _build_register_reads(op, src, dst, rows, lanes)
            if why is not None:
                _build_in_lane_spread(op, src, dst, rows, lanes, why)
            op.operation.erase()
            return
    kinds = (XLU_PERMUTE,) if "into_window" in op.attributes else None
    chosen = choose_xlu_op(pairs, s_shape, d_shape, lanes, src_uniform, view_lanes(src), kinds=kinds)
    if chosen is None:
        raise NotImplementedError(f"a {kind} copy no XLU op carries: {op}")
    if not src_uniform and _absorbed_into_its_load(
            op, _move_cost(chosen, src, dst, lanes, len(pairs), per_cycle)):
        return
    _build_move(op, chosen, src, dst, reaches_every_lane(pairs, lanes))
    op.operation.erase()


def _move_cost(chosen, src, dst, lanes, moved, per_cycle):
    """The unit's ideal cycles for the move, from the target's vcix keys and the runs the permute
    emits (lib_xlu.permute_runs); another XLU op's elements over the lanes a cycle. The core's own
    instructions around the pushes and pops are not counted (#291). None when a fact is unsaid."""
    from mlir.ir import MemRefType
    from .... import machine
    from ...passes.lib_ir import elem_bits
    from .lib_xlu import XLU_PERMUTE, crossing_bits, crossings_of, emitter_of, permute_runs
    xlu_op, plan = chosen
    if emitter_of(xlu_op) != XLU_PERMUTE:
        return None if per_cycle is None else -(-moved // per_cycle)
    t = machine.target()
    if "tpu_xlu_elements_per_cycle" not in t.vcix or "tpu_xlu_latency_cycles" not in t.vcix:
        return None
    width, latency = int(t.vcix["tpu_xlu_elements_per_cycle"]), int(t.vcix["tpu_xlu_latency_cycles"])
    elt = MemRefType(src.type).element_type
    crossings, _ = crossings_of(elem_bits(elt))
    if not crossings:
        return None
    runs = permute_runs(plan, min(view_lanes(dst) or lanes, lanes), t.vlen_bits // crossing_bits(elt))
    return latency + 1 + crossings * sum(-(-n // math.gcd(w, width)) + -(-n // w)
                                         for _, (*_, n), w, _ in runs)


def _absorbed_into_its_load(op, move_cost):
    """The copy done by the one plain DMA load that fills its source, restated into its destination
    (renumbered_per_axis), when that load's extra pieces at dma_load_latency_cycles each cost less
    than the move; True when it was. The source buffer and its load go."""
    from mlir.ir import InsertionPoint, Operation
    from .... import machine
    from ...passes.lib_descriptor import read_cut
    from ...passes.lib_ir import attrs_of, int_array, producer
    from ...passes.p11_fit_transfers_to_descriptor import runs_of
    latency = machine.target().dma_load_latency_cycles
    src, dst = op.operands[0], op.operands[1]
    groups = _physical_groups(op)
    load = _sole_plain_load(op) if groups is not None else None
    if move_cost is None or latency is None or load is None:
        return False
    try:
        restated = torchsim.renumbered_per_axis(load, groups, list(src.type.shape))
    except NotImplementedError:
        return False
    attrs = {k: v for k, v in dict(attrs_of(load), **restated).items() if v is not None}
    operands = list(load.operands)
    operands[2] = dst
    with InsertionPoint(op):
        new = Operation.create(torchsim.TransferOp.OPERATION_NAME, operands=operands,
                               attributes=attrs)
    cut, shape = read_cut(new) or {}, list(dst.type.shape)
    stride = list(int_array(new.attributes["dram_stride"]))
    pieces = math.prod(len(runs_of(digits, shape[d], stride[d])) for d, digits in cut.items())
    if (pieces - 1) * latency >= move_cost:
        new.erase()
        return False
    alloc = producer(src)
    for u in [_operation(u.owner) for u in src.uses if name_of(u.owner) == "memref.dealloc"]:
        u.erase()
    load.operation.erase()
    op.operation.erase()
    if not list(alloc.results[0].uses):
        alloc.operation.erase()
    return True


_LOAD_ONLY = ("cut", "indirect_dim", "subtile_size", "dyn_stride_axes", "offset_stride", "piece_box")


def _sole_plain_load(op):
    """The DMA load that is the only writer of the copy's fresh source, before it in its block, with no
    cut, index tile or run-time stride (a mask renumbers or refuses); None for any other source."""
    from mlir.ir import StringAttr
    from ...passes.lib_ir import producer
    src = op.operands[0]
    if name_of(producer(src)) != "memref.alloc":
        return None
    users = [_operation(u.owner) for u in src.uses]
    loads = [u for u in users if name_of(u) == torchsim.TransferOp.OPERATION_NAME]
    rest = [u for u in users if name_of(u) not in (torchsim.TransferOp.OPERATION_NAME,
                                                     "memref.dealloc")]
    if len(loads) != 1 or len(rest) != 1 or rest[0] != op.operation:
        return None
    load = loads[0]
    if (load.operands[2] != src or StringAttr(load.attributes["dma_kind"]).value != "MVIN"
            or any(a in load.attributes for a in _LOAD_ONLY) or any(
                torchsim.operand_number(load, a) != torchsim.operand_number(load, b)
                for a, b in (("index", "clamp"), ("dyn_strides", "dyn_offset_stride")))
            or torchsim.operand_number(load, "dyn_offset_stride") != len(load.operands)
            or load.block != op.operation.block):
        return None
    order = [_operation(o) for o in op.operation.block.operations]
    at = lambda o: next(k for k, x in enumerate(order) if x == o)
    return load.opview if at(load) < at(op.operation) else None


def _operation(x):
    return getattr(x, "operation", x)


def _physical_groups(op):
    """For a reshape copy into a destination laid as it reads, the source's physical dims each
    destination dim gathers, outer first: its logical dims' row-major collapse; else None."""
    src, dst = op.operands[0], op.operands[1]
    ps, pd = _get_perm(op, "src_perm"), _get_perm(op, "dst_perm")
    if _get_copy_kind(op) != "reshape" or pd != list(range(len(pd))):
        return None
    groups = _collapse_groups(_get_logical(list(src.type.shape), ps), list(dst.type.shape))
    return None if groups is None else [[ps.index(a) for a in g] for g in groups]


def _collapse_groups(s_shape, d_shape):
    """The groups of adjacent source dims each destination dim is the product of, or None."""
    groups, k = [], 0
    for n in d_shape:
        g, prod = [], 1
        while k < len(s_shape) and (prod < n or (not g and s_shape[k] == n)):
            prod *= s_shape[k]
            g.append(k)
            k += 1
            if prod == n:
                break
        if prod != n or not g:
            return None
        groups.append(g)
    return groups if k == len(s_shape) else None


def _get_in_lane_rows(pairs, held, s_shape, d_shape, lanes):
    """Each destination row (its offset in a lane's bank) and the source row it takes, or None when
    a row does not take the same source row in every lane, or the copy leaves a lane's part of a row
    it writes unwritten: a copy from each lane's own bank writes the row in every lane."""
    from .lib_xlu import fold_onto_lanes
    rows, written = {}, set()
    for (_sl, sd), at in fold_onto_lanes(pairs, s_shape, d_shape, lanes):
        if rows.setdefault(at[1], sd) != sd:
            return None
        written.add(at)
    if any(at[1] in rows and at not in written for at in held):
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
    from ....dialects.torchsim import view_root
    from ...passes.lib_ir import is_memref, reads_through_this_operand, writes_through_this_operand
    root = view_root(dst)
    uses, work = [], [root]
    while work:
        v = work.pop()
        for u in v.uses:
            o = op_of(u.owner)
            views = [r for r in o.results if is_memref(r) and r != root and view_root(r) == root]
            if views and u.operand_number == 0:
                work.extend(views)
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


def _get_pairs(op, kind, lanes):
    """Each element's `(source (lane, row), destination (lane, row))`; a UNIFORM destination is
    held by every lane, so each of its elements is listed once for every lane of the machine."""
    src, dst = op.operands[0], op.operands[1]
    ps, pd = _get_perm(op, "src_perm"), _get_perm(op, "dst_perm")
    ss, ds = _get_logical(list(src.type.shape), ps), _get_logical(list(dst.type.shape), pd)
    at_s, at_d = _get_positions(list(src.type.shape)), _get_positions(list(dst.type.shape))
    pairs = [(at_s[_get_flat(ss, ps, a)], at_d[_get_flat(ds, pd, b)])
             for a, b in _get_elements(op, kind, ss, ds)]
    if not is_uniform(dst):
        return pairs
    return [(f, (lane, t[1])) for f, t in pairs for lane in range(lanes)]


def _get_held(dst, lanes):
    """Every `(lane, row)` the destination holds, on the machine's lanes: a UNIFORM one in every
    lane."""
    from .lib_xlu import fold_onto_lanes
    shape = list(dst.type.shape)
    at = _get_positions(shape)
    if is_uniform(dst):
        return [(lane, t[1]) for t in at for lane in range(lanes)]
    return [t for _f, t in fold_onto_lanes([((0, 0), t) for t in at], [], shape, lanes)]


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


def _build_move(op, chosen, src, dst, every_lane):
    """The move, stamped with its XLU op, its rounds and, for whoever counts moves without knowing
    this unit, the kind of move that op is; EVERY_LANE when its pairs reach every lane."""
    from mlir.ir import (ArrayAttr, DenseI64ArrayAttr, InsertionPoint, IntegerAttr, IntegerType,
                         Operation, StringAttr, UnitAttr)
    from ...passes.lib_cross_lane import (CROSS_LANE_ATTR, CROSS_LANE_KIND, CROSS_LANE_OP,
                                          CROSS_LANE_ROUNDS, EVERY_LANE)
    from .lib_xlu import emitter_of
    xlu_op, plan = chosen
    with InsertionPoint(op):
        move = Operation.create(CROSS_LANE_OP, operands=[src, dst])
    move.attributes[CROSS_LANE_ATTR] = IntegerAttr.get(IntegerType.get_signless(64), xlu_op)
    move.attributes[CROSS_LANE_KIND] = StringAttr.get(emitter_of(xlu_op))
    move.attributes[CROSS_LANE_ROUNDS] = ArrayAttr.get(
        [DenseI64ArrayAttr.get(list(r)) for r in plan])
    if every_lane:
        move.attributes[EVERY_LANE] = UnitAttr.get()
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
