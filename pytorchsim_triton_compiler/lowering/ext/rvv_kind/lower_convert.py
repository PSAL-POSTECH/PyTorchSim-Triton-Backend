"""lower_convert (rvv_kind): a `pytorchsim.convert_layout` whose source every bank holds is each
bank's copy from its own store -- the readers take the rows straight from the source in registers, or
each bank spreads them into the destination -- on RVV alone. A convert that crosses banks has no
carrier here yet: the shared-scratchpad round trip comes with address space 2."""

from ...passes.lib_bank_layout import is_uniform
from ...passes.lib_convert import held_by, in_bank_rows, kind_of, pairs_of
from ...passes.lib_ir import funcs, name_of, op_of, walk
from ...passes.transform import LOWER_CONVERT
from ....dialects import pytorchsim
from ....dialects.pytorchsim_rvv import BANK_VIEW

NAME = "pytorchsim-lower-convert-generic"

SLOT = LOWER_CONVERT.name

FORBIDS = LOWER_CONVERT.lowers


def run(module, lanes, **_opts):
    """Each convert, in the order it stands, from each bank's own store; one that crosses banks is
    refused."""
    for f in funcs(module):
        for op in [op_of(o) for o in walk(f.operation)
                   if name_of(o) == pytorchsim.ConvertLayoutOp.OPERATION_NAME]:
            if not lower_in_bank(op, lanes):
                raise NotImplementedError(
                    f"a {kind_of(op)} convert_layout crosses banks, and rvv_kind has no carrier for "
                    f"that yet (the shared-scratchpad round trip comes with address space 2): {op}")
    return module


def lower_in_bank(op, lanes):
    """`op` lowered as each bank's copy from its own store, when its source is UNIFORM and every
    destination row takes one source row in every bank that holds it (in_bank_rows): True, the op
    gone; else False, nothing built."""
    src, dst = op.operands[0], op.operands[1]
    if not is_uniform(src):
        return False
    pairs = pairs_of(op, kind_of(op), lanes)
    rows = in_bank_rows(pairs, held_by(dst, lanes), list(src.type.shape), list(dst.type.shape), lanes)
    if rows is None:
        return False
    why = _build_register_reads(op, src, dst, rows, lanes)
    if why is not None:
        _build_in_lane_spread(op, src, dst, rows, lanes, why)
    op.operation.erase()
    return True


_READS = ("affine.vector_load", "vector.transfer_read")


SPREAD_WHY = "pytorchsim.spread_because"


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
        s_bank = bank_view(src, lanes, BANK_VIEW)
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
    from ....dialects.pytorchsim import view_root
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
        s_bank, d_bank = bank_view(src, lanes, BANK_VIEW), bank_view(dst, lanes, BANK_VIEW)
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
