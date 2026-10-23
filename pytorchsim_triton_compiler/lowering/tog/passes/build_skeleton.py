"""build_skeleton pass (C2): reduce a kernel's post-vcix MLIR to the *skeleton + API* form, in
place."""

import math
import os

from ....contract.stop import Reason, Refused
from ....dialects.torchsim import TransferOp, WaitOp, branch_controls, is_load, moved_shape, terminator_controls
TRANSFER, WAIT = TransferOp.OPERATION_NAME, WaitOp.OPERATION_NAME
from . import togsim_ops as ts
from ._mlir_util import attr_i64_array, attr_int, walk_ops, i32, i64, i64_array, str_attr
from .build_tog import (
    COMPUTE_TYPE_NAME,
    ir,
    TogBuilder,
    _build,
    _find_kernel,
    stamp_tile_ids,
    tile_ids,
    _is_block_arg,
    TOGDMANode,
    TOGDMAWaitNode,
)

_KEEP = {
    "affine.for", "scf.for", "scf.while", "scf.if",
    "affine.yield", "scf.yield", "scf.condition", "func.return",
    ts.DMA, ts.COMPUTE, ts.MEMORY_BAR, ts.ATTRIBUTE,
}


def _kernel_block(module):
    func_op = _find_kernel(module)
    if func_op is None:
        return None
    return func_op.regions[0].blocks[0]


def _arg_id_of(base_addr):
    s = str(base_addr)
    return int(s[3:]) if s.startswith("arg") and s[3:].lstrip("-").isdigit() else -1


def _live_extents(block, dma_by_op):
    """{transfer: [(dims axis, runtime extent)]} for a plain masked transfer: its clamp's [low, high)
    held to the tile (a subtile's), which may be 0; a gather, a piece, and an async load whose semaphore
    is not all in one block or waited for whole tiles keep their static size."""
    from mlir.dialects import arith
    from mlir.ir import IntegerAttr

    def plain(op):
        tx, o = op.operation.opview, op.operation
        node = dma_by_op.get(o)
        if (node is None or tx.masked_axes is None or tx.index is not None
                or "piece_box" in o.attributes):
            return None
        axes = [IntegerAttr(a).value for a in tx.masked_axes]
        # A subtile's clamp is of the subtile, as the descriptor that moves it reads it.
        moved = moved_shape(o)
        if (len(list(tx.clamp)) != 2 * len(axes)
                or any(node.tile_size[node.lead + a] != moved[a] for a in axes)):
            return None
        return axes

    def one_block(sem, load):
        users = [u.owner for u in sem.uses]
        if any(u.name not in (TRANSFER, WAIT) or u.operation.block != load.operation.block
               for u in users):
            return False
        loads = [u for u in users if u.name == TRANSFER and u.operation.opview.sem == sem]
        waits = [u for u in users if u.name == WAIT]
        if any(not is_load(u) or dma_by_op.get(u.operation) is None for u in loads):
            return False
        whole = sum(math.prod(moved_shape(u)) for u in loads)
        return all(_get_const_index(w.operation.opview.amount) == whole for w in waits)

    out = {}
    idx = ir.IndexType.get()
    for op in list(walk_ops(block)):
        if op.operation.name != TRANSFER:
            continue
        axes = plain(op)
        # A synchronous load has no barrier to rebuild: what reads its buffer waits on the buffer.
        if not axes or (is_load(op) and dma_by_op[op.operation].is_async
                        and not one_block(op.operation.opview.sem, op)):
            continue
        node = dma_by_op[op.operation]
        clamp = list(op.operation.opview.clamp)
        with ir.InsertionPoint(op), ir.Location.unknown():
            out[op.operation] = [(node.lead + axis, _extent(clamp[2 * i], clamp[2 * i + 1],
                                                            node.tile_size[node.lead + axis]))
                                 for i, axis in enumerate(axes)]
    return out


def _extent(low, high, tile):
    """min(high, tile) - max(low, 0) at the insertion point, as a constant when the clamp is one."""
    from mlir.dialects import arith
    idx = ir.IndexType.get()
    lo, hi = _get_const_index(low), _get_const_index(high)
    if lo is not None and hi is not None:
        return arith.ConstantOp(idx, ir.IntegerAttr.get(idx, min(hi, tile) - max(lo, 0))).result
    zero = arith.ConstantOp(idx, ir.IntegerAttr.get(idx, 0)).result
    return arith.SubIOp(arith.MinSIOp(high, arith.ConstantOp(idx, ir.IntegerAttr.get(idx, tile)).result).result,
                        arith.MaxSIOp(low, zero).result).result


_FLOAT_ELEMS = ("f16", "bf16", "f32", "f64")


def _elem(dma_node):
    """The togsim.hpp element type a transfer moves: its float type by name, else the integer of
    its width (TOGSim reads only the width)."""
    if dma_node.element_type in _FLOAT_ELEMS:
        return dma_node.element_type
    if dma_node.element_size in (8, 16, 32, 64):
        return f"i{dma_node.element_size}"
    raise RuntimeError(f"a transfer of {dma_node.element_type} moves no byte-sized element, "
                       f"which a TOG cannot state")


def _emit_dma(ctx, dma_node, dram_index, sem, read_bufs, write_bufs, extents=(), gather=None):
    op = dma_node.op
    attrs = {
        ts.ATTR_DIR: i32(ts.DIR_STORE if dma_node.is_write else ts.DIR_LOAD),
        ts.ATTR_DIMS: i64_array(dma_node.tile_size),
        ts.ATTR_STRIDES: i64_array(dma_node.tile_stride),
        ts.ATTR_ELEM: str_attr(_elem(dma_node)),
        ts.ATTR_IS_ASYNC: ir.BoolAttr.get(bool(dma_node.is_async)),
        ts.ATTR_ARG_ID: i32(_arg_id_of(dma_node.base_addr)),
        ts.ATTR_INDIRECT: i32(int(dma_node.indirect_mode)),
        "base": str_attr(dma_node.base_addr),
        ts.ATTR_READ_BUFS: i64_array(read_bufs),
        ts.ATTR_WRITE_BUFS: i64_array(write_bufs),
    }
    if extents:
        attrs[ts.ATTR_DYN_DIMS] = i64_array([a for a, _ in extents])
    tail = []
    if gather is not None:
        gid, at, axis, scale = gather
        attrs[ts.ATTR_GATHER] = i64_array([gid, len(at), axis, -1 if not isinstance(scale, int) else scale])
        tail = list(at) + ([] if isinstance(scale, int) else [scale])
    elif dma_node.indirect_mode:
        raise RuntimeError(f"an indirect transfer reaches the TOG with no index snapshot: {op}")
    ir.Operation.create(
        ts.DMA,
        results=[],
        operands=[dram_index, sem] + [e for _, e in extents] + tail,
        attributes=attrs,
        loc=ir.Location.unknown(ctx),
        ip=ir.InsertionPoint(op),
    )


def _emit_memory_bar(ctx, anchor_op, sem):
    """A barrier on `sem`: the TOG waits every load that signaled `sem` since its last wait."""
    ir.Operation.create(
        ts.MEMORY_BAR, results=[], operands=[sem],
        loc=ir.Location.unknown(ctx), ip=ir.InsertionPoint(anchor_op))


def _emit_compute(ctx, compute_node, tile_id, type_name, read_bufs, write_bufs):
    front = compute_node.operations[0]
    attrs = {
        ts.ATTR_TILE_ID: i64(tile_id),
        ts.ATTR_COMPUTE_TYPE: i32(int(compute_node.compute_type)),
        "compute_type_name": str_attr(type_name),
        ts.ATTR_READ_BUFS: i64_array(read_bufs),
        ts.ATTR_WRITE_BUFS: i64_array(write_bufs),
    }
    ir.Operation.create(
        ts.COMPUTE,
        results=[],
        operands=[],
        attributes=attrs,
        loc=ir.Location.unknown(ctx),
        ip=ir.InsertionPoint(front),
    )


def _has_nonempty_region(op):
    for region in op.operation.regions:
        for b in region.blocks:
            if len(list(b.operations)) > 0:
                return True
    return False


def _results_unused(op):
    for r in op.operation.results:
        if len(list(r.uses)) > 0:
            return False
    return True


def _strip_loop_iter_args(block, builder=None):
    """Drop every loop iter_arg the TOG does not run by -- its result unread and nothing of it reaching
    a TOG op -- reading its init instead; whether any was dropped."""
    changed = False
    while True:
        tgt = keep = None
        for op in walk_ops(block):
            o = op.operation
            if o.name not in ("affine.for", "scf.for") or not len(o.results):
                continue
            args = list(o.regions[0].blocks[0].arguments)[1:]
            want = [k for k, (a, r) in enumerate(zip(args, o.results))
                    if list(r.uses) or _feeds_tog(a, [])]
            if len(want) < len(args):
                tgt, keep = op, want
                break
        if tgt is None:
            return changed
        owner = None
        if builder is not None:
            for node in builder.loop_nodes:
                if node.op.operation == tgt.operation:
                    owner = node
                    break
        fresh = _rebuild_loop(tgt, keep)
        changed = True
        if owner is not None and fresh is not None:
            owner.op = fresh


def _feeds_tog(v, seen):
    """Whether `v` reaches, through any chain of uses, an op the TOG states, a loop's bounds or a
    branch's condition: what a loop's iter_arg must keep being for the TOG to recompute it."""
    if any(x == v for x in seen):
        return False
    seen.append(v)
    for use in v.uses:
        o = use.owner.operation if hasattr(use.owner, "operation") else use.owner
        if o.name in ts.OP_NAMES or o.name == ts.ATTRIBUTE or o.name == "scf.if":
            return True
        if o.name in ("affine.for", "scf.for"):
            n_bounds = len(o.operands) - len(o.results)
            if use.operand_number < n_bounds:
                return True
            k = use.operand_number - n_bounds
            nxt = [o.regions[0].blocks[0].arguments[k + 1], o.results[k]]
        elif o.name in ("affine.yield", "scf.yield"):
            p = o.parent.operation
            k = use.operand_number
            nxt = [p.results[k]] if k < len(p.results) else []
            if p.name in ("affine.for", "scf.for"):
                nxt.append(p.regions[0].blocks[0].arguments[k + 1])
        else:
            nxt = list(o.results)
        if any(_feeds_tog(x, seen) for x in nxt):
            return True
    return False


def _rebuild_loop(op, keep):
    """`op` rebuilt with the iter_args numbered `keep` alone; the others read their inits."""
    o = op.operation
    nres = len(o.results)
    n_in = len(o.operands)
    inits = [o.operands[n_in - nres + i] for i in range(nres)]
    keep_operands = [o.operands[i] for i in range(n_in - nres)] + [inits[k] for k in keep]
    old_block = o.regions[0].blocks[0]
    oargs = list(old_block.arguments)

    attrs = {o.attributes[i].name: o.attributes[i].attr
             for i in range(len(o.attributes))}
    if "operandSegmentSizes" in attrs:
        seg = list(ir.DenseI32ArrayAttr(attrs["operandSegmentSizes"]))
        seg[-1] = len(keep)
        attrs["operandSegmentSizes"] = ir.DenseI32ArrayAttr.get(seg)

    loc = ir.Location.unknown(o.context)
    with loc:
        new = ir.Operation.create(o.name, results=[o.results[k].type for k in keep],
                                  operands=keep_operands, attributes=attrs, regions=1, loc=loc,
                                  ip=ir.InsertionPoint(o))
        nb = new.regions[0].blocks.append(oargs[0].type, *[oargs[k + 1].type for k in keep])

        oargs[0].replace_all_uses_with(nb.arguments[0])
        old_term = list(old_block.operations)[-1].operation
        yielded = list(old_term.operands)
        for k, (ba, ini) in enumerate(zip(oargs[1:], inits)):
            if k in keep:
                ba.replace_all_uses_with(nb.arguments[1 + keep.index(k)])
                o.results[k].replace_all_uses_with(new.results[keep.index(k)])
            else:
                ba.replace_all_uses_with(ini)
                o.results[k].replace_all_uses_with(ini)

        term_name = "affine.yield" if o.name == "affine.for" else "scf.yield"
        with ir.InsertionPoint(nb):
            ir.Operation.create(term_name, results=[], operands=[yielded[k] for k in keep], loc=loc)
        new_term = list(nb.operations)[0]
        for bop in list(old_block.operations)[:-1]:
            bop.operation.move_before(new_term)
        o.erase()
    return new


def _strip_branch_results(block):
    """Every branch whose results nothing reads rebuilt with none, its terminators forwarding nothing:
    the TOG takes a branch for the arm it picks, not for what the arm yields; whether any was."""
    victims = [op.operation for op in walk_ops(block) if branch_controls(op.operation) is not None
               and len(op.operation.results) and _results_unused(op)]
    for o in victims:
        attrs = {o.attributes[i].name: o.attributes[i].attr for i in range(len(o.attributes))}
        new = ir.Operation.create(o.name, results=[], operands=list(o.operands), attributes=attrs,
                                  regions=len(o.regions), loc=o.location, ip=ir.InsertionPoint(o))
        for old, fresh in zip(o.regions, new.regions):
            for b in old.blocks:
                nb = fresh.blocks.append(*[a.type for a in b.arguments])
                for a, x in zip(b.arguments, nb.arguments):
                    a.replace_all_uses_with(x)
                ops = [x.operation for x in b.operations]
                for x in ops[:-1]:
                    nb.append(x)
                with ir.InsertionPoint(nb):
                    ir.Operation.create(ops[-1].name, results=[], operands=terminator_controls(ops[-1]) or [],
                                        loc=ops[-1].location)
        o.erase()
    return bool(victims)


def _dce(block):
    changed = True
    while changed:
        changed = False
        victims = []
        for op in walk_ops(block):
            name = op.operation.name
            if name in _KEEP:
                continue
            if _has_nonempty_region(op):
                continue
            if _results_unused(op):
                victims.append(op)
        for op in victims:
            try:
                op.operation.erase()
                changed = True
            except Exception:
                pass


def _collect_dma_nodes(builder):
    by_op = {}
    seen = set()

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        if isinstance(n, (TOGDMANode, TOGDMAWaitNode)) and n.op is not None:
            by_op[n.op.operation] = n
        for c in n.children:
            visit(c)

    for ln in builder.loop_nodes:
        visit(ln)
    for dn in getattr(builder, "dma_nodes", ()):
        visit(dn)
    return by_op


class _BufferIds:

    def __init__(self):
        self._ids = {}

    def of(self, name):
        """The id of buffer_of's key `name` (a global's symbol, an allocated value, CORE), given in
        the order asked; -1 for None."""
        if name is None:
            return -1
        return self._ids.setdefault(name, len(self._ids))

    def names(self):
        """Every name given an id, in id order."""
        return list(self._ids)


def _spad_buffer_bytes(module, bufs):
    """Bytes of the scratchpad buffer each id names -- a global's declared type, an allocated one's
    own -- by id; 0 for an id naming none (a virtual edge, a DRAM image). What the TOG states so
    TOGSim need not infer it from DMAs."""
    from ...passes.lib_bank_layout import in_scratchpad
    from ...passes.lib_ir import elem_bytes

    def nbytes(ty):
        return math.prod(ty.shape) * elem_bytes(ty.element_type)
    sizes = {}
    for op in module.body.operations:
        if op.operation.name == "memref.global":
            ty = ir.MemRefType(ir.TypeAttr(op.operation.attributes["type"]).value)
            if ty.memory_space is not None:
                sizes[ir.StringAttr(op.operation.attributes["sym_name"]).value] = nbytes(ty)
    return [sizes.get(name, 0) if isinstance(name, str)
            else nbytes(ir.MemRefType(name.type)) if in_scratchpad(name.type) else 0
            for name in bufs.names()]


def _emit_computes(ctx, builder, bufs, ids):
    from . import dep_analysis as dep
    for tile_id, cn in zip(ids, builder.compute_nodes, strict=True):
        reads, writes = dep.compute_buffers(cn)
        _emit_compute(ctx, cn, tile_id, COMPUTE_TYPE_NAME[cn.compute_type],
                      sorted(bufs.of(b) for b in reads),
                      sorted(bufs.of(b) for b in writes))
    return len(ids)


def _transfer_fields(op):
    tx = op.operation.opview
    return {"dram": tx.dram, "dram_idx": tx.dram_idx, "sram": tx.sram,
            "sram_idx": tx.sram_idx, "sem": tx.sem,
            "offset": tx.index}


def _gather_axis(op, node, row_rank):
    """The Desc dim a gather's snapshot row runs along, after the descriptor's `lead` dims (a uniform
    load's lane copies repeat the row), or -1 for a row of every element, which no lane copy repeats."""
    tile = moved_shape(op.operation)
    if row_rank == 1:
        d = op.operation.opview.indirect_dim
        d = ir.IntegerAttr(d).value if d is not None else next(
            (k for k, e in enumerate(tile) if e != 1), len(tile) - 1)
        return node.lead + d
    if node.lead:
        raise Refused(Reason.RUNTIME_ATTRIBUTE,
                      f"a gather of one index per element walked over {node.tile_size[:node.lead]} lane "
                      f"copies: a TOG Gather states one row for the whole walk ({op.operation})")
    return -1


def _emit_one_dma(ctx, op, node, builder, bufs, live, gathers=None):
    from . import dep_analysis as dep
    f = _transfer_fields(op)
    spad_id = bufs.of(dep.buffer_of(f["sram"]))
    read_bufs = [spad_id] if node.is_write else []
    write_bufs = [] if node.is_write else [spad_id]
    if f["offset"] is not None:
        off_id = bufs.of(dep.buffer_of(f["offset"]))
        if off_id not in read_bufs:
            read_bufs = read_bufs + [off_id]
    gather = None
    if f["offset"] is not None:
        tx = op.operation.opview
        gid, at, rank = next(g for o, g in (gathers or []) if o == op.operation)
        scale = (tx.dyn_offset_stride if tx.dyn_offset_stride is not None
                 else ir.IntegerAttr(tx.offset_stride).value)
        gather = (gid, at, _gather_axis(op, node, rank - len(at)), scale)
    _emit_dma(ctx, node, f["dram_idx"], f["sem"], read_bufs, write_bufs,
              live.get(op.operation, ()), gather)


def _signalled_by(sem):
    """For each transfer that signals `sem`, whether it is a load."""
    loads = set()
    for use in sem.uses:
        user = use.owner
        if user.name == TRANSFER and user.operation.opview.sem == sem:
            loads.add(is_load(user))
    return loads


def _emit_one_wait(ctx, op):
    """A barrier for a load wait; a store wait is not lowered and returns False."""
    wait = op.operation.opview
    loads = _signalled_by(wait.sem)
    if loads == {False}:
        return False
    if False in loads:
        raise RuntimeError(f"a wait's semaphore is signalled by a load and a store at once: {op}")
    if _get_const_index(wait.amount) is None:
        raise RuntimeError(f"a wait whose amount is not a constant cannot be stated in a TOG: {op}")
    _emit_memory_bar(ctx, op, wait.sem)
    return True


def _get_const_index(v):
    if v is None or _is_block_arg(v) or v.owner.name != "arith.constant":
        return None
    return ir.IntegerAttr(v.owner.attributes["value"]).value


def _emit_dmas_and_waits(ctx, block, builder, dma_by_op, bufs, gathers=None):
    originals = []
    n_dma = n_wait = 0
    live = _live_extents(block, dma_by_op)
    for op in list(walk_ops(block)):
        name = op.operation.name
        if name == TRANSFER:
            node = dma_by_op.get(op.operation)
            if node is None:
                continue
            _emit_one_dma(ctx, op, node, builder, bufs, live, gathers)
            originals.append(op)
            n_dma += 1
        elif name == WAIT:
            if _emit_one_wait(ctx, op):
                n_wait += 1
            originals.append(op)
    return originals, n_dma, n_wait


def _refuse_reads_left(block):
    """No op left reads memory: what the TOG takes from the run's data is a runtime attribute, and a
    read standing here is one the marking pass did not see."""
    from ...passes.lib_ir import is_memref
    from ....dialects.torchsim import effects_on
    for op in walk_ops(block):
        o = op.operation
        if o.name in ts.OP_NAMES:
            continue
        if any(is_memref(v) and effects_on(o, v) & {"read", "unknown"} for v in o.operands):
            raise RuntimeError(f"a scalar DRAM read reaches the TOG unmarked: {str(o).splitlines()[0][:160]}")


def _take_records(module):
    """Every runtime attribute record as the TOG has it -- an S or R a togsim.attribute_read, an A or G
    nothing -- the rank of each id onto the kernel; returns [(transfer, (id, trips))] of each G."""
    from ...passes.p11a_mark_runtime_attributes import records
    ranks, gathers = {}, []
    for o in records(module):
        rec = o.opview
        i, rank = rec.id.value, rec.rank.value
        if ranks.setdefault(i, rank) != rank:
            raise RuntimeError(f"runtime attribute {i} is recorded at ranks {ranks[i]} and {rank}")
        if rec.form.value in ("S", "R"):
            with ir.InsertionPoint(o), o.location:
                read = ir.Operation.create(ts.ATTRIBUTE, results=[o.results[0].type],
                                           operands=list(rec.indices),
                                           attributes={"id": o.attributes["id"], "rank": o.attributes["rank"]})
            o.results[0].replace_all_uses_with(read.results[0])
        if rec.form.value == "G":
            ops = [x.operation for x in o.block.operations]
            nxt = ops[next(k for k, x in enumerate(ops) if x == o) + 1]
            if nxt.name != TRANSFER or nxt.opview.index != rec.source:
                raise RuntimeError(f"an index snapshot does not sit right before its gather: {o}")
            gathers.append((nxt, (i, list(rec.indices), rank)))
        o.erase()
    if sorted(ranks) != list(range(len(ranks))):
        raise RuntimeError(f"runtime attribute ids {sorted(ranks)} are not 0..{len(ranks) - 1}")
    _find_kernel(module).operation.attributes[ts.ATTR_RUNTIME_ATTRIBUTES] = ir.DenseI32ArrayAttr.get(
        [ranks[i] for i in sorted(ranks)])
    return gathers


def _guard_live_extents(block):
    """A DMA sized at run time runs only when it moves something, and a barrier only when a load it
    waits for may have run: under `if` the extents are positive, or one of the loads ran."""
    from mlir.dialects import arith, scf
    idx = ir.IndexType.get()
    guards = {}
    for op in [o.operation for o in walk_ops(block)]:
        if op.name == ts.DMA and ts.ATTR_DYN_DIMS in op.attributes:
            dyn = list(op.operands)[2:2 + len(attr_i64_array(op, ts.ATTR_DYN_DIMS))]
            if all((_get_const_index(e) or 0) > 0 for e in dyn):
                continue
            with ir.InsertionPoint(op), ir.Location.unknown():
                zero = arith.ConstantOp(idx, ir.IntegerAttr.get(idx, 0)).result
                cond = None
                for e in dyn:
                    c = arith.CmpIOp(arith.CmpIPredicate.sgt, e, zero).result
                    cond = c if cond is None else arith.AndIOp(cond, c).result
                guard = scf.IfOp(cond)
            with ir.InsertionPoint(guard.then_block), ir.Location.unknown():
                scf.YieldOp([])
            op.move_before(list(guard.then_block.operations)[0])
            guards[op] = cond
    for b in {id(o.operation.block): o.operation.block for o in walk_ops(block)}.values():
        _guard_barriers(b, guards)


def _guard_barriers(block, guards):
    """Each barrier of `block` whose async loads since the last on its semaphore all run under an
    `if`, under the disjunction of theirs."""
    from mlir.dialects import arith, scf
    pending = []

    def take(sem):
        got = [c for s, c in pending if s == sem]
        pending[:] = [(s, c) for s, c in pending if s != sem]
        return got

    for op in [o.operation for o in block.operations]:
        dma = op
        if op.name == "scf.if" and len(op.regions[0].blocks[0].operations) == 2:
            dma = list(op.regions[0].blocks[0].operations)[0].operation
        if (dma.name == ts.DMA and attr_int(dma, ts.ATTR_DIR) == ts.DIR_LOAD
                and ir.BoolAttr(dma.attributes[ts.ATTR_IS_ASYNC]).value):
            pending.append((dma.operands[1], guards.get(dma)))
        elif op.name == ts.MEMORY_BAR:
            conds = take(op.operands[0])
            if conds and all(c is not None for c in conds):
                with ir.InsertionPoint(op), ir.Location.unknown():
                    cond = conds[0]
                    for c in conds[1:]:
                        cond = arith.OrIOp(cond, c).result
                    guard = scf.IfOp(cond)
                with ir.InsertionPoint(guard.then_block), ir.Location.unknown():
                    scf.YieldOp([])
                op.move_before(list(guard.then_block.operations)[0])


def build_skeleton(module, stamped=None):
    """Reduce `func.func @kernel` in `module` to the skeleton+API form, in place; first, every
    segment's tile id is stamped on its ops, and the post-vcix IR so stamped written to `stamped`."""
    from ...passes.p11a_mark_runtime_attributes import refuse_unrecorded
    refuse_unrecorded(module)
    builder = TogBuilder()
    _build(module, builder)

    block = _kernel_block(module)
    if block is None:
        return "no @kernel found"
    stamp_tile_ids(builder)
    if stamped is not None:
        _write_atomic(stamped, str(module))
    gathers = _take_records(module)
    ctx = module.context
    dma_by_op = _collect_dma_nodes(builder)
    bufs = _BufferIds()

    n_compute = _emit_computes(ctx, builder, bufs, tile_ids(builder, _find_kernel(module)))
    originals, n_dma, n_wait = _emit_dmas_and_waits(ctx, block, builder, dma_by_op, bufs, gathers)
    _find_kernel(module).operation.attributes[ts.ATTR_SPAD_BUFFER_BYTES] = (
        ir.DenseI64ArrayAttr.get(_spad_buffer_bytes(module, bufs)))

    for op in originals:
        try:
            op.operation.erase()
        except Exception:
            pass
    _dce(block)
    while _strip_loop_iter_args(block, builder) or _strip_branch_results(block):
        _dce(block)

    _refuse_reads_left(block)
    _guard_live_extents(block)
    _check_replayed_loops_are_sampled(block, builder)
    return ("skeleton: compute=%d dma=%d wait=%d (store waits not lowered)"
            % (n_compute, n_dma, n_wait))


def _write_atomic(path, text):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _check_replayed_loops_are_sampled(block, builder):
    from .build_tog import _LOOP_OPS

    def carries_work(op):
        """Does anything in here become a call the TOG makes?"""
        for region in op.operation.regions:
            for blk in region.blocks:
                for inner in blk.operations:
                    if inner.operation.name in ts.OP_NAMES:
                        return True
                    if inner.operation.regions and carries_work(inner):
                        return True
        return False

    nodes = {n.op.operation for n in builder.loop_nodes}
    orphans = [op for op in walk_ops(block)
               if op.operation.name in _LOOP_OPS
               and op.operation not in nodes and carries_work(op)]
    if orphans:
        raise RuntimeError(
            f"{len(orphans)} loop(s) survive into the TOG without a node, "
            f"so gem5 measures every trip of them and TOGSim simulates them too: "
            f"{[op.operation.name for op in orphans]}")


def run_skeleton(in_path, out_path=None):
    """Read post-vcix MLIR at `in_path`, reduce to skeleton+API, write it out."""
    if out_path is None:
        out_path = in_path
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(open(in_path).read(), ctx)
        report = build_skeleton(module)
        with open(out_path, "w") as fh:
            fh.write(str(module))
    return report


def main(argv):
    import argparse

    parser = argparse.ArgumentParser(prog="build_skeleton.py")
    parser.add_argument("input")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv[1:])
    report = run_skeleton(args.input, args.out)
    import sys
    sys.stderr.write(report + "\n")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
