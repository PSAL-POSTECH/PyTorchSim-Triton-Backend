"""build_skeleton pass (C2): reduce a kernel's post-vcix MLIR to the *skeleton + API* form, in
place."""

import math
import os

from ....dialects.torchsim import TransferOp, WaitOp, is_load, moved_shape
TRANSFER, WAIT = TransferOp.OPERATION_NAME, WaitOp.OPERATION_NAME
from . import togsim_ops as ts
from ._mlir_util import walk_ops, i32, i64, i64_array, str_attr
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
    "affine.for", "scf.for", "scf.while",
    "affine.yield", "scf.yield", "func.return",
    ts.DMA, ts.COMPUTE, ts.MEMORY_BAR,
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
    """{transfer: [(dims axis, runtime extent)]} for the masked transfers the TOG sizes at run
    time: a block wider than its matrix (N = 3072 in BLOCK_N 4096) moves only what the mask leaves
    live, where the TOG charged the whole tile. The extent is the clamp's high bound held to
    [1, tile], the tile a subtile's when it is one. Gathers and pieces of a cut keep the static
    size, and so does every asynchronous load of a semaphore whose transfers and waits are not in
    one block or whose wait does not expect the loads' whole tiles."""
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
            one = arith.ConstantOp(idx, IntegerAttr.get(idx, 1)).result
            out[op.operation] = [
                (node.lead + axis, arith.MaxSIOp(arith.MinSIOp(
                    clamp[2 * i + 1],
                    arith.ConstantOp(idx, IntegerAttr.get(
                        idx, node.tile_size[node.lead + axis])).result).result,
                    one).result)
                for i, axis in enumerate(axes)]
    return out


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


def _emit_dma(ctx, dma_node, dram_index, sem, read_bufs, write_bufs, extents=()):
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
    ir.Operation.create(
        ts.DMA,
        results=[],
        operands=[dram_index, sem] + [e for _, e in extents],
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
    while True:
        tgt = None
        for op in walk_ops(block):
            n = op.operation.name
            if (n in ("affine.for", "scf.for") and len(op.operation.results) > 0
                    and _results_unused(op)):
                tgt = op
                break
        if tgt is None:
            return
        owner = None
        if builder is not None:
            for node in builder.loop_nodes:
                if node.op.operation == tgt.operation:
                    owner = node
                    break
        fresh = _rebuild_loop_no_iter(tgt)
        if owner is not None and fresh is not None:
            owner.op = fresh


def _rebuild_loop_no_iter(op):
    o = op.operation
    nres = len(o.results)
    n_in = len(o.operands)
    inits = [o.operands[n_in - nres + i] for i in range(nres)]
    keep_operands = [o.operands[i] for i in range(n_in - nres)]
    old_block = o.regions[0].blocks[0]
    oargs = list(old_block.arguments)

    attrs = {o.attributes[i].name: o.attributes[i].attr
             for i in range(len(o.attributes))}
    if "operandSegmentSizes" in attrs:
        seg = list(ir.DenseI32ArrayAttr(attrs["operandSegmentSizes"]))
        seg[-1] = 0
        attrs["operandSegmentSizes"] = ir.DenseI32ArrayAttr.get(seg)

    loc = ir.Location.unknown(o.context)
    with loc:
        new = ir.Operation.create(o.name, results=[], operands=keep_operands,
                                  attributes=attrs, regions=1, loc=loc,
                                  ip=ir.InsertionPoint(o))
        nb = new.regions[0].blocks.append(oargs[0].type)

        oargs[0].replace_all_uses_with(nb.arguments[0])
        for ba, ini in zip(oargs[1:], inits):
            ba.replace_all_uses_with(ini)
        for res, ini in zip(o.results, inits):
            res.replace_all_uses_with(ini)

        term_name = "affine.yield" if o.name == "affine.for" else "scf.yield"
        with ir.InsertionPoint(nb):
            ir.Operation.create(term_name, results=[], operands=[], loc=loc)
        new_term = list(nb.operations)[0]
        for bop in list(old_block.operations)[:-1]:
            bop.operation.move_before(new_term)
        o.erase()
    return new


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


def _emit_one_dma(ctx, op, node, builder, bufs, live):
    from . import dep_analysis as dep
    f = _transfer_fields(op)
    spad_id = bufs.of(dep.buffer_of(f["sram"]))
    read_bufs = [spad_id] if node.is_write else []
    write_bufs = [] if node.is_write else [spad_id]
    if f["offset"] is not None:
        off_id = bufs.of(dep.buffer_of(f["offset"]))
        if off_id not in read_bufs:
            read_bufs = read_bufs + [off_id]
    _emit_dma(ctx, node, f["dram_idx"], f["sem"], read_bufs, write_bufs,
              live.get(op.operation, ()))


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


def _emit_dmas_and_waits(ctx, block, builder, dma_by_op, bufs):
    originals = []
    n_dma = n_wait = 0
    live = _live_extents(block, dma_by_op)
    for op in list(walk_ops(block)):
        name = op.operation.name
        if name == TRANSFER:
            node = dma_by_op.get(op.operation)
            if node is None:
                continue
            _emit_one_dma(ctx, op, node, builder, bufs, live)
            originals.append(op)
            n_dma += 1
        elif name == WAIT:
            if _emit_one_wait(ctx, op):
                n_wait += 1
            originals.append(op)
    return originals, n_dma, n_wait


def _drop_scalar_dram_reads(block):
    from mlir.dialects import arith

    def dram(v):
        try:
            return ir.MemRefType(v.type).memory_space is None
        except Exception:
            return False

    n = 0
    for op in list(walk_ops(block)):
        o = op.operation
        if o.name not in ("memref.load", "affine.load") or not dram(o.operands[0]):
            continue
        res = o.results[0]
        ty = res.type
        with ir.InsertionPoint(o), ir.Location.unknown():
            attr = (ir.FloatAttr.get(ty, 0.0) if isinstance(ty, ir.FloatType)
                    else ir.IntegerAttr.get(ty, 0))
            zero = arith.ConstantOp(ty, attr).result
        for use in list(res.uses):
            owner = use.owner
            for i in range(len(owner.operands)):
                if owner.operands[i] == res:
                    owner.operands[i] = zero
        o.erase()
        n += 1
    return n


def build_skeleton(module, stamped=None):
    """Reduce `func.func @kernel` in `module` to the skeleton+API form, in place; first, every
    segment's tile id is stamped on its ops, and the post-vcix IR so stamped written to `stamped`."""
    builder = TogBuilder()
    _build(module, builder)

    block = _kernel_block(module)
    if block is None:
        return "no @kernel found"
    stamp_tile_ids(builder)
    if stamped is not None:
        _write_atomic(stamped, str(module))
    ctx = module.context
    dma_by_op = _collect_dma_nodes(builder)
    bufs = _BufferIds()

    n_compute = _emit_computes(ctx, builder, bufs, tile_ids(builder, _find_kernel(module)))
    originals, n_dma, n_wait = _emit_dmas_and_waits(ctx, block, builder, dma_by_op, bufs)
    _find_kernel(module).operation.attributes[ts.ATTR_SPAD_BUFFER_BYTES] = (
        ir.DenseI64ArrayAttr.get(_spad_buffer_bytes(module, bufs)))

    for op in originals:
        try:
            op.operation.erase()
        except Exception:
            pass
    _dce(block)
    _strip_loop_iter_args(block, builder)
    _drop_scalar_dram_reads(block)
    _dce(block)
    # A loop whose results only ops the last _dce took read (a row's running value handed from one
    # pop loop to the next one's inits, say) has unread results only now: strip again until none has.
    while any(op.operation.name in ("affine.for", "scf.for") and len(op.operation.results)
              and _results_unused(op) for op in walk_ops(block)):
        _strip_loop_iter_args(block, builder)
        _dce(block)

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
            f"so gem5 measures every trip of them and TOGSim replays them too: "
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
