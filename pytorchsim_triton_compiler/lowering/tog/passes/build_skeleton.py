"""build_skeleton pass (C2): reduce a kernel's post-vcix MLIR to the *skeleton + API* form, in
place."""

from ....dialects.torchsim import TransferOp, WaitOp, is_load
TRANSFER, WAIT = TransferOp.OPERATION_NAME, WaitOp.OPERATION_NAME
from . import togsim_ops as ts
from ._mlir_util import walk_ops, i32, i64, i64_array, str_attr
from .build_tog import (
    ir,
    TogBuilder,
    _build,
    _reset_ids,
    _find_kernel,
    _is_block_arg,
    TOGDMANode,
    TOGDMAWaitNode,
    _COMPUTE_TYPE_NAME,
)

MARKERS = (TRANSFER, WAIT)

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


def _emit_dma(ctx, dma_node, dram_index, sem, read_bufs, write_bufs):
    op = dma_node.op
    attrs = {
        ts.ATTR_DIR: i32(ts.DIR_STORE if dma_node.is_write else ts.DIR_LOAD),
        ts.ATTR_DIMS: i64_array(dma_node.tile_size),
        ts.ATTR_STRIDES: i64_array(dma_node.tile_stride),
        ts.ATTR_ELEM_BITS: i32(dma_node.element_size),
        ts.ATTR_IS_ASYNC: ir.BoolAttr.get(bool(dma_node.is_async)),
        ts.ATTR_ARG_ID: i32(_arg_id_of(dma_node.base_addr)),
        ts.ATTR_INDIRECT: i32(int(dma_node.indirect_mode)),
        "base": str_attr(dma_node.base_addr),
        ts.ATTR_READ_BUFS: i64_array(read_bufs),
        ts.ATTR_WRITE_BUFS: i64_array(write_bufs),
    }
    ir.Operation.create(
        ts.DMA,
        results=[],
        operands=[dram_index, sem],
        attributes=attrs,
        loc=ir.Location.unknown(ctx),
        ip=ir.InsertionPoint(op),
    )


def _emit_memory_bar(ctx, anchor_op, sem, expected):
    ir.Operation.create(
        ts.MEMORY_BAR, results=[], operands=[sem],
        attributes={ts.ATTR_EXPECTED: i64(expected)},
        loc=ir.Location.unknown(ctx), ip=ir.InsertionPoint(anchor_op))


def _emit_compute(ctx, compute_node, tile_id, read_bufs, write_bufs):
    front = compute_node.operations[0]
    attrs = {
        ts.ATTR_TILE_ID: i64(tile_id),
        ts.ATTR_COMPUTE_TYPE: i32(int(compute_node.compute_type)),
        "compute_type_name": str_attr(_COMPUTE_TYPE_NAME[compute_node.compute_type]),
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
        seg = [int(x) for x in str(attrs["operandSegmentSizes"]).split(":")[1].strip(" >").split(",")]
        seg[-1] = 0
        attrs["operandSegmentSizes"] = ir.Attribute.parse(
            "array<i32: " + ", ".join(str(s) for s in seg) + ">")

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
        if name is None:
            return -1
        return self._ids.setdefault(name, len(self._ids))


def _emit_computes(ctx, builder, bufs):
    from . import dep_analysis as dep
    n = 0
    for tile_id, cn in enumerate(builder.compute_nodes):
        if not cn.operations:
            continue
        reads, writes = dep.compute_buffers(cn, builder.grouping)
        _emit_compute(ctx, cn, tile_id,
                      sorted(bufs.of(b) for b in reads),
                      sorted(bufs.of(b) for b in writes))
        n += 1
    return n


def _transfer_fields(op):
    tx = op.operation.opview
    return {"dram": tx.dram, "dram_idx": tx.dram_idx, "sram": tx.sram,
            "sram_idx": tx.sram_idx, "sem": tx.sem,
            "offset": tx.index}


def _emit_one_dma(ctx, op, node, builder, bufs):
    from . import dep_analysis as dep
    f = _transfer_fields(op)
    spad_id = bufs.of(dep._global_of(f["sram"]))
    read_bufs = [spad_id] if node.is_write else []
    write_bufs = [] if node.is_write else [spad_id]
    if f["offset"] is not None:
        off_owner = f["offset"].owner
        off_sym = str(off_owner.attributes["name"]).strip('@" ')
        off_id = bufs.of(off_sym)
        if off_id not in read_bufs:
            read_bufs = read_bufs + [off_id]
    _emit_dma(ctx, node, f["dram_idx"], f["sem"], read_bufs, write_bufs)


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
    expected = _get_const_index(wait.amount)
    if expected is None:
        raise RuntimeError(f"a wait whose amount is not a constant cannot be traced: {op}")
    _emit_memory_bar(ctx, op, wait.sem, expected)
    return True


def _get_const_index(v):
    if v is None or _is_block_arg(v) or v.owner.name != "arith.constant":
        return None
    return ir.IntegerAttr(v.owner.attributes["value"]).value


def _emit_dmas_and_waits(ctx, block, builder, dma_by_op, bufs):
    originals = []
    n_dma = n_wait = 0
    for op in list(walk_ops(block)):
        name = op.operation.name
        if name == TRANSFER:
            node = dma_by_op.get(op.operation)
            if node is None:
                continue
            _emit_one_dma(ctx, op, node, builder, bufs)
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
            attr = (ir.FloatAttr.get(ty, 0.0) if str(ty).startswith("f")
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


def build_skeleton(module, grouping="typed"):
    """Reduce `func.func @kernel` in `module` to the skeleton+API form, in place."""
    _reset_ids()
    builder = TogBuilder(grouping)
    _build(module, builder)

    block = _kernel_block(module)
    if block is None:
        return "no @kernel found"
    ctx = module.context
    dma_by_op = _collect_dma_nodes(builder)
    bufs = _BufferIds()

    n_compute = _emit_computes(ctx, builder, bufs)
    originals, n_dma, n_wait = _emit_dmas_and_waits(ctx, block, builder, dma_by_op, bufs)

    for op in originals:
        try:
            op.operation.erase()
        except Exception:
            pass
    _dce(block)
    _strip_loop_iter_args(block, builder)
    _drop_scalar_dram_reads(block)
    _dce(block)

    _check_replayed_loops_are_sampled(block, builder)
    return ("skeleton: compute=%d dma=%d wait=%d (store waits not lowered)"
            % (n_compute, n_dma, n_wait))


def _check_replayed_loops_are_sampled(block, builder):
    from .build_tog import _LOOP_OPS

    def carries_work(op):
        """Does anything in here become a TRACE EVENT?"""
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
            f"{len(orphans)} loop(s) survive into the trace without a TOG node, "
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
