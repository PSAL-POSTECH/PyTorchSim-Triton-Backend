"""lower_to_emitc pass (C4): skeleton+API MLIR -> EmitC -> the trace producer C++."""

import os
import subprocess

from mlir.passmanager import PassManager

from . import togsim_ops as ts
from ._mlir_util import walk_ops, i32, i64, attr_int, attr_i64_array
from .build_tog import ir, _find_kernel

ENTRY = ts.ENTRY_SYMBOL

CTX_TYPE = '!emitc.ptr<!emitc.opaque<"EmitCtx">>'

_PIPELINE = ("builtin.module("
             "convert-vector-to-scf{full-unroll=true},"
             "func.func(lower-affine),"
             "func.func(lower-vector-multi-reduction),"
             "convert-scf-to-emitc,"
             "convert-arith-to-emitc,"
             "convert-func-to-emitc)")

_INCLUDES = (
    "#include <cstddef>\n"
    "#include <cstdint>\n"
    "using std::size_t;\n"
    '#include "togsim_runtime.h"\n'
)


def _idx(v):
    return ir.IntegerAttr.get(ir.IndexType.get(), int(v))


def _opaque(ctx, text):
    return ir.Attribute.parse('#emitc.opaque<"%s">' % text, ctx)


def _arr(ctx, vals):
    vals = list(vals)
    if not vals:
        return _opaque(ctx, "nullptr")
    return _opaque(ctx, "(const int64_t[]){%s}" % ", ".join(str(int(v)) for v in vals))


def _runtime_dims(ctx, dims, dyn):
    """An int64_t[len(dims)] holding `dims` with dyn[axis], an index value, in place of dims[axis]:
    the dims of a DMA whose masked axes are sized at run time."""
    i64t = ir.IntegerType.get_signless(64)
    loc = ir.Location.unknown(ctx)
    arr = ir.Operation.create("emitc.variable",
                              results=[ir.Type.parse("!emitc.array<%dxi64>" % len(dims), ctx)],
                              attributes={"value": _opaque(ctx, "")}, loc=loc).results[0]
    for k, d in enumerate(dims):
        kc = ir.Operation.create("emitc.constant", results=[i64t],
                                 attributes={"value": ir.IntegerAttr.get(i64t, k)}, loc=loc).results[0]
        if k in dyn:
            val = ir.Operation.create("arith.index_cast", results=[i64t],
                                      operands=[dyn[k]], loc=loc).results[0]
        else:
            val = ir.Operation.create("emitc.constant", results=[i64t],
                                      attributes={"value": ir.IntegerAttr.get(i64t, int(d))},
                                      loc=loc).results[0]
        sub = ir.Operation.create("emitc.subscript", results=[_lvalue(i64t)],
                                  operands=[arr, kc], loc=loc).results[0]
        ir.Operation.create("emitc.assign", results=[], operands=[sub, val], loc=loc)
    return arr


_MEMORY_READS = ("memref.load", "affine.load", "memref.get_global", "vector.load",
                 "vector.transfer_read")


def _traceable(value, entry):
    """(whether the trace can compute `value`, the ops it is built from). The trace has the
    program ids, already the grid loop's ivs, and its own loop ivs; a kernel argument still
    standing is a size shape_args does not carry, and a memory read is data it never has."""
    ops, todo, seen, ok = [], [value], [], True
    while todo:
        v = todo.pop()
        owner = v.owner
        if isinstance(owner, ir.Block):
            ok = ok and owner != entry
            continue
        op = owner.operation
        if op in seen:
            continue
        seen.append(op)
        if op.name in _MEMORY_READS or op.regions:
            ok = False
            continue
        ops.append(op)
        todo.extend(op.operands)
    return ok, ops


def _drop_untraceable_extents(kernel):
    """Put the static sizes back on every DMA and barrier of the kernel, and erase what fed the
    runtime ones, when any of them needs a value the trace cannot compute: a barrier counts what
    its loads moved, so the two are static together or not at all."""
    entry = kernel.regions[0].blocks[0]
    sized = []
    for op in walk_ops(entry):
        o = op.operation
        if o.name == ts.DMA and ts.ATTR_DYN_DIMS in o.attributes:
            sized.append((o, 2))
        elif o.name == ts.MEMORY_BAR and len(o.operands) > 1:
            sized.append((o, 1))
    verdicts = [_traceable(v, entry) for o, keep in sized for v in list(o.operands)[keep:]]
    if all(ok for ok, _ in verdicts):
        return
    for o, keep in sized:
        attrs = {name: o.attributes[name] for name in o.attributes if name != ts.ATTR_DYN_DIMS}
        ir.Operation.create(o.name, results=[], operands=list(o.operands)[:keep],
                            attributes=attrs, loc=o.location, ip=ir.InsertionPoint(o))
        o.erase()
    dead = []
    for _, ops in verdicts:
        dead += [op for op in ops if op not in dead]
    changed = True
    while changed:
        changed = False
        for op in list(dead):
            if all(not list(r.uses) for r in op.results):
                dead.remove(op)
                op.erase()
                changed = True


def _attr_bool(op, key):
    return 1 if ir.BoolAttr(op.operation.attributes[key]).value else 0


def _strip_aux(module, keep=None):
    keep_op = keep.operation if keep is not None else None
    victims = []
    for op in module.body.operations:
        name = op.operation.name
        if name == "memref.global":
            victims.append(op)
        elif name == "func.func":
            if keep_op is not None:
                if op.operation != keep_op:
                    victims.append(op)
            elif ir.StringAttr(op.operation.attributes["sym_name"]).value != "kernel":
                victims.append(op)
    for op in victims:
        op.operation.erase()


def _emit_spad_buffer_bytes(module, kernel):
    """The kernel's spad buffer sizes (build_skeleton's table) as the two constants trace ABI 15
    has a producer export: togsim_spad_buffer_count and togsim_spad_buffer_bytes[]."""
    import numpy as np
    attrs = kernel.operation.attributes
    sizes = list(ir.DenseI64ArrayAttr(attrs[ts.ATTR_SPAD_BUFFER_BYTES]))
    del attrs[ts.ATTR_SPAD_BUFFER_BYTES]
    with ir.InsertionPoint.at_block_begin(module.body), ir.Location.unknown():
        i32, i64 = ir.IntegerType.get_signless(32), ir.IntegerType.get_signless(64)
        consts = [(ts.SPAD_BUFFER_COUNT_SYMBOL, i32, ir.IntegerAttr.get(i32, len(sizes)))]
        if sizes:
            consts.append((ts.SPAD_BUFFER_BYTES_SYMBOL,
                           ir.Type.parse("!emitc.array<%dxi64>" % len(sizes)),
                           ir.DenseElementsAttr.get(np.array(sizes, dtype=np.int64),
                                                    type=ir.RankedTensorType.get([len(sizes)], i64))))
        for name, ty, value in consts:
            ir.Operation.create("emitc.global", attributes={
                "sym_name": ir.StringAttr.get(name), "type": ir.TypeAttr.get(ty),
                "initial_value": value, "extern_specifier": ir.UnitAttr.get(),
                "const_specifier": ir.UnitAttr.get()})


class WorkItem:
    """A kernel whose body is ONE work-item, plus the grid over it; `key_shifts` places each
    axis's program id in the work-item's key (kernel_object.PROGRAM_KEY_SHIFT)."""

    def __init__(self, parallel_args, grid, key_shifts=None):
        if len(parallel_args) != len(grid):
            raise ValueError(
                f"parallel_args {parallel_args} and grid {grid} must have the "
                f"same length -- one program-id argument per grid axis")
        self.parallel_args = list(parallel_args)
        self.grid = [None if g is None else int(g) for g in grid]
        self.key_shifts = list(key_shifts) if key_shifts is not None else [0] * len(grid)
        if len(self.key_shifts) != len(grid):
            raise ValueError(f"key_shifts {key_shifts}: one per grid axis, {len(grid)}")

    @property
    def dynamic_axes(self):
        """Indices into `grid` whose extent arrives at run time."""
        return [i for i, g in enumerate(self.grid) if g is None]


def _materialize_grid_loop(kernel, work_item, ctx):
    from mlir.dialects import arith, scf

    block = kernel.regions[0].blocks[0]
    idxty = ir.IndexType.get()
    loc = ir.Location.unknown(ctx)

    pid_args = [block.arguments[i] for i in work_item.parallel_args]
    body_ops = [o for o in block.operations
                if o.operation.name not in _LOOP_TERMINATORS]
    terminator = [o for o in block.operations
                  if o.operation.name in _LOOP_TERMINATORS][0]

    with ir.InsertionPoint(terminator), loc:
        c0 = arith.ConstantOp(idxty, ir.IntegerAttr.get(idxty, 0)).result
        c1 = arith.ConstantOp(idxty, ir.IntegerAttr.get(idxty, 1)).result
        ubs = [arith.ConstantOp(idxty, ir.IntegerAttr.get(idxty, e or 1)).result
               for e in work_item.grid]

    loops, inner = [], None
    for ub in ubs:
        ip = ir.InsertionPoint(terminator) if inner is None \
            else ir.InsertionPoint.at_block_terminator(inner.body)
        with ip, loc:
            loop = scf.ForOp(c0, ub, c1)
        if len(loop.body.operations) == 0:
            with ir.InsertionPoint(loop.body), loc:
                scf.YieldOp([])
        loop.operation.attributes["outer_loop"] = ir.BoolAttr.get(True)
        loops.append(loop)
        inner = loop

    inner_block = inner.body
    inner_term = inner_block.operations[len(inner_block.operations) - 1]
    for op in body_ops:
        op.operation.move_before(inner_term)

    with ir.InsertionPoint(inner_block.operations[0]), loc:
        casts = []
        for loop, pid in zip(loops, pid_args):
            iv = loop.body.arguments[0]
            casts.append(arith.IndexCastOp(pid.type, iv).result
                         if pid.type != idxty else iv)

    for pid, new in zip(pid_args, casts):
        _replace_all_uses(pid, new)

    return [(loops[i], ubs[i]) for i in work_item.dynamic_axes]


def _bind_runtime_bounds(pending, shape_arg, ctx):
    if not pending:
        return
    from mlir.dialects import arith

    i64 = ir.IntegerType.get_signless(64)
    idxty = ir.IndexType.get()
    loc = ir.Location.unknown(ctx)
    for k, (loop, placeholder) in enumerate(pending):
        with ir.InsertionPoint(placeholder.owner), loc:
            kc = ir.Operation.create(
                "emitc.constant", results=[i64],
                attributes={"value": ir.IntegerAttr.get(i64, k)}).results[0]
            ref = ir.Operation.create(
                "emitc.subscript", results=[_lvalue(i64)],
                operands=[shape_arg, kc]).results[0]
            elem = ir.Operation.create(
                "emitc.load", results=[i64], operands=[ref]).results[0]
            bound = arith.IndexCastOp(idxty, elem).result
        _replace_all_uses(placeholder, bound)
        placeholder.owner.erase()


def _lvalue(ty):
    return ir.Type.parse(f"!emitc.lvalue<{ty}>", ty.context)


def _replace_all_uses(old, new):
    for use in list(old.uses):
        owner = use.owner
        for i in range(len(owner.operands)):
            if owner.operands[i] == old:
                owner.operands[i] = new


def _rewrite_signature(kernel, ctx):
    block = kernel.regions[0].blocks[0]
    for i, arg in enumerate(block.arguments):
        uses = list(arg.uses)
        if not uses:
            continue
        what = ("the DCE left a tensor-data op"
                if str(arg.type).startswith("memref")
                else "the trace would need its value, and shape_args carries "
                     "the grid extents only")
        raise ValueError(
            f"kernel arg {i} ({arg.type}) still used after build_skeleton, by "
            f"`{str(uses[0].owner).split(chr(10))[0].strip()[:60]}` -- {what}")
    for i in reversed(range(len(block.arguments))):
        block.erase_argument(i)
    ptr = ir.Type.parse(CTX_TYPE, ctx)
    i64ptr = ir.Type.parse("!emitc.ptr<i64>", ctx)
    i32 = ir.IntegerType.get_signless(32)
    loc = ir.Location.unknown(ctx)
    block.add_argument(ptr, loc)
    block.add_argument(i64ptr, loc)
    block.add_argument(i32, loc)
    kernel.operation.attributes["function_type"] = ir.TypeAttr.get(
        ir.FunctionType.get([ptr, i64ptr, i32], []))
    kernel.operation.attributes["sym_name"] = ir.StringAttr.get(ENTRY)
    return block.arguments[0]


def _call(ctx, ctx_val, op, callee, arg_attrs):
    ir.Operation.create(
        "emitc.call_opaque", results=[], operands=[ctx_val],
        attributes={"callee": ir.StringAttr.get(callee),
                    "args": ir.ArrayAttr.get([_idx(0)] + arg_attrs)},
        loc=ir.Location.unknown(ctx), ip=ir.InsertionPoint(op))


def _innermost_outer_loop(block):
    found = [None]

    def is_outer(op):
        a = op.operation.attributes
        return "outer_loop" in a and ir.BoolAttr(a["outer_loop"]).value

    def walk(b):
        for op in b.operations:
            if op.operation.name == "affine.for" and is_outer(op):
                found[0] = op
            for r in op.operation.regions:
                for bb in r.blocks:
                    walk(bb)

    walk(block)
    return found[0]


def _is_outer(forop):
    a = forop.operation.attributes
    return "outer_loop" in a and ir.BoolAttr(a["outer_loop"]).value


_LOOP_OPS = ("affine.for", "scf.for")
_LOOP_TERMINATORS = ("affine.yield", "scf.yield", "func.return")


def _parallel_loop_chain(block):
    chain = []
    cur = block
    while True:
        nxt = None
        for op in cur.operations:
            if op.operation.name in _LOOP_OPS and _is_outer(op):
                nxt = op
                break
        if nxt is None:
            break
        chain.append(nxt)
        cur = nxt.operation.regions[0].blocks[0]
    return chain


def _const_op(value):
    owner = value.owner
    if isinstance(owner, ir.Block):
        return None
    return owner if owner.name in ("arith.constant", "emitc.constant") else None


def _work_item_key(ctx, ivs, key_shifts, loc):
    """The work-item's key as an i64: each program id shifted into its field, or'ed."""
    from mlir.dialects import arith
    i64 = ir.IntegerType.get_signless(64)
    key = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, 0), loc=loc).result
    for iv, shift in zip(ivs, key_shifts):
        v = arith.IndexCastOp(i64, iv, loc=loc).result
        s = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, shift), loc=loc).result
        key = arith.OrIOp(key, arith.ShLIOp(v, s, loc=loc).result, loc=loc).result
    return key


def _outline_work_item(ctx, kernel, ctx_val, key_shifts=()):
    kblk = kernel.regions[0].blocks[0]
    chain = _parallel_loop_chain(kblk)
    if chain:
        L = chain[-1]
        Lbody = L.operation.regions[0].blocks[0]
        ivs = [c.operation.regions[0].blocks[0].arguments[0] for c in chain]
    else:
        L = None
        Lbody = kblk
        ivs = []

    i64 = ir.IntegerType.get_signless(64)
    i32 = ir.IntegerType.get_signless(32)
    idxty = ir.IndexType.get()
    ctxty = ir.Type.parse(CTX_TYPE, ctx)
    i64ptr = ir.Type.parse("!emitc.ptr<i64>", ctx)
    loc = ir.Location.unknown(ctx)

    tile = ir.Operation.create(
        "func.func", results=[], regions=1,
        attributes={
            "function_type": ir.TypeAttr.get(ir.FunctionType.get([ctxty, i64ptr, i32], [])),
            "sym_name": ir.StringAttr.get(ts.TILE_SYMBOL),
            "sym_visibility": ir.StringAttr.get("private")},
        loc=loc, ip=ir.InsertionPoint(kernel))
    with loc:
        tblk = tile.regions[0].blocks.append(ctxty, i64ptr, i32)
    ctx2, iv2, _n2 = tblk.arguments
    with ir.InsertionPoint(tblk):
        tret = ir.Operation.create("func.return", results=[], operands=[], loc=loc)

    idx_vals = []
    with ir.InsertionPoint(tret):
        for k in range(len(ivs)):
            kc = ir.Operation.create("emitc.constant", results=[i64],
                    attributes={"value": ir.IntegerAttr.get(i64, k)}, loc=loc).results[0]
            ref = ir.Operation.create("emitc.subscript", results=[_lvalue(i64)],
                    operands=[iv2, kc], loc=loc).results[0]
            elem = ir.Operation.create("emitc.load", results=[i64],
                    operands=[ref], loc=loc).results[0]
            idx_vals.append(ir.Operation.create("arith.index_cast", results=[idxty],
                    operands=[elem], loc=loc).results[0])

    for op in [o for o in Lbody.operations
               if o.operation.name not in _LOOP_TERMINATORS]:
        op.operation.move_before(tret)

    caps = [(ctx_val, ctx2)] + list(zip(ivs, idx_vals))
    internal_consts = []
    def _collect_internal(block):
        for op in block.operations:
            c = _const_op(op.operation.results[0]) if len(op.operation.results) == 1 else None
            if c is not None:
                internal_consts.append(op.operation.results[0])
            for rg in op.operation.regions:
                for b in rg.blocks:
                    _collect_internal(b)
    _collect_internal(tblk)
    const_clones = []
    ext_consts = []
    def _find_ext_consts(block):
        for op in block.operations:
            for opnd in op.operation.operands:
                if _const_op(opnd) is None:
                    continue
                if any(opnd == ic for ic in internal_consts):
                    continue
                if any(opnd == e for e in ext_consts):
                    continue
                ext_consts.append(opnd)
            for rg in op.operation.regions:
                for b in rg.blocks:
                    _find_ext_consts(b)
    _find_ext_consts(tblk)
    top = ir.InsertionPoint(tblk.operations[0])
    for e in ext_consts:
        c = _const_op(e)
        clone = ir.Operation.create(c.name, results=[e.type],
                    attributes={"value": c.attributes["value"]}, loc=loc, ip=top).results[0]
        const_clones.append((e, clone))

    allcaps = caps + const_clones
    def _remap(block):
        for op in block.operations:
            for i in range(len(op.operation.operands)):
                cur = op.operation.operands[i]
                for orig, new in allcaps:
                    if cur == orig:
                        op.operation.operands[i] = new
                        break
            for rg in op.operation.regions:
                for b in rg.blocks:
                    _remap(b)
    _remap(tblk)

    term = [o for o in Lbody.operations
            if o.operation.name in _LOOP_TERMINATORS][0]
    fn_ref = _opaque(ctx, ts.TILE_SYMBOL)
    if ivs and len(key_shifts) != len(ivs):
        raise ValueError(f"{len(ivs)} parallel loop(s) and {len(key_shifts)} key field(s): "
                         f"the work-item's key would name another program")
    with ir.InsertionPoint(term):
        if ivs:
            arrty = ir.Type.parse("!emitc.array<%dxi64>" % len(ivs), ctx)
            arr = ir.Operation.create("emitc.variable", results=[arrty],
                    attributes={"value": _opaque(ctx, "")}, loc=loc).results[0]
            for k, iv in enumerate(ivs):
                kc = ir.Operation.create("emitc.constant", results=[i64],
                        attributes={"value": ir.IntegerAttr.get(i64, k)}, loc=loc).results[0]
                v64 = ir.Operation.create("arith.index_cast", results=[i64],
                        operands=[iv], loc=loc).results[0]
                sub = ir.Operation.create("emitc.subscript", results=[_lvalue(i64)],
                        operands=[arr, kc], loc=loc).results[0]
                ir.Operation.create("emitc.assign", results=[], operands=[sub, v64], loc=loc)
            key = _work_item_key(ctx, ivs, key_shifts, loc)
            ir.Operation.create(
                "emitc.call_opaque", results=[], operands=[ctx_val, arr, key],
                attributes={"callee": ir.StringAttr.get(ts.DISPATCH_CALLEE),
                            "args": ir.ArrayAttr.get(
                                [_idx(0), fn_ref, _idx(1), ir.IntegerAttr.get(i32, len(ivs)),
                                 _idx(2)])},
                loc=loc)
        else:
            ir.Operation.create(
                "emitc.call_opaque", results=[], operands=[ctx_val],
                attributes={"callee": ir.StringAttr.get(ts.DISPATCH_CALLEE),
                            "args": ir.ArrayAttr.get(
                                [_idx(0), fn_ref, _opaque(ctx, "nullptr"), ir.IntegerAttr.get(i32, 0),
                                 ir.IntegerAttr.get(i64, 0)])},
                loc=loc)


def _rewrite_togsim_ops(ctx, kernel, ctx_val):
    block = kernel.regions[0].blocks[0]
    victims = []
    for op in walk_ops(block):
        name = op.operation.name
        ipo = ir.InsertionPoint(op)
        if name == ts.DMA:
            dims = attr_i64_array(op, ts.ATTR_DIMS)
            dram_operand, sem_operand, *extents = op.operation.operands
            operands = [ctx_val, dram_operand, sem_operand]
            dims_arg = _arr(ctx, dims)
            if extents:
                with ipo:
                    operands.append(_runtime_dims(ctx, dims, dict(zip(
                        attr_i64_array(op, ts.ATTR_DYN_DIMS), extents))))
                dims_arg = _idx(3)
            args = [_idx(0),
                    i32(attr_int(op, ts.ATTR_DIR)),
                    i32(attr_int(op, ts.ATTR_ARG_ID)),
                    _idx(1),
                    i32(len(dims)),
                    dims_arg,
                    _arr(ctx, attr_i64_array(op, ts.ATTR_STRIDES)),
                    i32(attr_int(op, ts.ATTR_ELEM_BITS)),
                    i32(_attr_bool(op, ts.ATTR_IS_ASYNC)),
                    _idx(2)]
            _rb = attr_i64_array(op, ts.ATTR_READ_BUFS)
            _wb = attr_i64_array(op, ts.ATTR_WRITE_BUFS)
            args += [_arr(ctx, _rb), i32(len(_rb)), _arr(ctx, _wb), i32(len(_wb)),
                     i32(attr_int(op, ts.ATTR_INDIRECT))]
            ir.Operation.create(
                "emitc.call_opaque", results=[], operands=operands,
                attributes={"callee": ir.StringAttr.get(ts.EMITC_CALLEE[ts.DMA]),
                            "args": ir.ArrayAttr.get(args)},
                loc=ir.Location.unknown(ctx), ip=ipo)
            victims.append(op)
        elif name == ts.MEMORY_BAR:
            sem_operand, *live = op.operation.operands
            operands = [ctx_val, sem_operand]
            expected = i64(attr_int(op, ts.ATTR_EXPECTED))
            if live:
                operands.append(ir.Operation.create(
                    "arith.index_cast", results=[ir.IntegerType.get_signless(64)],
                    operands=live, loc=ir.Location.unknown(ctx), ip=ipo).results[0])
                expected = _idx(2)
            ir.Operation.create(
                "emitc.call_opaque", results=[], operands=operands,
                attributes={"callee": ir.StringAttr.get(ts.EMITC_CALLEE[ts.MEMORY_BAR]),
                            "args": ir.ArrayAttr.get([_idx(0), _idx(1), expected])},
                loc=ir.Location.unknown(ctx), ip=ipo)
            victims.append(op)
        elif name == ts.COMPUTE:
            _rb = attr_i64_array(op, ts.ATTR_READ_BUFS)
            _wb = attr_i64_array(op, ts.ATTR_WRITE_BUFS)
            _call(ctx, ctx_val, op, ts.EMITC_CALLEE[ts.COMPUTE],
                  [i64(attr_int(op, ts.ATTR_TILE_ID)),
                   i32(attr_int(op, ts.ATTR_COMPUTE_TYPE)),
                   i32(0), _opaque(ctx, "nullptr"),
                   _arr(ctx, _rb), i32(len(_rb)), _arr(ctx, _wb), i32(len(_wb))])
            victims.append(op)
    for op in victims:
        op.operation.erase()


def _retype_for_to_size_t(module):
    idx = ir.IndexType.get()
    st = ir.Type.parse("!emitc.size_t", module.context)

    for op in list(walk_ops(module.body)):
        if op.operation.name == "emitc.for":
            op.operation.regions[0].blocks[0].arguments[0].set_type(st)

    dead = []
    for op in list(walk_ops(module.body)):
        if op.operation.name != "builtin.unrealized_conversion_cast":
            continue
        res = op.results[0]
        src = list(op.operation.operands)[0]
        if src.type in (idx, st) and res.type in (idx, st):
            res.replace_all_uses_with(src)
            dead.append(op)
    for d in dead:
        try:
            d.operation.erase()
        except Exception:
            pass

    for op in list(walk_ops(module.body)):
        if op.operation.name != "emitc.for":
            continue
        bounds = {v.type for v in list(op.operation.operands)[:3]}
        if len(bounds) == 1:
            op.operation.regions[0].blocks[0].arguments[0].set_type(bounds.pop())


def _add_extern_c(module, ctx):
    for op in module.body.operations:
        if (op.operation.name == "emitc.func"
                and ir.StringAttr(op.operation.attributes["sym_name"]).value == ENTRY):
            op.operation.attributes["specifiers"] = ir.ArrayAttr.get(
                [ir.StringAttr.get('extern "C"')])
            return
    raise ValueError("emitc.func @%s not found after conversion" % ENTRY)


_INT_MINMAX = {"arith.minsi": "slt", "arith.maxsi": "sgt",
               "arith.minui": "ult", "arith.maxui": "ugt"}


def _expand_int_minmax(module):
    from mlir.dialects import arith
    victims = [(op, _INT_MINMAX[op.operation.name]) for op in walk_ops(module.body)
               if op.operation.name in _INT_MINMAX]
    done = 0
    for op, pred in victims:
        a, b = op.operation.operands
        with ir.InsertionPoint(op.operation), op.operation.location:
            cond = arith.CmpIOp(arith.CmpIPredicate[pred], a, b).result
            new = arith.SelectOp(cond, a, b).result
        _replace_all_uses(op.operation.results[0], new)
        op.operation.erase()
        done += 1
    return done


def lower_to_emitc(skeleton_module, work_item=None):
    """Lower a skeleton+API module (in place) to an EmitC module with the `togsim_kernel` entry
    function."""
    ctx = skeleton_module.context
    kernel = _find_kernel(skeleton_module)
    if kernel is None:
        raise ValueError("no kernel function found in skeleton module")

    _strip_aux(skeleton_module, keep=kernel)
    _emit_spad_buffer_bytes(skeleton_module, kernel)
    pending = []
    if work_item is not None:
        pending = _materialize_grid_loop(kernel, work_item, ctx)
    _drop_untraceable_extents(kernel)
    ctx_val = _rewrite_signature(kernel, ctx)
    _bind_runtime_bounds(pending, kernel.regions[0].blocks[0].arguments[1], ctx)
    _rewrite_togsim_ops(ctx, kernel, ctx_val)
    _outline_work_item(ctx, kernel, ctx_val,
                       work_item.key_shifts if work_item is not None else ())
    _expand_int_minmax(skeleton_module)

    PassManager.parse(_PIPELINE, ctx).run(skeleton_module.operation)

    _retype_for_to_size_t(skeleton_module)
    _add_extern_c(skeleton_module, ctx)
    return skeleton_module


def trace_banner():
    """The generated file's header: the ABI version and the call-format block."""
    import re

    from .... import toolchain
    path = os.path.join(toolchain.TOGSIM_INCLUDE, "togsim_runtime.h")
    head = "// GENERATED by pytorchsim-triton-compiler -- TOGSim trace producer. DO NOT EDIT.\n"
    try:
        text = open(path).read()
    except OSError:
        return head
    ver = re.search(r"#define\s+TOGSIM_ABI_VERSION\s+(\d+)", text)
    blk = re.search(r"// --- BEGIN trace-producer call formats[^\n]*\n(.*?)"
                    r"// --- END trace-producer call formats[^\n]*\n", text, re.S)
    head = ("// GENERATED by pytorchsim-triton-compiler -- TOGSim trace producer "
            "(ABI v%s). DO NOT EDIT.\n" % (ver.group(1) if ver else "?"))
    return head + (blk.group(1) if blk else "")


def emitc_to_cpp(emitc_module):
    """Render `emitc_module` to C++: banner, includes, and the mlir-to-cpp body."""
    from .... import toolchain
    proc = subprocess.run(
        [toolchain.MLIR_TRANSLATE, "--mlir-to-cpp"],
        input=str(emitc_module), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("mlir-translate --mlir-to-cpp failed:\n" + proc.stderr)
    return trace_banner() + _INCLUDES + proc.stdout


def compile_so(cpp_path, so_path):
    """Compile the producer C++ to `so_path`."""
    from .... import toolchain
    cmd = [toolchain.HOST_CXX, "-shared", "-fPIC", "-std=gnu++17", "-O2",
           "-I", toolchain.TOGSIM_INCLUDE, cpp_path, "-o", so_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("%s failed:\n%s" % (" ".join(cmd), proc.stderr))
    return so_path


def main(argv):
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="pytorchsim_triton_compiler.lowering.tog.passes.lower_to_emitc")
    parser.add_argument("input", help="post-vcix kernel .mlir")
    parser.add_argument("--emit-cpp", required=True, help="write the C++ here")
    parser.add_argument("--so", default=None, help="also compile it to this .so")
    parser.add_argument("--emit-mlir", default=None,
                        help="also write the EmitC MLIR here")
    args = parser.parse_args(argv[1:])

    from . import build_skeleton as bs
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(open(args.input).read(), ctx)
        bs.build_skeleton(module)
        emitc = lower_to_emitc(module)
        if args.emit_mlir:
            open(args.emit_mlir, "w").write(str(emitc))
        open(args.emit_cpp, "w").write(emitc_to_cpp(emitc))
        if args.so:
            compile_so(args.emit_cpp, args.so)
    sys.stderr.write("wrote %s\n" % (args.so or args.emit_cpp))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
