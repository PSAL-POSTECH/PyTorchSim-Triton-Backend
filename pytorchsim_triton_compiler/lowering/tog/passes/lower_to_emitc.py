"""lower_to_emitc pass (C4): skeleton+API MLIR -> EmitC -> the TOG's C++, written against TOGSim's
togsim.hpp (TOGSim API v1) with the same types and calls a hand-written TOG uses."""

import os
import subprocess

from mlir.passmanager import PassManager

from ....contract.kernel_object import CYCLE_CALLS
from ....contract.stop import Reason, Refused
from . import togsim_ops as ts
from ._mlir_util import walk_ops, attr_int, attr_i64_array
from .build_tog import ir, _find_kernel

ENTRY = "tog_entry"
WORK_ITEM = "tog_work_item"
ARGS = "WorkItemArgs"

VARIANT = CYCLE_CALLS - 1
"""The variant every compute names: gem5's last call of its tile, row `id + count * VARIANT` of the
cycle and unit tables, which hold every call c of tile t at row t + count * c."""

_PIPELINE = ("builtin.module("
             "convert-vector-to-scf{full-unroll=true},"
             "func.func(lower-affine),"
             "func.func(lower-vector-multi-reduction),"
             "convert-scf-to-emitc,"
             "convert-arith-to-emitc,"
             "convert-func-to-emitc)")

_INCLUDES = '#include "togsim.hpp"\n'


def _idx(v):
    return ir.IntegerAttr.get(ir.IndexType.get(), int(v))


def _opaque(ctx, text):
    return ir.Attribute.parse('#emitc.opaque<"%s">' % text.replace("\\", "\\\\").replace('"', '\\"'),
                              ctx)


def _type(ctx, cxx):
    return ir.Type.parse('!emitc.opaque<"%s">' % cxx, ctx)


def _buf(i):
    return f"BUF{int(i)}"


def _bufs(ids):
    return "{%s}" % ", ".join(_buf(b) for b in ids)


def _node(t):
    return f"NODE{int(t)}"


def _call(ctx, callee, operands, result=None, args=None, template=None):
    """An emitc.call_opaque of `callee` at the current insertion point; `result` is the C++ type
    it returns, if any."""
    attrs = {"callee": ir.StringAttr.get(callee)}
    if args is not None:
        attrs["args"] = ir.ArrayAttr.get(args)
    if template is not None:
        attrs["template_args"] = ir.ArrayAttr.get(template)
    op = ir.Operation.create("emitc.call_opaque", results=[_type(ctx, result)] if result else [],
                             operands=operands, attributes=attrs, loc=ir.Location.unknown(ctx))
    return op.results[0] if result else None


def _dims(ctx, vals):
    """togsim::Dims(...) of `vals`, each an int or an i64 value."""
    i64t = ir.IntegerType.get_signless(64)
    operands, args = [], []
    for v in vals:
        if isinstance(v, int):
            args.append(ir.IntegerAttr.get(i64t, v))
        else:
            args.append(_idx(len(operands)))
            operands.append(v)
    return _call(ctx, "togsim::Dims", operands, "togsim::Dims", args)


def _touches_memory(op):
    """Does `op` make a buffer or read one, by its results' types and its effects (UNKNOWN when
    it states none): what it gives is data the TOG never has."""
    from ....dialects.pytorchsim import effects_on
    if any(isinstance(r.type, ir.MemRefType) for r in op.results):
        return True
    return any(isinstance(v.type, ir.MemRefType) and effects_on(op, v) & {"read", "unknown"}
               for v in op.operands)


def _tog_computable(value, entry):
    """(whether the TOG can compute `value`, the ops it is built from). The TOG has the
    program ids, already the grid loop's ivs, and its own loop ivs; a kernel argument still
    standing is one shape_args does not carry (a float), and a memory read is data it never has."""
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
        if op.regions or _touches_memory(op):
            ok = False
            continue
        ops.append(op)
        todo.extend(op.operands)
    return ok, ops


def _refuse_uncomputable_extents(kernel):
    """Every DMA extent sized at run time is one the TOG computes: from program ids, its loops' ivs,
    the launch's integer scalars, runtime attributes and constants."""
    entry = kernel.regions[0].blocks[0]
    for op in walk_ops(entry):
        o = op.operation
        if o.name == ts.DMA and ts.ATTR_DYN_DIMS in o.attributes:
            for v in list(o.operands)[2:]:
                if not _tog_computable(v, entry)[0]:
                    raise Refused(Reason.RUNTIME_ATTRIBUTE, "runtime attribute: a DMA extent the TOG cannot "
                                  f"compute and no record states: {str(v).splitlines()[0][:120]}")


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


def _spad_buffer_bytes(kernel):
    """The kernel's spad buffer sizes by id (build_skeleton's table), taken off the kernel."""
    attrs = kernel.operation.attributes
    sizes = list(ir.DenseI64ArrayAttr(attrs[ts.ATTR_SPAD_BUFFER_BYTES]))
    del attrs[ts.ATTR_SPAD_BUFFER_BYTES]
    return sizes


def _node_count(kernel):
    """How many compute nodes the kernel has; their tile ids must be 0 .. count-1."""
    ids = sorted(attr_int(op, ts.ATTR_TILE_ID) for op in walk_ops(kernel.regions[0].blocks[0])
                 if op.operation.name == ts.COMPUTE)
    if ids != list(range(len(ids))):
        raise ValueError(f"the computes carry tile ids {ids}, not 0..{len(ids) - 1} once each")
    return len(ids)


def _attr(i):
    return f"ATTR{int(i)}"


def _runtime_attribute_ranks(kernel):
    """The rank of each runtime attribute by id (build_skeleton's table), taken off the kernel."""
    attrs = kernel.operation.attributes
    if ts.ATTR_RUNTIME_ATTRIBUTES not in attrs:
        return []
    ranks = list(ir.DenseI32ArrayAttr(attrs[ts.ATTR_RUNTIME_ATTRIBUTES]))
    del attrs[ts.ATTR_RUNTIME_ATTRIBUTES]
    return ranks


def _emit_globals(module, sizes, n_nodes, args_fields, ranks=()):
    """The TOG's constants before its functions: a togsim::Buf per spad buffer, a togsim::Attribute
    per runtime attribute, a togsim::Node per compute node and the work-item's argument struct;
    after them, the one register_tog."""
    ctx = module.context

    def const(name, cxx, value):
        ir.Operation.create("emitc.global", attributes={
            "sym_name": ir.StringAttr.get(name), "type": ir.TypeAttr.get(_type(ctx, cxx)),
            "initial_value": _opaque(ctx, value), "static_specifier": ir.UnitAttr.get(),
            "const_specifier": ir.UnitAttr.get()})

    with ir.InsertionPoint.at_block_begin(module.body), ir.Location.unknown(ctx):
        for i, b in enumerate(sizes):
            const(_buf(i), "togsim::Buf", f"{{.id = {i}, .bytes = {int(b)}}}")
        for i, r in enumerate(ranks):
            const(_attr(i), "togsim::Attribute", f"{{.id = {i}, .rank = {int(r)}}}")
        for t in range(n_nodes):
            const(_node(t), "togsim::Node", f"{{.id = {t}, .count = {n_nodes}}}")
        if args_fields:
            ir.InsertionPoint.current.insert(ir.Operation.parse(
                "emitc.class @%s {\n%s}" % (ARGS, "".join(f"  emitc.field @{f} : i64\n"
                                                          for f in args_fields)), context=ctx))
    with ir.InsertionPoint(module.body), ir.Location.unknown(ctx):
        attrs = (", {%s}" % ", ".join(_attr(i) for i in range(len(ranks)))) if ranks else ""
        const("tog", "bool", f"togsim::register_tog<{ENTRY}>({_bufs(range(len(sizes)))}{attrs})")


class WorkItem:
    """A kernel whose body is ONE work-item, plus the grid over it; `key_shifts` places each
    axis's program id in the work-item's key (kernel_object.PROGRAM_KEY_SHIFT). `grid_args` are
    the arguments holding each axis's number of programs, `fixed_args` arguments of a known value,
    `scalar_args` the (argument, name, example) of each scalar shape_args carries after the grid.
    `kernel`, `axes` and `example` (the compiled grid) name the shape the TOG reads."""

    extents_handed = ()
    """Indices into `grid` of the extents a work-item is handed after its program ids, in that
    order; lower_to_emitc sets it."""

    scalars_handed = ()
    """Names of the scalar_args a work-item is handed after those extents, in that order;
    lower_to_emitc sets it."""

    def __init__(self, parallel_args, grid, key_shifts=None, grid_args=None, fixed_args=None,
                 kernel="kernel", axes=None, example=None, scalar_args=None):
        if len(parallel_args) != len(grid):
            raise ValueError(
                f"parallel_args {parallel_args} and grid {grid} must have the "
                f"same length -- one program-id argument per grid axis")
        self.parallel_args = list(parallel_args)
        self.grid = [None if g is None else int(g) for g in grid]
        self.key_shifts = list(key_shifts) if key_shifts is not None else [0] * len(grid)
        if len(self.key_shifts) != len(grid):
            raise ValueError(f"key_shifts {key_shifts}: one per grid axis, {len(grid)}")
        self.grid_args = list(grid_args) if grid_args is not None else []
        if len(self.grid_args) not in (0, len(grid)):
            raise ValueError(f"grid_args {grid_args}: one per grid axis, {len(grid)}, or none")
        self.fixed_args = {int(a): int(v) for a, v in (fixed_args or {}).items()}
        self.scalar_args = [(int(a), str(n), int(v)) for a, n, v in (scalar_args or [])]
        self.kernel = kernel
        self.axes = list(axes) if axes is not None else [str(i) for i in range(len(grid))]
        self.example = (list(example) if example is not None
                        else [1 if g is None else int(g) for g in self.grid])
        if len(self.axes) != len(grid) or len(self.example) != len(grid):
            raise ValueError(f"axes {axes} and example {example}: one per grid axis, {len(grid)}")

    @property
    def dynamic_axes(self):
        """Indices into `grid` whose extent arrives at run time."""
        return [i for i, g in enumerate(self.grid) if g is None]


def _materialize_grid_loop(kernel, work_item, ctx):
    """One loop per grid axis around the body: a program-id argument becomes its loop's iv, a
    number-of-programs argument its loop's upper bound, a used scalar_args argument a placeholder.
    Returns the (shape_args index, placeholder) of each dynamic axis's bound and each scalar."""
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
        held = [(len(ubs) + j, block.arguments[a],
                 arith.ConstantOp(block.arguments[a].type,
                                  ir.IntegerAttr.get(block.arguments[a].type, v)).result)
                for j, (a, _, v) in enumerate(work_item.scalar_args)
                if list(block.arguments[a].uses)]
    for _, arg, placeholder in held:
        _replace_all_uses(arg, placeholder)

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

    grid_args = [block.arguments[i] for i in work_item.grid_args]
    fixed = [(block.arguments[i], v) for i, v in work_item.fixed_args.items()]
    with ir.InsertionPoint(inner_block.operations[0]), loc:
        new = [(pid, loop.body.arguments[0]) for loop, pid in zip(loops, pid_args)]
        new += [(arg, ub) for arg, ub in zip(grid_args, ubs) if list(arg.uses)]
        new = [(arg, arith.IndexCastOp(arg.type, v).result if arg.type != idxty else v)
               for arg, v in new]
        new += [(arg, arith.ConstantOp(arg.type, ir.IntegerAttr.get(arg.type, v)).result)
                for arg, v in fixed if list(arg.uses)]

    for arg, value in new:
        _replace_all_uses(arg, value)

    return ([(i, ubs[i]) for i in work_item.dynamic_axes]
            + [(k, placeholder) for k, _, placeholder in held])


def _bind_runtime_bounds(pending, kernel_arg, work_item, ctx):
    """The kernel's shape, togsim::shape<one per grid axis, then one per scalar_args>, read first;
    each `pending` placeholder becomes its value. Returns the scalars' (value, name)."""
    if work_item is None or not (work_item.grid or work_item.scalar_args):
        return []
    from mlir.dialects import arith

    i64 = ir.IntegerType.get_signless(64)
    loc = ir.Location.unknown(ctx)
    n_grid = len(work_item.grid)
    n = n_grid + len(work_item.scalar_args)
    names = work_item.axes + [name for _, name, _ in work_item.scalar_args]
    example = work_item.example + [v for _, _, v in work_item.scalar_args]
    extents = ", .extents = %d" % n_grid if work_item.scalar_args else ""
    shape = ("{.kernel = \"%s\", .names = \"%s\", .example = {%s}%s}"
             % (work_item.kernel, " ".join(names), ", ".join(map(str, example)), extents))
    block = kernel_arg.owner
    with ir.InsertionPoint.at_block_begin(block):
        shape_arg = _call(ctx, "togsim::shape", [kernel_arg], f"std::array<int64_t, {n}>",
                          [_idx(0), _opaque(ctx, shape)],
                          [ir.IntegerAttr.get(ir.IntegerType.get_signless(32), n)])
    scalars = []
    for k, placeholder in pending:
        with ir.InsertionPoint(placeholder.owner), loc:
            kc = ir.Operation.create(
                "emitc.constant", results=[i64],
                attributes={"value": ir.IntegerAttr.get(i64, k)}).results[0]
            ref = ir.Operation.create(
                "emitc.subscript", results=[_lvalue(i64)],
                operands=[shape_arg, kc]).results[0]
            elem = ir.Operation.create(
                "emitc.load", results=[i64], operands=[ref]).results[0]
            got = _from_i64(placeholder.type, elem)
        _replace_all_uses(placeholder, got)
        placeholder.owner.erase()
        if k >= n_grid:
            scalars.append((got, names[k]))
    return scalars


def _from_i64(ty, v):
    """`v`, an i64, as the integer or index type `ty`."""
    from mlir.dialects import arith
    if isinstance(ty, ir.IndexType):
        return arith.IndexCastOp(ty, v).result
    return v if ir.IntegerType(ty).width == 64 else arith.TruncIOp(ty, v).result


def _to_i64(v):
    """The integer or index value `v` as an i64, sign-extended."""
    from mlir.dialects import arith
    i64 = ir.IntegerType.get_signless(64)
    if isinstance(v.type, ir.IndexType):
        return arith.IndexCastOp(i64, v).result
    return v if ir.IntegerType(v.type).width == 64 else arith.ExtSIOp(i64, v).result


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
                else "the TOG would need its value, and shape_args carries "
                     "the grid extents and the integer scalars only")
        raise ValueError(
            f"kernel arg {i} ({arg.type}) still used after build_skeleton, by "
            f"`{str(uses[0].owner).split(chr(10))[0].strip()[:60]}` -- {what}")
    for i in reversed(range(len(block.arguments))):
        block.erase_argument(i)
    kern = _type(ctx, "togsim::Kernel&")
    block.add_argument(kern, ir.Location.unknown(ctx))
    kernel.operation.attributes["function_type"] = ir.TypeAttr.get(ir.FunctionType.get([kern], []))
    kernel.operation.attributes["sym_name"] = ir.StringAttr.get(ENTRY)
    kernel.operation.attributes["sym_visibility"] = ir.StringAttr.get("private")
    return block.arguments[0]


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


def _entry_values(body, entry, chain, scalars=()):
    """(value, field) of each entry value `body` reads, constants aside: a loop bound of `chain` (its
    axis index) in first-read order, then a `scalars` (value, name) (its name) in their order. Any
    other value the work-item cannot be handed, and is refused."""
    found = []

    def walk(block):
        for op in block.operations:
            for v in op.operation.operands:
                owner = v.owner
                if isinstance(owner, ir.Block) or _const_op(v) is not None:
                    continue
                if owner.parent == entry.owner and all(v != f for f in found):
                    found.append(v)
            for rg in op.operation.regions:
                for b in rg.blocks:
                    walk(b)

    walk(body)
    bounds = [loop.operation.operands[1] for loop in chain]
    stray = [v for v in found if all(v != b for b in bounds) and all(v != s for s, _ in scalars)]
    if stray:
        raise ValueError(f"the work-item reads `{str(stray[0].owner).strip()[:60]}` of the entry "
                         f"function: it is handed its program ids, grid extents and scalars only")
    return ([(v, [v == b for b in bounds].index(True)) for v in found if any(v == b for b in bounds)]
            + [(s, name) for s, name in scalars if any(s == v for v in found)])


def _outline_work_item(ctx, kernel, kernel_arg, work_item, keyed=False, scalars=()):
    """Move the innermost grid loop's body into the work-item function, dispatched under the program's
    key with ARGS fields: its ids, the run-time grid extents it reads, the `scalars` it reads. Returns
    (those extents' axes, those scalars' names, the field names, the function, its Item argument)."""
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
    n_ids = len(ivs)
    handed = _entry_values(Lbody, kblk, chain, scalars) if chain else []
    ivs = ivs + [v for v, _ in handed]
    key_shifts = work_item.key_shifts if work_item is not None else ()
    axes = work_item.axes if work_item is not None else []
    fields = ([f"pid_{a}" for a in axes[:n_ids]]
              + [f"arg_{i}" if isinstance(i, str) else f"grid_{axes[i]}" for _, i in handed]
              if len(axes) >= n_ids else [f"iv{k}" for k in range(len(ivs))])

    i64 = ir.IntegerType.get_signless(64)
    item_ty = _type(ctx, "togsim::Item&")
    args_ty = _type(ctx, ARGS)
    ins = [item_ty] + ([args_ty] if ivs else [])
    loc = ir.Location.unknown(ctx)

    tile = ir.Operation.create(
        "func.func", results=[], regions=1,
        attributes={
            "function_type": ir.TypeAttr.get(ir.FunctionType.get(ins, [])),
            "sym_name": ir.StringAttr.get(WORK_ITEM),
            "sym_visibility": ir.StringAttr.get("private")},
        loc=loc, ip=ir.InsertionPoint(kernel))
    with loc:
        tblk = tile.regions[0].blocks.append(*ins)
    with ir.InsertionPoint(tblk):
        tret = ir.Operation.create("func.return", results=[], operands=[], loc=loc)

    idx_vals = []
    with ir.InsertionPoint(tret), loc:
        if ivs:
            copy = ir.Operation.create("emitc.variable", results=[_lvalue(args_ty)],
                    attributes={"value": _opaque(ctx, "")}, loc=loc).results[0]
            ir.Operation.create("emitc.assign", results=[], operands=[copy, tblk.arguments[1]],
                                loc=loc)
        for name, iv in zip(fields, ivs):
            ref = ir.Operation.create("emitc.member", results=[_lvalue(i64)], operands=[copy],
                    attributes={"member": ir.StringAttr.get(name)}, loc=loc).results[0]
            elem = ir.Operation.create("emitc.load", results=[i64],
                    operands=[ref], loc=loc).results[0]
            idx_vals.append(_from_i64(iv.type, elem))

    for op in [o for o in Lbody.operations
               if o.operation.name not in _LOOP_TERMINATORS]:
        op.operation.move_before(tret)

    caps = list(zip(ivs, idx_vals))
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
    fn = [_opaque(ctx, WORK_ITEM)]
    if ivs and len(key_shifts) != n_ids:
        raise ValueError(f"{n_ids} parallel loop(s) and {len(key_shifts)} key field(s): "
                         f"the work-item's key would name another program")
    with ir.InsertionPoint(term), loc:
        if ivs:
            vals = [_to_i64(iv) for iv in ivs]
            args = _call(ctx, ARGS, vals, ARGS)
            key = _work_item_key(ctx, ivs[:n_ids], key_shifts, loc)
            handle = _call(ctx, f"togsim::Keyed<{ARGS}>", [key, args], f"togsim::Keyed<{ARGS}>")
            _call(ctx, "togsim::dispatch", [kernel_arg, handle], template=fn)
        elif keyed:
            i64k = ir.IntegerType.get_signless(64)
            zero = arith_const(ctx, i64k, 0, loc)
            key = _call(ctx, "togsim::Keyed<void>", [zero], "togsim::Keyed<void>")
            _call(ctx, "togsim::dispatch", [kernel_arg, key], template=fn)
        else:
            _call(ctx, "togsim::dispatch", [kernel_arg], template=fn)
    return ([i for _, i in handed if not isinstance(i, str)],
            [i for _, i in handed if isinstance(i, str)], fields if ivs else [], tile, tblk.arguments[0])


def arith_const(ctx, ty, v, loc):
    from mlir.dialects import arith
    return arith.ConstantOp(ty, ir.IntegerAttr.get(ty, v), loc=loc).result


def _elem(op):
    return "togsim::" + ir.StringAttr(op.operation.attributes[ts.ATTR_ELEM]).value


def _rewrite_attribute_reads(ctx, tile, item):
    """Every togsim.attribute_read of the work-item as the togsim.hpp read it is: togsim::scalar of
    a rank-0 attribute, togsim::element at its indices of an array, cast to the type it yields."""
    from mlir.dialects import arith
    i64t = ir.IntegerType.get_signless(64)
    for op in [o.operation for o in walk_ops(tile.regions[0].blocks[0]) if o.operation.name == ts.ATTRIBUTE]:
        rank, i = attr_int(op, "rank"), attr_int(op, "id")
        with ir.InsertionPoint(op), op.location:
            at = [arith.IndexCastOp(i64t, v).result for v in op.operands]
            got = ir.Operation.create(
                "emitc.call_opaque", results=[i64t], operands=[item] + at,
                attributes={"callee": ir.StringAttr.get("togsim::element" if rank else "togsim::scalar"),
                            "args": ir.ArrayAttr.get([_idx(0), _opaque(ctx, _attr(i))]
                                                     + [_idx(k + 1) for k in range(len(at))])}).results[0]
            ty = op.results[0].type
            if isinstance(ty, ir.IndexType):
                got = arith.IndexCastOp(ty, got).result
            elif ir.IntegerType(ty).width < 64:
                got = arith.TruncIOp(ty, got).result
        op.results[0].replace_all_uses_with(got)
        op.erase()


def _desc(ctx, op):
    """togsim::Desc of a skeleton DMA: its argument, element type, offset, dims (its dyn_dims sized
    at run time) and strides, and a gather's togsim::Gather of its index snapshot."""
    i64t = ir.IntegerType.get_signless(64)
    dram, _sem, *extents = op.operation.operands
    n_dyn = len(attr_i64_array(op, ts.ATTR_DYN_DIMS, []))
    dims = attr_i64_array(op, ts.ATTR_DIMS)
    dyn = dict(zip(attr_i64_array(op, ts.ATTR_DYN_DIMS, []), extents))
    dims_v = _dims(ctx, [ir.Operation.create("arith.index_cast", results=[i64t],
                                             operands=[dyn[k]]).results[0] if k in dyn else int(d)
                         for k, d in enumerate(dims)])
    strides_v = _dims(ctx, [int(s) for s in attr_i64_array(op, ts.ATTR_STRIDES)])
    args = [ir.IntegerAttr.get(ir.IntegerType.get_signless(32), attr_int(op, ts.ATTR_ARG_ID)),
            _opaque(ctx, _elem(op)), _idx(0), _idx(1), _idx(2)]
    operands = [dram, dims_v, strides_v]
    gather = attr_i64_array(op, ts.ATTR_GATHER)
    if gather:
        gid, n_at, axis, scale = gather
        tail = extents[n_dyn:]
        at = _dims(ctx, [ir.Operation.create("arith.index_cast", results=[i64t], operands=[v]).results[0]
                         for v in tail[:n_at]])
        g_args = [_opaque(ctx, _attr(gid)), _idx(0), ir.IntegerAttr.get(ir.IntegerType.get_signless(32), axis)]
        g_ops = [at]
        if scale == -1:
            g_args.append(_idx(1))
            g_ops.append(tail[n_at] if tail[n_at].type == i64t else
                         ir.Operation.create("arith.extsi", results=[i64t], operands=[tail[n_at]]).results[0])
        else:
            g_args.append(ir.IntegerAttr.get(i64t, scale))
        operands.append(_call(ctx, "togsim::Gather", g_ops, "togsim::Gather", g_args))
        args.append(_idx(3))
    return _call(ctx, "togsim::Desc", operands, "togsim::Desc", args)


def _sem_slot(op):
    sem = op.operation.operands[0 if op.operation.name == ts.MEMORY_BAR else 1]
    c = _const_op(sem)
    if c is None:
        raise ValueError(f"an async load or a barrier on a semaphore that is not a constant: "
                         f"{str(op).splitlines()[0][:80]}")
    return ir.IntegerAttr(c.attributes["value"]).value


def _rewrite_togsim_ops(ctx, tile, item):
    """Every skeleton op of the work-item as the togsim.hpp call it is: a load, a store, a wait or
    a compute. One togsim::Semaphore per semaphore slot, made at entry: an async load signals it,
    a barrier waits it, and it stays usable after the wait."""
    block = tile.regions[0].blocks[0]
    loc = ir.Location.unknown(ctx)
    ops = list(walk_ops(block))
    slots = []
    for op in ops:
        o = op.operation
        if (o.name == ts.DMA and attr_int(op, ts.ATTR_DIR) == ts.DIR_LOAD
                and _attr_bool(op, ts.ATTR_IS_ASYNC) and _sem_slot(op) not in slots):
            slots.append(_sem_slot(op))
    with ir.InsertionPoint(block.operations[0]), loc:
        sems = {s: _call(ctx, "togsim::semaphore", [item], "togsim::Semaphore") for s in slots}

    def sem_of(op):
        return sems[_sem_slot(op)]

    for op in ops:
        name = op.operation.name
        if name not in ts.OP_NAMES:
            continue
        with ir.InsertionPoint(op), loc:
            reads = attr_i64_array(op, ts.ATTR_READ_BUFS, [])
            writes = attr_i64_array(op, ts.ATTR_WRITE_BUFS, [])
            if name == ts.DMA and attr_int(op, ts.ATTR_DIR) == ts.DIR_LOAD:
                if len(writes) != 1:
                    raise ValueError(f"a load writes buffers {writes}, not one")
                desc = _desc(ctx, op)
                args = [_idx(0), _opaque(ctx, _buf(writes[0]))]
                operands = [desc]
                if _attr_bool(op, ts.ATTR_IS_ASYNC):
                    args.append(_idx(1))
                    operands.append(sem_of(op))
                elif reads:
                    args.append(_opaque(ctx, "{}"))
                if reads:
                    args.append(_opaque(ctx, _bufs(reads)))
                load = _call(ctx, "togsim::Load", operands, "togsim::Load", args)
                _call(ctx, "togsim::load", [item, load])
            elif name == ts.DMA:
                if not reads or writes:
                    raise ValueError(f"a store reads buffers {reads} and writes {writes}")
                desc = _desc(ctx, op)
                args = [_opaque(ctx, _buf(reads[0])), _idx(0)]
                if reads[1:]:
                    args.append(_opaque(ctx, _bufs(reads[1:])))
                store = _call(ctx, "togsim::Store", [desc], "togsim::Store", args)
                _call(ctx, "togsim::store", [item, store])
            elif name == ts.MEMORY_BAR:
                if _sem_slot(op) not in sems:
                    raise ValueError(f"a barrier on semaphore {_sem_slot(op)}, which no async "
                                     f"load signals")
                wait = _call(ctx, "togsim::Wait", [sem_of(op)], "togsim::Wait")
                _call(ctx, "togsim::wait", [item, wait])
            else:
                if attr_int(op, ts.ATTR_COMPUTE_TYPE):
                    raise ValueError("a typed compute: togsim.hpp computes are untyped")
                t = attr_int(op, ts.ATTR_TILE_ID)
                _call(ctx, "togsim::compute", [item], args=[_idx(0), _opaque(
                    ctx, f"{{.node = {_node(t)}, .variant = {VARIANT}, "
                         f".reads = {_bufs(reads)}, .writes = {_bufs(writes)}}}")])
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
    """Lower a skeleton+API module (in place) to an EmitC module of a togsim.hpp TOG: the entry
    ENTRY, the work-item WORK_ITEM, their constants and register_tog; and record on `work_item`
    which grid extents a work-item is handed."""
    ctx = skeleton_module.context
    kernel = _find_kernel(skeleton_module)
    if kernel is None:
        raise ValueError("no kernel function found in skeleton module")

    _strip_aux(skeleton_module, keep=kernel)
    sizes = _spad_buffer_bytes(kernel)
    ranks = _runtime_attribute_ranks(kernel)
    n_nodes = _node_count(kernel)
    pending = []
    if work_item is not None:
        pending = _materialize_grid_loop(kernel, work_item, ctx)
    _refuse_uncomputable_extents(kernel)
    kernel_arg = _rewrite_signature(kernel, ctx)
    scalars = _bind_runtime_bounds(pending, kernel_arg, work_item, ctx)
    handed, handed_scalars, fields, tile, item = _outline_work_item(
        ctx, kernel, kernel_arg, work_item, keyed=bool(ranks), scalars=scalars)
    _rewrite_attribute_reads(ctx, tile, item)
    _rewrite_togsim_ops(ctx, tile, item)
    if work_item is not None:
        work_item.extents_handed = tuple(handed)
        work_item.scalars_handed = tuple(handed_scalars)
    _expand_int_minmax(skeleton_module)

    PassManager.parse(_PIPELINE, ctx).run(skeleton_module.operation)

    _retype_for_to_size_t(skeleton_module)
    _emit_globals(skeleton_module, sizes, n_nodes, fields, ranks)
    return skeleton_module


def tog_banner():
    """The generated file's first line: the TOGSim version togsim_runtime.h states, refused unless it
    takes what this compiler writes (kernel_object.TOGSIM_VERSION): the same major, no older."""
    from .... import toolchain
    from ....contract.kernel_object import TOGSIM_VERSION, TOGSIM_VERSION_NUMBER
    ver = toolchain.togsim_version()
    if ver is None or ver[1] >> 16 != TOGSIM_VERSION[0] or ver[1] < TOGSIM_VERSION_NUMBER:
        raise RuntimeError(
            f"{os.path.join(toolchain.TOGSIM_INCLUDE, 'togsim_runtime.h')} states TOGSim "
            f"{'no version' if ver is None else 'v' + ver[0]}, and this compiler writes for "
            f"v{'.'.join(map(str, TOGSIM_VERSION))}: they are not a pair. Pin third_party/TOGSim to "
            f"the commit this checkout names, or set TOGSIM_DIR.")
    return ("// GENERATED by pytorchsim-triton-compiler -- tile-operation-graph (TOG) for TOGSim "
            "v%s (togsim.hpp). DO NOT EDIT.\n" % ver[0])


def emitc_to_cpp(emitc_module):
    """Render `emitc_module` to C++: banner, includes, and the mlir-to-cpp body."""
    from .... import toolchain
    proc = subprocess.run(
        [toolchain.MLIR_TRANSLATE, "--mlir-to-cpp"],
        input=str(emitc_module), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("mlir-translate --mlir-to-cpp failed:\n" + proc.stderr)
    return tog_banner() + _INCLUDES + proc.stdout


def compile_so(cpp_path, so_path):
    """Compile the TOG's C++ to `so_path`."""
    from .... import toolchain
    cmd = [toolchain.HOST_CXX, "-shared", "-fPIC", "-std=gnu++20", "-O2",
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
