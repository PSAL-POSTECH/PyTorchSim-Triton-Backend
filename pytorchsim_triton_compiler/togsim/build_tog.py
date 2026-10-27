"""Python port of the C++ `-test-tile-operation-graph` analysis pass."""

import mlir.ir as ir

from ..dialect.pytorchsim import (
    TransferOp, WaitOp, branch_controls, is_load, is_loop, loop_like, terminator_controls)
from ..passes.transforms.mark_runtime_attributes import PLUMBING, RECORD, erase_records, rests_on_data
from ..passes.utils.ir_utils import bounds_lives_only, is_view, loop_bounds
from ..contract.gem5_samples import SAMPLE_CLOSE, SAMPLE_OPEN
from ..contract.kernel_object import CYCLE_CALL_GLOBAL
TRANSFER, WAIT = TransferOp.OPERATION_NAME, WaitOp.OPERATION_NAME


SEGMENT_COMPUTE = 0
"""A segment holds the core for its cycles; TOGSim's vector pipeline is the one queue per core."""

COMPUTE_TYPE_NAME = {SEGMENT_COMPUTE: "VectorCompute"}
"""Every compute type the TOG emits, by the name its marker carries."""

CORE = "__CORE__"
"""The buffer every segment reads and writes: one core runs its segments in program order."""

TILE_ID = "pytorchsim.tile_id"
"""The tile a post-vcix op's segment is: stamped once by the skeleton (stamp_tile_ids), read back by
the TOG and the cycle binary (tile_ids), so the two walks name each segment the same tile."""


class TOGNode:
    """A node of the walk: the op it stands for and the nodes under it, in program order."""

    def __init__(self):
        self.children = []
        self.op = None

    def get_last_child(self):
        return self.children[-1] if self.children else None

    def add_child(self, c):
        self.children.append(c)


class TOGLoopNode(TOGNode):

    def __init__(self, start, end, step):
        super().__init__()
        self.loop_start = start
        self.loop_end = end
        self.loop_step = step


class TOGDMANode(TOGNode):

    def __init__(self, addr, tile_size, tile_stride, elem_size, is_write, is_async, indirect_mode):
        super().__init__()
        self.base_addr = addr
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.element_size = elem_size
        self.is_write = is_write
        self.is_async = is_async
        self.indirect_mode = indirect_mode


class TOGDMAWaitNode(TOGNode):
    pass


class TOGComputeNode(TOGNode):

    def __init__(self, ctype):
        super().__init__()
        self.compute_type = ctype
        self.operations = []


def _op_name(value_or_op):
    return value_or_op.operation.name


def _is_block_arg(value):
    return isinstance(value, ir.BlockArgument)


def _memref_space(memref_type):
    mt = ir.MemRefType(memref_type)
    sp = mt.memory_space
    if sp is None:
        return 0
    return ir.IntegerAttr(sp).value


def _descriptor_walk(op):
    """(dims, DRAM strides, lead) a transfer's descriptor walks, as the subtarget's target states it: what
    TOGSim times and the model moves, a uniform load's lane copies too."""
    from ..driver import machine
    from ..targets import target
    return target().walk(op, machine.VECTORLANE_SIZE)


_LOOP_OPS = ("affine.for", "scf.for")

SKIP_OPS = {
    "affine.yield", "scf.yield", "scf.condition", "affine.apply", "memref.get_global",
    "arith.constant", "llvm.mlir.constant",
}


class Segments:
    """Compute grouped by boundaries alone: a segment is what one core runs between two ops that cut
    (cuts). Every segment is SEGMENT_COMPUTE and reads and writes CORE."""

    def __init__(self, builder):
        self.builder = builder

    def replays(self, loop):
        """A loop the TOG holds and the sample runs once: one that cuts."""
        return cuts(loop, self.builder.func)

    def append(self, op, node):
        """`op` joins the segment its parent left open in its own block, or opens one: what closes a
        segment is an op that cuts becoming the parent's last child."""
        if cuts(op, self.builder.func):
            raise RuntimeError(f"{_op_name(op)} cuts a segment, yet no TOG node takes it")
        last = node.get_last_child()
        if isinstance(last, TOGComputeNode) and last.operations[-1].operation.block == op.operation.block:
            last.operations.append(op)
            return
        cn = TOGComputeNode(SEGMENT_COMPUTE)
        cn.operations.append(op)
        cn.op = op
        node.add_child(cn)
        self.builder.compute_nodes.append(cn)


class TogBuilder:
    def __init__(self):
        self.compute_nodes = []
        self.loop_nodes = []
        self.branch_nodes = []
        self.dma_nodes = []
        self.func = None
        self.segments = Segments(self)

    def visit_operation(self, op, node):
        """Walk `op` and attach the nodes it produces under `node`."""
        name = _op_name(op)
        if (name in SKIP_OPS or is_view(op) or bounds_lives_only(op.operation)
                or name == RECORD or PLUMBING in op.operation.attributes):
            return

        if name in _LOOP_OPS:
            oper = op.operation
            if not self.segments.replays(op):
                if node is not None:
                    self.segments.append(op, node)
                return
            start, end, step = loop_bounds(op)
            if None not in (start, end) and end <= start:
                raise NotImplementedError(
                    f"a {name} from {start} to {end} makes no trip yet the TOG replays it: "
                    f"its sample would run one trip the program never makes")
            loop_node = TOGLoopNode(start, end, step)
            loop_node.op = op
            self.loop_nodes.append(loop_node)
            if node is not None:
                node.add_child(loop_node)
            for region in oper.regions:
                for block in region.blocks:
                    for inner in block.operations:
                        self.visit_operation(inner, loop_node)
            return

        if name == TRANSFER:
            self._handle_dma_start(op, node)
            return

        if is_loop(op) and self.segments.replays(op):
            self._visit_loop(op, node)
            return

        if branch_controls(op) is not None:
            self._visit_branch(op, node)
            return

        if name == WAIT:
            self._handle_dma_wait(op, node)
            return

        if node is None:
            return

        self.segments.append(op, node)

    def _visit_branch(self, op, node):
        """A branch as the TOG's, whatever its regions hold: every region of it, each as it runs."""
        bn = TOGNode()
        bn.op = op
        self.branch_nodes.append(bn)
        if node is not None:
            node.add_child(bn)
        for region in op.operation.regions:
            for block in region.blocks:
                for inner in block.operations:
                    self.visit_operation(inner, bn)

    def _visit_loop(self, op, node):
        """A loop no iv counts (an scf.while) that cuts: the TOG runs it as the program does, to the
        condition it computes; every region of it is the loop's body."""
        loop_node = TOGLoopNode(None, None, None)
        loop_node.op = op
        self.loop_nodes.append(loop_node)
        if node is not None:
            node.add_child(loop_node)
        for region in op.operation.regions:
            for block in region.blocks:
                for inner in block.operations:
                    self.visit_operation(inner, loop_node)

    def _dma_start_fields(self, op):
        tx = op.operation.opview
        if is_load(op):
            src, dst = tx.dram, tx.sram
        else:
            src, dst = tx.sram, tx.dram
        return {
            "src": src, "src_type": src.type,
            "dst": dst, "dst_type": dst.type,
            "offset": tx.index,
        }

    def _handle_dma_start(self, op, node):
        oper = op.operation
        f = self._dma_start_fields(op)
        dma_async = bool(self._get_async(oper))

        dst_space = _memref_space(f["dst_type"])
        src_space = _memref_space(f["src_type"])

        if dst_space == 0 and src_space == 1:
            is_write = True
            tile_type = f["src_type"]
            dram_memref = f["dst"]
        elif dst_space == 1 and src_space == 0:
            is_write = False
            tile_type = f["dst_type"]
            dram_memref = f["src"]
        else:
            raise RuntimeError("Unexpected memory space")

        tile_mt = ir.MemRefType(tile_type)
        tile_size, tile_stride, lead = _descriptor_walk(oper)

        address = "arg"
        if "dram_arg" in oper.attributes:
            address += str(ir.IntegerAttr(oper.attributes["dram_arg"]).value)
        elif _is_block_arg(dram_memref):
            address += str(ir.BlockArgument(dram_memref).arg_number)

        et = tile_mt.element_type
        if isinstance(et, ir.IntegerType):
            element_size = ir.IntegerType(et).width
        elif isinstance(et, ir.FloatType):
            element_size = ir.FloatType(et).width
        else:
            raise RuntimeError("Unsupported element type")

        dma_node = TOGDMANode(address, tile_size, tile_stride, element_size, is_write, dma_async,
                              f["offset"] is not None)
        dma_node.op = op
        dma_node.lead = lead
        dma_node.element_type = str(et)
        self.dma_nodes.append(dma_node)
        node.add_child(dma_node)

    def _handle_dma_wait(self, op, node):
        wait_node = TOGDMAWaitNode()
        wait_node.op = op
        self.dma_nodes.append(wait_node)
        node.add_child(wait_node)

    @staticmethod
    def _get_async(oper):
        if "async" not in oper.attributes:
            return 0
        attr = oper.attributes["async"]
        try:
            return ir.IntegerAttr(attr).value
        except Exception:
            pass
        try:
            return 1 if ir.BoolAttr(attr).value else 0
        except Exception:
            return 1


def _is_empty_affine_for(op):
    """An affine.for that does nothing: its body only yields and nothing reads a result of it."""
    if op.operation.name != "affine.for" or len(op.operation.results):
        return False
    body = op.operation.regions[0].blocks[0]
    ops = list(body.operations)
    if len(ops) != 1:
        return False
    return ops[0].operation.name == "affine.yield"


def _erase_empty_affine_for(func_op):
    changed = True
    while changed:
        changed = False
        to_erase = []

        def walk(block):
            for op in block.operations:
                for region in op.operation.regions:
                    for b in region.blocks:
                        walk(b)
                if _is_empty_affine_for(op):
                    to_erase.append(op)

        walk(func_op.regions[0].blocks[0])
        for op in to_erase:
            op.operation.erase()
            changed = True


def _make_inline_asm(ctx, ip, asm_string, compute_type=None, tile_id=None):
    """A marker the backend keeps in place: it clobbers memory and the VCIX state register, which
    LLVM's VCIX instructions read and write; a start marker hands gem5 its tile id in a0, call in a1."""
    i64 = ir.IntegerType.get_signless(64)
    regs = "~{a0}"
    if tile_id is not None:
        asm_string = (f"li a0, {tile_id}\nla a1, {CYCLE_CALL_GLOBAL}\nld a1, 0(a1)\n"
                      + asm_string)
        regs = "~{a0},~{a1}"
    attrs = {
        "has_side_effects": ir.UnitAttr.get(),
        "asm_dialect": ir.IntegerAttr.get(i64, 0),
        "asm_string": ir.StringAttr.get(asm_string),
        "constraints": ir.StringAttr.get(regs + ",~{memory},~{sf.vcix_state}"),
    }
    if compute_type is not None:
        attrs["compute_type"] = ir.StringAttr.get(compute_type)
    if tile_id is not None:
        attrs[TILE_ID] = ir.IntegerAttr.get(i64, tile_id)
    return ir.Operation.create(
        "llvm.inline_asm", results=[], operands=[], attributes=attrs,
        loc=ir.Location.unknown(ctx), ip=ip)


def _rewrite_loop_steps(builder):
    """Every replayed loop makes one trip from its own start: an scf.for ends at lb + step (a step no
    constant states, which the sample may not hold, becomes 1), an affine.for steps over its whole
    range, a loop no iv counts is its one trip inlined."""
    from mlir.dialects import arith
    index_t = ir.IndexType.get()
    for node in builder.loop_nodes:
        forop = node.op.operation
        if forop.name == "scf.for":
            ty = forop.operands[0].type
            with ir.InsertionPoint(forop), ir.Location.unknown():
                if node.loop_step is None:
                    forop.operands[2] = arith.ConstantOp(ty, ir.IntegerAttr.get(ty, 1)).result
                forop.operands[1] = (arith.ConstantOp(ty, ir.IntegerAttr.get(
                    ty, node.loop_start + (node.loop_step or 1))).result if node.loop_start is not None
                    else arith.AddIOp(forop.operands[0], forop.operands[2]).result)
            continue
        if forop.name != "affine.for":
            _one_trip(forop)
            continue
        if None in (node.loop_start, node.loop_end):
            node.op = _affine_one_trip(forop, node.loop_step)
            continue
        once = max(node.loop_end - node.loop_start, node.loop_step)
        forop.attributes["step"] = ir.IntegerAttr.get(index_t, once)


def _affine_one_trip(forop, step):
    """`forop`, an affine.for whose bounds are not constants, rebuilt to end at its lower bound + step:
    one trip from its own start. Its lower bound is one result, or no single trip is its first."""
    lb = ir.AffineMapAttr(forop.attributes["lowerBoundMap"]).value
    if len(lb.results) != 1:
        raise NotImplementedError(f"an affine.for whose lower bound is the max of {len(lb.results)} "
                                  f"results holds a DMA or a wait: its sample has no one first trip")
    seg = list(ir.DenseI32ArrayAttr(forop.attributes["operandSegmentSizes"]))
    ops = list(forop.operands)
    lb_ops, inits = ops[:seg[0]], ops[seg[0] + seg[1]:]
    ub = ir.AffineMap.get(lb.n_dims, lb.n_symbols, [lb.results[0] + ir.AffineConstantExpr.get(step)])
    attrs = {forop.attributes[i].name: forop.attributes[i].attr for i in range(len(forop.attributes))}
    attrs["upperBoundMap"] = ir.AffineMapAttr.get(ub)
    attrs["operandSegmentSizes"] = ir.DenseI32ArrayAttr.get([seg[0], seg[0], seg[2]])
    old = forop.regions[0].blocks[0]
    new = ir.Operation.create("affine.for", results=[r.type for r in forop.results],
                              operands=lb_ops + lb_ops + inits, attributes=attrs, regions=1,
                              loc=forop.location, ip=ir.InsertionPoint(forop))
    blk = new.regions[0].blocks.append(*[a.type for a in old.arguments])
    for a, b in zip(old.arguments, blk.arguments):
        a.replace_all_uses_with(b)
    for o in list(old.operations):
        blk.append(o.operation)
    for r, q in zip(forop.results, new.results):
        r.replace_all_uses_with(q)
    forop.erase()
    return new


def _one_trip(loop):
    """`loop`, a while (a region that decides, then the body), run as one trip in place: the first
    region's ops on the inits, then the body's on what the decision forwards, which are its results."""
    from ..dialect.pytorchsim import terminator_controls
    if len(loop.regions) != 2:
        raise NotImplementedError(f"a {loop.name} holding a DMA or a wait with {len(loop.regions)} "
                                  f"regions: its sample has no one trip to run")
    values = list(loop.operands)
    for region in loop.regions:
        block = region.blocks[0]
        for a, v in zip(block.arguments, values):
            a.replace_all_uses_with(v)
        ops = list(block.operations)
        term = ops[-1].operation
        controls = terminator_controls(term) or []
        values = [v for v in term.operands if not any(v == c for c in controls)]
        for o in ops[:-1]:
            o.operation.move_before(loop)
        if region == loop.regions[0]:
            for r, v in zip(loop.results, values):
                r.replace_all_uses_with(v)
        term.erase()
    loop.erase()


def _take_branches(builder):
    """Every branch the TOG holds runs each of its regions once in the sample, so gem5 times what the
    TOG may run: an scf.if's then on the op itself, its else on a copy right after it."""
    i1 = ir.IntegerType.get_signless(1)
    for node in reversed(builder.branch_nodes):
        op = node.op.operation
        work = [any(len(list(b.operations)) > 1 for b in r.blocks) for r in op.regions]
        controls = branch_controls(op)
        if work[1:] != [False] * (len(work) - 1) and (
                len(work) != 2 or len(controls) != 1 or controls[0].type != i1):
            raise NotImplementedError("a branch holding a DMA or a wait with work past its first region "
                                      "that is no two-region if: its sample cannot run each region")
        if all(work):
            copy = op.clone(ir.InsertionPoint.at_block_begin(op.block))
            copy.move_after(op)
            _set_controls(copy, branch_controls(copy), 0)
        _set_controls(op, controls, 1 if work[0] else 0)


def _set_controls(op, controls, value):
    from mlir.dialects import arith
    i1 = ir.IntegerType.get_signless(1)
    with ir.InsertionPoint(op), ir.Location.unknown():
        c = arith.ConstantOp(i1, ir.IntegerAttr.get(i1, value)).result
    for v in controls:
        for k, o in enumerate(op.operands):
            if o == v:
                op.operands[k] = c


def stamp_tile_ids(builder):
    """Each segment's tile id, its place in program order, onto every op it holds: the one place a
    tile id is decided."""
    i64 = ir.IntegerType.get_signless(64)
    for tile_id, cn in enumerate(builder.compute_nodes):
        if not cn.operations:
            raise RuntimeError(f"segment {tile_id} holds no op, so no tile id can ride on it")
        for op in cn.operations:
            op.operation.attributes[TILE_ID] = ir.IntegerAttr.get(i64, tile_id)


def tile_ids(builder, func_op):
    """The tile id stamped on each segment, in program order; refused unless every op of a segment
    carries the same one, they run 0, 1, ... and no op outside a segment carries one."""
    ids, held = [], set()
    for cn in builder.compute_nodes:
        got = {ir.IntegerAttr(o.operation.attributes[TILE_ID]).value
               if TILE_ID in o.operation.attributes else None for o in cn.operations}
        if len(got) != 1 or None in got:
            raise RuntimeError(f"a segment's ops carry tile ids {sorted(got, key=str)}, not one "
                               f"stamped id: the TOG and this walk cut different segments")
        ids.append(got.pop())
        held.update(o.operation for o in cn.operations)
    if ids != list(range(len(ids))):
        raise RuntimeError(f"segments carry tile ids {ids} in program order, not 0..{len(ids) - 1}")
    stray = [o for o in _all_ops(func_op) if TILE_ID in o.operation.attributes
             and o.operation not in held]
    if stray:
        raise RuntimeError(f"{len(stray)} op(s) carry a tile id outside every segment this walk "
                           f"cut: {stray[0].operation.name}")
    return ids


def _all_ops(op):
    for region in op.operation.regions:
        for block in region.blocks:
            for inner in block.operations:
                yield inner
                yield from _all_ops(inner)


def _check_segments_are_contiguous(builder):
    """Each segment's markers bracket its own ops alone: they are siblings in program order, and
    nothing between the first and the last is or holds a DMA or a wait."""
    in_block = {}
    for cn in builder.compute_nodes:
        ops = [o.operation for o in cn.operations]
        block = ops[0].block
        if block not in in_block:
            listed = [o.operation for o in block.operations]
            in_block[block] = (listed, {o: i for i, o in enumerate(listed)})
        siblings, place = in_block[block]
        at = [place.get(o, -1) for o in ops]
        if -1 in at or at != sorted(at):
            raise RuntimeError(f"a segment's ops are not siblings in program order: {ops[0].name}")
        inside = siblings[at[0]:at[-1] + 1]
        if any(cuts(o, builder.func) for o in inside):
            raise RuntimeError(f"a segment's markers would bracket an op that cuts: {ops[0].name}")


def _insert_compute_markers(builder, ids):
    for node, tile_id in zip(builder.compute_nodes, ids, strict=True):
        ops = node.operations
        front = ops[0]
        back = ops[-1]
        ctx = front.operation.context
        ctype_name = COMPUTE_TYPE_NAME[node.compute_type]
        close = _make_inline_asm(ctx, ir.InsertionPoint.at_block_begin(back.operation.block), SAMPLE_CLOSE)
        close.move_after(back.operation)
        _make_inline_asm(ctx, ir.InsertionPoint(front), SAMPLE_OPEN, ctype_name, tile_id)


def _find_kernel(module):
    funcs = [op for op in module.body.operations
             if op.operation.name == "func.func"]
    for op in funcs:
        if ir.StringAttr(op.operation.attributes["sym_name"]).value == "kernel":
            return op
    return funcs[0] if len(funcs) == 1 else None


def cuts(op, func):
    """Whether `op` of `func` ends the segment before it: a DMA, a wait, a branch, a loop whose control
    rests on data (rests_on_data), or an op whose regions hold one. The one test of a segment's end."""
    return _is_boundary(op, func) or _holds(op, lambda o: _is_boundary(o, func))


def _is_boundary(op, func):
    o = op.operation
    if o.name in (TRANSFER, WAIT) or branch_controls(o) is not None:
        return True
    return is_loop(o) and any(rests_on_data(v, func) for v in _controls(o))


def _controls(loop):
    """The values that decide how often `loop` runs: its bounds and steps, or its regions' terminators'
    controls (a while's condition)."""
    got = loop_like(loop)
    if got is not None:
        return [b for b in got.lbs + got.ubs + got.steps if not isinstance(b, int)]
    return [c for r in loop.regions for b in r.blocks
            for c in terminator_controls(list(b.operations)[-1]) or ()]


def _holds(op, hit):
    found = [False]

    def walk(o):
        if found[0]:
            return
        if hit(o):
            found[0] = True
            return
        for region in o.operation.regions:
            for block in region.blocks:
                for inner in block.operations:
                    walk(inner)

    for region in op.operation.regions:
        for block in region.blocks:
            for inner in block.operations:
                walk(inner)
    return found[0]


def _is_address_plumbing(op):
    name = _op_name(op)
    if name in ("func.return", "memref.cast"):
        return True
    if not name.startswith("arith."):
        return False
    results = list(op.operation.results)
    if not results:
        return False
    return all(isinstance(r.type, ir.IndexType) or isinstance(r.type, ir.IntegerType)
               for r in results)


def _build(module, builder):
    """Walk `@kernel` into `builder`: its loops, DMAs, waits and the segments between them."""
    func_op = _find_kernel(module)
    if func_op is None:
        return
    builder.func = func_op.operation

    _erase_empty_affine_for(func_op)

    block = func_op.regions[0].blocks[0]
    root = TOGNode()
    for op in block.operations:
        if _is_address_plumbing(op):
            continue
        builder.visit_operation(op, root)


def build_tog_and_mutate(module):
    """Cut the post-vcix IR to one sample tile in place: every replayed loop makes one trip, every
    branch runs each region, and the runtime attribute records the timed binary does not make are
    erased first; each segment is bracketed by markers carrying its stamped tile id."""
    erase_records(module)
    builder = TogBuilder()
    _build(module, builder)
    ids = tile_ids(builder, _find_kernel(module))
    _rewrite_loop_steps(builder)
    _check_segments_are_contiguous(builder)
    _insert_compute_markers(builder, ids)
    _take_branches(builder)


def run_tog(in_path, out_path):
    """Read the post-vcix IR at `in_path` and write its sample-tile cut to `out_path`."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(open(in_path).read(), ctx)
        build_tog_and_mutate(module)
        with open(out_path, "w") as fh:
            fh.write(str(module))
