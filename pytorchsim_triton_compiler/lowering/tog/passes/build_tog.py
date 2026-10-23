"""Python port of the C++ `-test-tile-operation-graph` analysis pass."""

import mlir.ir as ir

from ....dialects.torchsim import TransferOp, WaitOp, is_load
from ...passes.lib_ir import VIEW_OPS, bounds_lives_only, loop_bounds
from ....contract.kernel_object import CYCLE_CALL_GLOBAL
from ....contract.stop import Reason, Refusal
TRANSFER, WAIT = TransferOp.OPERATION_NAME, WaitOp.OPERATION_NAME


SEGMENT_COMPUTE = 0
"""A segment holds the core for its cycles; TOGSim's vector pipeline is the one queue per core."""

COMPUTE_TYPE_NAME = {SEGMENT_COMPUTE: "VectorCompute"}
"""Every compute type the TOG emits, by the name its marker carries."""

CORE = "__CORE__"
"""The buffer every segment reads and writes: one core runs its segments in program order."""

TILE_ID = "torchsim.tile_id"
"""The tile a post-vcix op's segment is: stamped once by the skeleton (stamp_tile_ids), read back by
the TOG and the cycle binary (tile_ids), so the two walks name each segment the same tile."""


class UncountedLoopInTog(NotImplementedError, Refusal):
    """A DMA or a wait in a loop whose trip count is not a constant: a TOG replays counted loops."""
    reason = Reason.TOG_UNCOUNTED_LOOP


class BranchInTog(NotImplementedError, Refusal):
    """A DMA or a wait under a branch: a TOG has no branches."""
    reason = Reason.TOG_BRANCH


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
    """(dims, DRAM strides, lead) a transfer's descriptor walks, as the target's lower-dma provider
    states it: what TOGSim times and what Spike's index dump counts, a uniform load's lane copies too."""
    from .... import machine
    from ...passes.lower import dma_lowering
    return dma_lowering(machine.target()).walk(op, machine.VECTORLANE_SIZE)


_LOOP_OPS = ("affine.for", "scf.for")

SKIP_OPS = {
    "affine.yield", "scf.yield", "affine.apply", "memref.get_global",
    "arith.constant", "llvm.mlir.constant", *VIEW_OPS,
}


class Segments:
    """Compute grouped by boundaries alone: a segment is what one core runs between two of a DMA,
    a wait and a replayed loop. Every segment is SEGMENT_COMPUTE and reads and writes CORE."""

    def __init__(self, builder):
        self.builder = builder

    def replays(self, loop):
        """A loop the TOG holds and the sample runs once: one holding a DMA or a wait."""
        return _holds_boundary(loop)

    def refuse_unreplayed(self, loop):
        if _holds_boundary(loop):
            raise UncountedLoopInTog(
                f"a {_op_name(loop)} whose trip count is not a constant holds a DMA or a wait: "
                f"a TOG holds counted loops only")

    def append(self, op, node):
        """`op` joins the segment its parent left open, or opens one: what closes a segment is a
        DMA, a wait or a replayed loop becoming the parent's last child."""
        if _holds_boundary(op):
            raise BranchInTog(
                f"a DMA or a wait inside {_op_name(op)}: a TOG has no branches, so it "
                f"cannot say whether that transfer happens")
        last = node.get_last_child()
        if isinstance(last, TOGComputeNode):
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
        self.dma_nodes = []
        self.segments = Segments(self)

    def visit_operation(self, op, node):
        """Walk `op` and attach the nodes it produces under `node`."""
        name = _op_name(op)
        if name in SKIP_OPS or bounds_lives_only(op.operation):
            return

        if name in _LOOP_OPS:
            oper = op.operation
            bounds = loop_bounds(op)
            if bounds is not None and None in bounds:
                bounds = None
            if bounds is None or not self.segments.replays(op):
                self.segments.refuse_unreplayed(op)
                if node is not None:
                    self.segments.append(op, node)
                return

            start, end, step = bounds
            if end <= start:
                raise NotImplementedError(
                    f"a {name} from {start} to {end} makes no trip yet holds a DMA or a wait: "
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

        if name == WAIT:
            self._handle_dma_wait(op, node)
            return

        if node is None:
            return

        self.segments.append(op, node)

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


def _containing_block(op):
    parent = op.operation.parent
    for region in parent.regions:
        for block in region.blocks:
            for sib in block.operations:
                if sib.operation == op.operation:
                    return block
    return None


def _next_sibling(op):
    block = _containing_block(op)
    siblings = list(block.operations)
    for i, sib in enumerate(siblings):
        if sib.operation == op.operation:
            return siblings[i + 1] if i + 1 < len(siblings) else None
    return None


_ASM_START = ".insn r CUSTOM_3, 0, 0x40, x0, x0, x0"
_ASM_END = ".insn r CUSTOM_3, 0, 0x41, x0, x0, x0"


def _rewrite_loop_steps(builder):
    """Every replayed loop makes one trip from its own start: an scf.for ends at lb + step, an
    affine.for steps over its whole range."""
    from mlir.dialects import arith
    index_t = ir.IndexType.get()
    for node in builder.loop_nodes:
        forop = node.op.operation
        if forop.name == "scf.for":
            ty = forop.operands[0].type
            with ir.InsertionPoint(forop), ir.Location.unknown():
                forop.operands[1] = arith.ConstantOp(ty, ir.IntegerAttr.get(
                    ty, node.loop_start + node.loop_step)).result
            continue
        once = max(node.loop_end - node.loop_start, node.loop_step)
        forop.attributes["step"] = ir.IntegerAttr.get(index_t, once)


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
    for cn in builder.compute_nodes:
        ops = [o.operation for o in cn.operations]
        siblings = [o.operation for o in _containing_block(ops[0]).operations]
        at = [siblings.index(o) if o in siblings else -1 for o in ops]
        if -1 in at or at != sorted(at):
            raise RuntimeError(f"a segment's ops are not siblings in program order: {ops[0].name}")
        inside = siblings[at[0]:at[-1] + 1]
        if any(o.name in (TRANSFER, WAIT) or _holds_boundary(o) for o in inside):
            raise RuntimeError(f"a segment's markers would bracket a DMA or a wait: {ops[0].name}")


def _insert_compute_markers(builder, ids):
    for node, tile_id in zip(builder.compute_nodes, ids, strict=True):
        ops = node.operations
        front = ops[0]
        back = ops[-1]
        ctx = front.operation.context
        ctype_name = COMPUTE_TYPE_NAME[node.compute_type]
        after = _next_sibling(back)
        if after is not None:
            end_ip = ir.InsertionPoint(after)
        else:
            end_ip = ir.InsertionPoint(_containing_block(back))
        _make_inline_asm(ctx, end_ip, _ASM_END)
        _make_inline_asm(ctx, ir.InsertionPoint(front), _ASM_START, ctype_name, tile_id)


def _find_kernel(module):
    funcs = [op for op in module.body.operations
             if op.operation.name == "func.func"]
    for op in funcs:
        if ir.StringAttr(op.operation.attributes["sym_name"]).value == "kernel":
            return op
    return funcs[0] if len(funcs) == 1 else None


def _holds_boundary(op):
    """Does a DMA or a wait sit anywhere in `op`'s regions?"""
    return _holds(op, lambda o: o.operation.name in (TRANSFER, WAIT))


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

    _erase_empty_affine_for(func_op)

    block = func_op.regions[0].blocks[0]
    root = TOGNode()
    for op in block.operations:
        if _is_address_plumbing(op):
            continue
        builder.visit_operation(op, root)


def build_tog_and_mutate(module):
    """Cut the post-vcix IR to one sample tile in place: every replayed loop makes one trip, and
    each segment is bracketed by markers carrying its stamped tile id."""
    builder = TogBuilder()
    _build(module, builder)
    ids = tile_ids(builder, _find_kernel(module))
    _rewrite_loop_steps(builder)
    _check_segments_are_contiguous(builder)
    _insert_compute_markers(builder, ids)


def run_tog(in_path, out_path):
    """Read the post-vcix IR at `in_path` and write its sample-tile cut to `out_path`."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(open(in_path).read(), ctx)
        build_tog_and_mutate(module)
        with open(out_path, "w") as fh:
            fh.write(str(module))
