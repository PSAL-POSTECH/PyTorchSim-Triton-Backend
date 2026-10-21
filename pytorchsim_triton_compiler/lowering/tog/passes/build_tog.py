"""Python port of the C++ `-test-tile-operation-graph` analysis pass."""

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    import pytorchsim_triton_compiler  # noqa: F401

import mlir.ir as ir  # noqa: E402

from ....dialects.torchsim import TransferOp, WaitOp, is_load, moved_shape  # noqa: E402
from ...passes.lib_ir import VIEW_OPS, bounds_lives_only, loop_bounds  # noqa: E402
from ....contract.kernel_object import CYCLE_CALL_GLOBAL  # noqa: E402
TRANSFER, WAIT = TransferOp.OPERATION_NAME, WaitOp.OPERATION_NAME


BASE_NODE = 0
COMPUTE_NODE = 1
LOOP_NODE = 2
DMA_NODE = 3
DMA_WAIT_NODE = 4

SEGMENT_COMPUTE = 0
"""A segment holds the core for its cycles; TOGSim's vector pipeline is the one queue per core."""

COMPUTE_TYPE_NAME = {SEGMENT_COMPUTE: "VectorCompute"}
"""Every compute type the TOG emits, by the name its marker carries."""

CORE = "__CORE__"
"""The buffer every segment reads and writes: one core runs its segments in program order."""

TILE_ID = "torchsim.tile_id"
"""The tile a post-vcix op's segment is: stamped once by the skeleton (stamp_tile_ids), read back by
the TOG and the cycle binary (tile_ids), so the two walks name each segment the same tile."""


import threading  # noqa: E402

_ids = threading.local()


def _new_id():
    i = getattr(_ids, "counter", 0)
    _ids.counter = i + 1
    return i


def _reset_ids():
    _ids.counter = 0


class TOGNode:
    kind = BASE_NODE

    def __init__(self, name, node_id=None):
        self.node_id = _new_id() if node_id is None else node_id
        self.node_name = name
        self.parents = []
        self.children = []
        self.op = None

    def get_last_child(self):
        return self.children[-1] if self.children else None

    def add_child(self, c):
        self.children.append(c)

    def add_parent(self, p):
        self.parents.append(p)

    @staticmethod
    def _print_list(name, vec, out):
        out.append('\t"%s": [' % name)
        out.append(",".join(str(v) for v in vec))
        out.append("]")

    @staticmethod
    def _print_str_list(name, vec, out):
        out.append('\t"%s": [' % name)
        out.append(",".join('"%s"' % v for v in vec))
        out.append("]")

    def _print_node_list(self, name, vec, out):
        out.append('\t"%s": [' % name)
        out.append(",".join(str(n.node_id) for n in vec))
        out.append("]")

    def display(self, out):
        out.append("%d : {\n" % self.node_id)
        out.append('\t"node_id": %d,\n' % self.node_id)
        out.append('\t"node_name": "%s",\n' % self.node_name)
        out.append('\t"node_type": %d,\n' % self.kind)
        self._print_node_list("parents", self.parents, out)
        out.append(",\n")
        self._print_node_list("children", self.children, out)
        if self.kind == BASE_NODE:
            out.append("\n}\n")

    def bfs(self, out):
        if self.node_name == "root":
            out.append("graph = {\n")
        self.display(out)
        out.append(",")
        for child in self.children:
            child.bfs(out)
        if self.node_name == "root":
            out.append("}")


class TOGLoopNode(TOGNode):
    kind = LOOP_NODE

    def __init__(self, name, idx, start, end, step, loop_type):
        super().__init__(name)
        self.loop_idx = idx
        self.loop_start = start
        self.loop_end = end
        self.loop_step = step
        self.loop_type = loop_type

    def display(self, out):
        TOGNode.display(self, out)
        out.append(",\n")
        out.append('\t"loop_index": "%s",\n' % self.loop_idx)
        out.append('\t"loop_start": %d,\n' % self.loop_start)
        out.append('\t"loop_end": %d,\n' % self.loop_end)
        out.append('\t"loop_step": %d,\n' % self.loop_step)
        out.append('\t"loop_type": "%s"\n' % self.loop_type)
        out.append("}\n")


class TOGDMANode(TOGNode):
    kind = DMA_NODE

    def __init__(self, name, addr, tile_size, tile_stride, elem_size, is_write,
                 is_async, tag_idx_list, tag_stride_list, loop_idx_list,
                 loop_stride_list, indirect_mode):
        super().__init__(name)
        self.base_addr = addr
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.element_size = elem_size
        self.is_write = is_write
        self.is_async = is_async
        self.tag_idx_list = tag_idx_list
        self.tag_stride_list = tag_stride_list
        self.loop_idx_list = loop_idx_list
        self.loop_stride_list = loop_stride_list
        self.indirect_mode = indirect_mode

    def display(self, out):
        TOGNode.display(self, out)
        out.append(",\n")
        out.append('\t"is_write": %d,\n' % int(self.is_write))
        out.append('\t"is_async": %d,\n' % int(self.is_async))
        out.append('\t"base_address": "%s",\n' % self.base_addr)
        out.append('\t"indirect_mode": %d,\n' % int(self.indirect_mode))
        self._print_list("tile_size", self.tile_size, out)
        out.append(",\n")
        self._print_list("tile_stride", self.tile_stride, out)
        out.append(",\n")
        self._print_str_list("tag_idx_list", self.tag_idx_list, out)
        out.append(",\n")
        self._print_list("tag_stride_list", self.tag_stride_list, out)
        out.append(",\n")
        self._print_str_list("loop_idx_list", self.loop_idx_list, out)
        out.append(",\n")
        self._print_list("loop_stride_list", self.loop_stride_list, out)
        out.append(",\n")
        out.append('\t"element_size": %d\n' % self.element_size)
        out.append("}\n")


class TOGDMAWaitNode(TOGNode):
    kind = DMA_WAIT_NODE

    def __init__(self, name, idx_list, tag_stride_list, tag_divider_list, addr, amount):
        super().__init__(name)
        self.tag_idx_list = idx_list
        self.tag_stride_list = tag_stride_list
        self.tag_divider_list = tag_divider_list
        self.base_addr = addr
        self.amount = amount

    def display(self, out):
        TOGNode.display(self, out)
        out.append(",\n")
        out.append('\t"base_address": "%s",\n' % self.base_addr)
        self._print_str_list("tag_idx_list", self.tag_idx_list, out)
        out.append(",\n")
        self._print_list("tag_stride_list", self.tag_stride_list, out)
        out.append(",\n")
        self._print_list("tag_divider_list", self.tag_divider_list, out)
        out.append(",\n")
        out.append('\t"amount": %s\n' % ("null" if self.amount is None else self.amount))
        out.append("}\n")


class TOGComputeNode(TOGNode):
    kind = COMPUTE_NODE

    def __init__(self, name, cycle, ctype):
        super().__init__(name)
        self.compute_cycle = cycle
        self.compute_type = ctype
        self.operations = []

    def display(self, out):
        TOGNode.display(self, out)
        out.append(",\n")
        out.append('\t"compute_cycle": %d,\n' % self.compute_cycle)
        out.append('\t"compute_type": %d\n' % self.compute_type)
        out.append("}\n")


_INDEX_ARITH = ("arith.index_cast", "arith.index_castui", "arith.extsi",
                "arith.muli", "arith.addi", "arith.subi")


def _op_name(value_or_op):
    return value_or_op.operation.name


def _is_block_arg(value):
    return isinstance(value, ir.BlockArgument)


def _defining_op_name(value):
    if _is_block_arg(value):
        return None
    return value.owner.name


def _const_index_value(value):
    if _is_block_arg(value):
        return None
    owner = value.owner
    if owner.name != "arith.constant":
        return None
    attr = owner.attributes["value"]
    try:
        return ir.IntegerAttr(attr).value
    except Exception:
        return None


def _value_key(value):
    if _is_block_arg(value):
        ba = ir.BlockArgument(value)
        return ("ba", ba.owner, ba.arg_number)
    return ("res", value.owner, 0)


def _memref_space(memref_type):
    mt = ir.MemRefType(memref_type)
    sp = mt.memory_space
    if sp is None:
        return 0
    return ir.IntegerAttr(sp).value


def _int_array_attr(op, key):
    from ._mlir_util import attr_i64_array
    return attr_i64_array(op, key)


def _is_function_of_dim(expr, dim):
    if isinstance(expr, ir.AffineDimExpr):
        return ir.AffineDimExpr(expr).position == dim
    if isinstance(expr, ir.AffineBinaryExpr):
        b = ir.AffineBinaryExpr(expr)
        return _is_function_of_dim(b.lhs, dim) or _is_function_of_dim(b.rhs, dim)
    return False


def _get_coefficient_from_dim(expr, dim):
    if isinstance(expr, ir.AffineMulExpr):
        b = ir.AffineMulExpr(expr)
        lhs, rhs = b.lhs, b.rhs
        if isinstance(lhs, ir.AffineConstantExpr) and isinstance(rhs, ir.AffineDimExpr):
            if _is_function_of_dim(rhs, dim):
                return ir.AffineConstantExpr(lhs).value
        elif isinstance(rhs, ir.AffineConstantExpr) and isinstance(lhs, ir.AffineDimExpr):
            if _is_function_of_dim(lhs, dim):
                return ir.AffineConstantExpr(rhs).value
        return -1
    if isinstance(expr, ir.AffineAddExpr):
        b = ir.AffineAddExpr(expr)
        r = _get_coefficient_from_dim(b.lhs, dim)
        if r != -1:
            return r
        r = _get_coefficient_from_dim(b.rhs, dim)
        if r != -1:
            return r
        return -1
    if isinstance(expr, ir.AffineDimExpr):
        if _is_function_of_dim(expr, dim):
            return 1
    return -1


def _collect_coefficients(expr):
    out = []

    def rec(e):
        if isinstance(e, ir.AffineMulExpr):
            b = ir.AffineMulExpr(e)
            lhs, rhs = b.lhs, b.rhs
            if isinstance(lhs, ir.AffineConstantExpr):
                if isinstance(rhs, ir.AffineDimExpr):
                    out.append(ir.AffineConstantExpr(lhs).value)
                elif isinstance(rhs, ir.AffineFloorDivExpr):
                    out.append(ir.AffineConstantExpr(lhs).value)
            elif isinstance(rhs, ir.AffineConstantExpr):
                if isinstance(lhs, ir.AffineDimExpr):
                    out.append(ir.AffineConstantExpr(rhs).value)
                elif isinstance(lhs, ir.AffineFloorDivExpr):
                    out.append(ir.AffineConstantExpr(rhs).value)
        elif isinstance(e, ir.AffineAddExpr):
            b = ir.AffineAddExpr(e)
            rec(b.lhs)
            rec(b.rhs)
        elif isinstance(e, ir.AffineFloorDivExpr):
            b = ir.AffineFloorDivExpr(e)
            if isinstance(b.rhs, ir.AffineConstantExpr):
                out.append(ir.AffineConstantExpr(b.rhs).value)
        elif isinstance(e, ir.AffineDimExpr):
            out.append(1)

    rec(expr)
    return out


def _collect_dividers(expr):
    out = []

    def rec(e):
        if isinstance(e, ir.AffineMulExpr):
            b = ir.AffineMulExpr(e)
            lhs, rhs = b.lhs, b.rhs
            if isinstance(lhs, ir.AffineConstantExpr):
                if isinstance(rhs, ir.AffineDimExpr):
                    out.append(1)
                elif isinstance(rhs, ir.AffineFloorDivExpr):
                    rec(rhs)
            elif isinstance(rhs, ir.AffineConstantExpr):
                if isinstance(lhs, ir.AffineDimExpr):
                    out.append(1)
                elif isinstance(lhs, ir.AffineFloorDivExpr):
                    rec(lhs)
        elif isinstance(e, ir.AffineAddExpr):
            b = ir.AffineAddExpr(e)
            rec(b.lhs)
            rec(b.rhs)
        elif isinstance(e, ir.AffineFloorDivExpr):
            b = ir.AffineFloorDivExpr(e)
            if isinstance(b.rhs, ir.AffineConstantExpr):
                out.append(ir.AffineConstantExpr(b.rhs).value)
        elif isinstance(e, ir.AffineDimExpr):
            out.append(1)

    rec(expr)
    return out


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
            raise NotImplementedError(
                f"a {_op_name(loop)} whose trip count is not a constant holds a DMA or a wait: "
                f"a TOG holds counted loops only")

    def append(self, op, node):
        """`op` joins the segment its parent left open, or opens one: what closes a segment is a
        DMA, a wait or a replayed loop becoming the parent's last child."""
        if _holds_boundary(op):
            raise NotImplementedError(
                f"a DMA or a wait inside {_op_name(op)}: a TOG has no branches, so it "
                f"cannot say whether that transfer happens")
        last = node.get_last_child()
        if isinstance(last, TOGComputeNode):
            last.operations.append(op)
            return
        cn = TOGComputeNode("ComputeNode", 0, SEGMENT_COMPUTE)
        cn.operations.append(op)
        cn.op = op
        node.add_child(cn)
        cn.add_parent(node)
        self.builder.compute_nodes.append(cn)


class TogBuilder:
    def __init__(self):
        self.nr_loop = 0
        self.loop_var_name = {}
        self.compute_nodes = []
        self.loop_nodes = []
        self.dma_nodes = []
        self.segments = Segments(self)

    def _process_dram_indices(self, value, loop_index_list):
        if not _is_block_arg(value) and value.owner.name == "affine.apply":
            apply_op = value.owner
            amap = ir.AffineMapAttr(apply_op.attributes["map"]).value
            operands = list(apply_op.operands)
            for op_v in operands:
                if _is_block_arg(op_v):
                    expr = amap.results[0]
                    index_pos = operands.index(op_v)
                    coeff = _get_coefficient_from_dim(expr, index_pos)
                    key = self.loop_var_name[_value_key(op_v)]
                    loop_index_list.append((key, coeff))
                else:
                    self._process_dram_indices(op_v, loop_index_list)
        elif _is_block_arg(value):
            key = self.loop_var_name.get(_value_key(value))
            if key is not None:
                loop_index_list.append((key, 1))
        elif value.owner.name in _INDEX_ARITH:
            self._process_arith_index(value, 1, loop_index_list)
        else:
            c = _const_index_value(value)
            if c is not None:
                loop_index_list.append(("c" + str(c), c))

    def _process_arith_index(self, value, coeff, loop_index_list):
        if _is_block_arg(value):
            key = self.loop_var_name.get(_value_key(value))
            if key is not None:
                loop_index_list.append((key, coeff))
            return
        name = value.owner.name
        operands = list(value.owner.operands)
        if name in ("arith.index_cast", "arith.index_castui", "arith.extsi"):
            self._process_arith_index(operands[0], coeff, loop_index_list)
        elif name == "arith.muli":
            for a, b in ((0, 1), (1, 0)):
                c = _const_index_value(operands[a])
                if c is not None:
                    self._process_arith_index(operands[b], coeff * c,
                                              loop_index_list)
                    return
        elif name in ("arith.addi", "arith.subi"):
            for v in operands[:2]:
                self._process_arith_index(v, coeff, loop_index_list)

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
            loop_index = "loop_arg%03d" % self.nr_loop
            self.nr_loop += 1
            iter_var = oper.regions[0].blocks[0].arguments[0]
            self.loop_var_name[_value_key(iter_var)] = loop_index
            loop_node = TOGLoopNode("loopNode", loop_index, start, end, step, "")
            loop_node.op = op
            self.loop_nodes.append(loop_node)
            if node is not None:
                node.add_child(loop_node)
                loop_node.add_parent(node)
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
        dram, dram_idx, sram, sram_idx = tx.dram, tx.dram_idx, tx.sram, tx.sram_idx
        sem, offset = tx.sem, tx.index

        if is_load(op):
            src, src_idx = dram, dram_idx
            dst, dst_idx = sram, sram_idx
        else:
            src, src_idx = sram, sram_idx
            dst, dst_idx = dram, dram_idx
        return {
            "src": src, "src_type": src.type, "src_indices": [src_idx],
            "dst": dst, "dst_type": dst.type, "dst_indices": [dst_idx],
            "tag_indices": [sem],
            "offset": offset,
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
            dram_indices = f["dst_indices"]
        elif dst_space == 1 and src_space == 0:
            is_write = False
            tile_type = f["dst_type"]
            dram_memref = f["src"]
            dram_indices = f["src_indices"]
        else:
            raise RuntimeError("Unexpected memory space")

        tile_mt = ir.MemRefType(tile_type)
        tile_size = [int(x) for x in moved_shape(oper)]

        tile_stride = []
        ds = _int_array_attr(oper, "dram_stride")
        if ds:
            tile_stride = list(ds)

        loop_index_map = []
        self._process_dram_indices(dram_indices[0], loop_index_map)

        reordered = {}
        for key, stride in loop_index_map:
            if key not in reordered:
                reordered[key] = stride
        loop_idx_list = []
        loop_stride_list = []
        for key in sorted(reordered.keys()):
            loop_idx_list.append(key)
            loop_stride_list.append(reordered[key])

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

        tag_index_list = []
        tag_stride_list = []
        for tag_idx in f["tag_indices"]:
            c = _const_index_value(tag_idx)
            if c is not None:
                tag_index_list.append(str(c))
                continue
            if not _is_block_arg(tag_idx) and tag_idx.owner.name == "affine.apply":
                apply_op = tag_idx.owner
                for operand in apply_op.operands:
                    if _is_block_arg(operand):
                        tag_index_list.append(
                            self.loop_var_name[_value_key(operand)])
                    elif (not _is_block_arg(operand)
                          and operand.owner.name == "affine.apply"):
                        nested = operand.owner
                        for nop in nested.operands:
                            if _is_block_arg(nop):
                                tag_index_list.append(
                                    self.loop_var_name[_value_key(nop)])
                            else:
                                nc = _const_index_value(nop)
                                if nc is not None:
                                    tag_index_list.append(str(nc))
                    else:
                        oc = _const_index_value(operand)
                        if oc is not None:
                            tag_index_list.append(str(oc))
                amap = ir.AffineMapAttr(apply_op.attributes["map"]).value
                tag_stride_list = _collect_coefficients(amap.results[0])

        if len(tag_index_list) == 0:
            tag_index_list.append("0")
        if len(tag_stride_list) == 0:
            tag_stride_list.append(1)

        dma_node = TOGDMANode("DMANode", address, tile_size, tile_stride,
                              element_size, is_write, dma_async, tag_index_list,
                              tag_stride_list, loop_idx_list, loop_stride_list,
                              f["offset"] is not None)
        dma_node.op = op
        self.dma_nodes.append(dma_node)
        node.add_child(dma_node)
        dma_node.add_parent(node)

    def _dram_arg_of(self, op):
        """The DRAM argument a transfer moves: its `dram_arg`, else its DRAM memref's block arg."""
        if "dram_arg" in op.attributes:
            return ir.IntegerAttr(op.attributes["dram_arg"]).value
        f = self._dma_start_fields(op)
        dst_space, src_space = _memref_space(f["dst_type"]), _memref_space(f["src_type"])
        dram = f["dst"] if (dst_space, src_space) == (0, 1) else (
            f["src"] if (dst_space, src_space) == (1, 0) else None)
        if dram is not None and _is_block_arg(dram):
            return ir.BlockArgument(dram).arg_number
        return None

    def _handle_dma_wait(self, op, node):
        oper = op.operation
        slot = oper.opview.sem
        tag_indices = [slot]

        tag_index_list = []
        tag_stride_list = []
        tag_divider_list = []
        for tag_idx in tag_indices:
            if not _is_block_arg(tag_idx) and tag_idx.owner.name == "affine.apply":
                apply_op = tag_idx.owner
                for operand in apply_op.operands:
                    if _is_block_arg(operand):
                        tag_index_list.append(
                            self.loop_var_name[_value_key(operand)])
                    else:
                        c = _const_index_value(operand)
                        if c is not None:
                            tag_index_list.append(str(c))
                amap = ir.AffineMapAttr(apply_op.attributes["map"]).value
                tag_stride_list = _collect_coefficients(amap.results[0])
                tag_divider_list = _collect_dividers(amap.results[0])

        args = []
        for use in slot.uses:
            user = use.owner
            if user.name == TRANSFER and user.operation.opview.sem == slot:
                arg = self._dram_arg_of(user)
                if arg is not None and arg not in args:
                    args.append(arg)
        address = ",".join("arg%d" % a for a in args) or "arg"
        amount = _const_index_value(oper.opview.amount)

        if len(tag_stride_list) == 0:
            tag_stride_list.append(1)
            tag_divider_list.append(1)

        wait_node = TOGDMAWaitNode("DMAWaitNode", tag_index_list, tag_stride_list,
                                   tag_divider_list, address, amount)
        wait_node.op = op
        self.dma_nodes.append(wait_node)
        node.add_child(wait_node)
        wait_node.add_parent(node)

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
    func_op = _find_kernel(module)
    if func_op is None:
        return ""

    _erase_empty_affine_for(func_op)

    block = func_op.regions[0].blocks[0]
    out = []
    root = TOGNode("root")
    for op in block.operations:
        if _is_address_plumbing(op):
            continue
        builder.visit_operation(op, root)
    root.bfs(out)
    return "".join(out)


def build_tog_string(module):
    _reset_ids()
    return _build(module, TogBuilder())


def build_tog_and_mutate(module, sample_mode=True):
    """Build the TOG and apply the C++ pass IR mutation in place."""
    _reset_ids()
    builder = TogBuilder()
    result = _build(module, builder)
    ids = tile_ids(builder, _find_kernel(module))
    if sample_mode:
        _rewrite_loop_steps(builder)
    _check_segments_are_contiguous(builder)
    _insert_compute_markers(builder, ids)
    return result


def run_tog(in_path, tog_out_path, custom_out_path, sample_mode=True):
    """In-process replacement for the C++ -test-tile-operation-graph pass."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(open(in_path).read(), ctx)
        graph = build_tog_and_mutate(module, sample_mode=bool(sample_mode))
        with open(custom_out_path, "w") as fh:
            fh.write(str(module))
    with open(tog_out_path, "w") as fh:
        fh.write(graph)


def main(argv):
    import argparse

    parser = argparse.ArgumentParser(prog="build_tog.py")
    parser.add_argument("input")
    parser.add_argument("--sample-mode", type=int, default=1, choices=(0, 1))
    parser.add_argument("--mutate-out", default=None)
    args = parser.parse_args(argv[1:])

    with open(args.input) as fh:
        src = fh.read()
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(src, ctx)
        if args.mutate_out is not None:
            result = build_tog_and_mutate(module, sample_mode=bool(args.sample_mode))
            with open(args.mutate_out, "w") as fh:
                fh.write(str(module))
        else:
            result = build_tog_string(module)
    sys.stdout.write(result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
