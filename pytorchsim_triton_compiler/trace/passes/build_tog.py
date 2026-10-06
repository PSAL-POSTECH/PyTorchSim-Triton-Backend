"""Python port of the C++ `-test-tile-operation-graph` analysis pass."""

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    import pytorchsim_triton_compiler  # noqa: F401

import mlir.ir as ir  # noqa: E402

from ...dialects.torchsim import TransferOp, WaitOp  # noqa: E402
TRANSFER, WAIT = TransferOp.OPERATION_NAME, WaitOp.OPERATION_NAME


BASE_NODE = 0
COMPUTE_NODE = 1
LOOP_NODE = 2
DMA_NODE = 3
DMA_WAIT_NODE = 4

VECTOR_COMPUTE = 0
MATMUL_COMPUTE = 1
MATMUL_PRELOAD = 2
CROSS_LANE_COMPUTE = 3

_COMPUTE_TYPE_NAME = {
    VECTOR_COMPUTE: "VectorCompute",
    MATMUL_COMPUTE: "MatmulCompute",
    MATMUL_PRELOAD: "MatmulPreload",
    CROSS_LANE_COMPUTE: "CrossLaneCompute",
}


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

    def __init__(self, name, idx_list, tag_stride_list, tag_divider_list, addr):
        super().__init__(name)
        self.tag_idx_list = idx_list
        self.tag_stride_list = tag_stride_list
        self.tag_divider_list = tag_divider_list
        self.base_addr = addr

    def display(self, out):
        TOGNode.display(self, out)
        out.append(",\n")
        out.append('\t"base_address": "%s",\n' % self.base_addr)
        self._print_str_list("tag_idx_list", self.tag_idx_list, out)
        out.append(",\n")
        self._print_list("tag_stride_list", self.tag_stride_list, out)
        out.append(",\n")
        self._print_list("tag_divider_list", self.tag_divider_list, out)
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


_VCIX_OF_INTRINSIC = {
    "llvm.riscv.sf.vc.iv.se": "vcix.iv",
    "llvm.riscv.sf.vc.ivv.se": "vcix.ivv",
    "llvm.riscv.sf.vc.i.se": "vcix.i",
    "llvm.riscv.sf.vc.v.i.se": "vcix.v.i",
}

_VCIX_PUSHES = ("vcix.iv", "vcix.ivv")


def _vcix_name(op):
    """The VCIX op this is, spelled as a dialect op or as an intrinsic call."""
    name = _op_name(op)
    if name != "llvm.call_intrinsic":
        return name
    attrs = op.operation.attributes
    if "intrin" not in attrs:
        return name
    return _VCIX_OF_INTRINSIC.get(ir.StringAttr(attrs["intrin"]).value, name)


def _vcix_opcode(op):
    """A VCIX op's opcode, from its attribute or its first operand constant."""
    attrs = op.operation.attributes
    if "opcode" in attrs:
        return ir.IntegerAttr(attrs["opcode"]).value
    operands = op.operation.operands
    if not len(operands) or _is_block_arg(operands[0]):
        return None
    const = operands[0].owner
    if (const.name not in ("llvm.mlir.constant", "arith.constant")
            or "value" not in const.attributes):
        return None
    return ir.IntegerAttr(const.attributes["value"]).value


def _xlu_roles(op):
    """Which of the cross-lane unit's "push" and "pop" `op` is or holds."""
    from ...elf.passes.lib_cross_lane import XLU_POP_OPCODE, XLU_PUSH_OPCODE
    roles, stack = set(), [op]
    while stack:
        o = stack.pop()
        name = _vcix_name(o)
        if name in _VCIX_PUSHES and _vcix_opcode(o) == XLU_PUSH_OPCODE:
            roles.add("push")
        elif name == "vcix.v.i" and _vcix_opcode(o) == XLU_POP_OPCODE:
            roles.add("pop")
        for region in o.operation.regions:
            for block in region.blocks:
                stack.extend(block.operations)
    return roles


def _is_block_arg(value):
    return isinstance(value, ir.BlockArgument)


def _defining_op_name(value):
    """Return the op name that defines `value`, or None if it is a block arg."""
    if _is_block_arg(value):
        return None
    return value.owner.name


def _const_index_value(value):
    """If `value` is an arith.constant of index type, return its int value."""
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
    """Identity key for a Value (induction vars map -> loop name)."""
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
    """The subtree's one reader -- see _mlir_util.attr_i64_array for why both
    the dense and the ArrayAttr spelling have to be read here."""
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
    """Port of getCoefficientFromDim: coefficient of `dim` in expr, or -1."""
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
    """Port of collectCoefficientsFromAffineExpr."""
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
    """Port of collectDividersFromAffineExpr."""
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
    "arith.constant", "llvm.mlir.constant",
    "memref.alloc", "memref.reinterpret_cast",
}


class MatmulFsm:
    Idle = 0
    Preload = 1
    MMPush = 2
    MMVpop = 3
    MMEpilogue = 4


class TogBuilder:
    def __init__(self):
        self.nr_loop = 0
        self.loop_var_name = {}
        self.compute_nodes = []
        self.loop_nodes = []
        self.dma_nodes = []
        self._reset_matmul_fsm()

    def _reset_matmul_fsm(self):
        self.matmul_fsm = MatmulFsm.Idle
        self.closed_matmul_compute_node = None
        self.mm_iv_op0_count = 0
        self.mm_expected_vpop = 0
        self.mm_vpop_seen = 0
        self.mm_expected_transfer_writes = 0
        self.mm_transfer_writes_seen = 0
        self.current_preload_node = None
        self.current_matmul_compute_node = None

    def _finish_matmul_block(self):
        self.matmul_fsm = MatmulFsm.Idle
        self.closed_matmul_compute_node = self.current_matmul_compute_node
        self.current_matmul_compute_node = None
        self.mm_iv_op0_count = 0
        self.mm_expected_vpop = 0
        self.mm_vpop_seen = 0
        self.mm_expected_transfer_writes = 0
        self.mm_transfer_writes_seen = 0

    def _reopen_for_pop(self):
        """A pop past the counted ones, before any push, still drains the same matmul: a
        wider accumulator pops more often than the input pushed. Reopen that block for one
        more pop and its write, so the drain stays on the matmul node."""
        cn = self.current_matmul_compute_node or self.closed_matmul_compute_node
        if (cn is None or not self.compute_nodes or self.compute_nodes[-1] is not cn
                or self.matmul_fsm not in (MatmulFsm.Idle, MatmulFsm.MMEpilogue)):
            return False
        if self.matmul_fsm == MatmulFsm.Idle:
            self.current_matmul_compute_node = cn
            self.mm_expected_transfer_writes = self.mm_transfer_writes_seen = 0
            self.matmul_fsm = MatmulFsm.MMEpilogue
        self.mm_expected_transfer_writes += 1
        return True

    def _enter_epilogue_or_finish(self):
        if self.mm_expected_transfer_writes == 0:
            self._finish_matmul_block()
        else:
            self.matmul_fsm = MatmulFsm.MMEpilogue

    def _append_mm_with_write_count(self, op):
        cn = self.current_matmul_compute_node
        if cn is None:
            return
        cn.operations.append(op)
        if _op_name(op) != "vector.transfer_write":
            return
        if self.mm_expected_transfer_writes == 0:
            return
        self.mm_transfer_writes_seen += 1
        if self.mm_transfer_writes_seen >= self.mm_expected_transfer_writes:
            self._finish_matmul_block()

    def _steal_leading_transfer_read(self, parent):
        prepend = []
        if parent is None:
            return prepend
        last = parent.get_last_child()
        if last is None or not isinstance(last, TOGComputeNode):
            return prepend
        if last.compute_type != VECTOR_COMPUTE:
            return prepend
        ops = last.operations
        idx = None
        for i, o in enumerate(ops):
            if _op_name(o) == "vector.transfer_read":
                idx = i
                break
        if idx is None:
            return prepend
        prepend.append(ops[idx])
        del ops[idx]
        if not ops:
            if parent.children and parent.children[-1] is last:
                parent.children.pop()
            else:
                parent.children = [c for c in parent.children if c is not last]
            if last in self.compute_nodes:
                self.compute_nodes.remove(last)
        return prepend

    def _append_vector_compute(self, parent, op):
        """`op` into the VPU's node, or the cross-lane unit's if it holds or follows a push/pop."""
        if parent is None:
            return
        last = parent.get_last_child()
        last_ctype = last.compute_type if isinstance(last, TOGComputeNode) else None
        ctype = VECTOR_COMPUTE
        if _xlu_roles(op) or (last_ctype == CROSS_LANE_COMPUTE
                              and len(op.operation.regions) == 0):
            ctype = CROSS_LANE_COMPUTE
        if last_ctype == ctype:
            last.operations.append(op)
            return
        cn = TOGComputeNode("ComputeNode", 0, ctype)
        if ctype == CROSS_LANE_COMPUTE:
            cn.operations = self._steal_straight_tail(parent)
        cn.operations.append(op)
        cn.op = cn.operations[0]
        parent.add_child(cn)
        cn.add_parent(parent)
        self.compute_nodes.append(cn)

    def _steal_straight_tail(self, parent):
        """The VPU node's trailing straight-line ops, which feed the first push."""
        last = parent.get_last_child()
        if not isinstance(last, TOGComputeNode) or last.compute_type != VECTOR_COMPUTE:
            return []
        ops = last.operations
        i = len(ops)
        while i and len(ops[i - 1].operation.regions) == 0:
            i -= 1
        last.operations, tail = ops[:i], ops[i:]
        if not last.operations:
            parent.children.pop()
            self.compute_nodes.remove(last)
        return tail

    @staticmethod
    def _vcix_iv_opcode(op):
        if _vcix_name(op) not in _VCIX_PUSHES:
            return None
        return _vcix_opcode(op)

    @staticmethod
    def _vcix_vi_opcode(op):
        if _vcix_name(op) != "vcix.v.i":
            return None
        return _vcix_opcode(op)

    @staticmethod
    def _affine_for_bounds(forop):
        oper = forop.operation
        step = ir.IntegerAttr(oper.attributes["step"]).value
        lb_map = ir.AffineMapAttr(oper.attributes["lowerBoundMap"]).value
        ub_map = ir.AffineMapAttr(oper.attributes["upperBoundMap"]).value
        seg_attr = oper.attributes["operandSegmentSizes"]
        seg = [seg_attr[i] for i in range(len(seg_attr))]
        operands = list(oper.operands)

        def single_const(m):
            return len(m.results) == 1 and isinstance(m.results[0], ir.AffineConstantExpr)

        if single_const(lb_map):
            start = ir.AffineConstantExpr(lb_map.results[0]).value
        else:
            start = _const_index_value(operands[0])
        n_lb = seg[0] if seg else 0
        if single_const(ub_map):
            end = ir.AffineConstantExpr(ub_map.results[0]).value
        else:
            end = _const_index_value(operands[n_lb])
        return start, end, step

    @staticmethod
    def _scf_for_bounds(forop):
        """(start, end, step) of an scf.for, or None if a bound is not constant."""
        operands = list(forop.operation.operands)
        if len(operands) < 3:
            return None
        bounds = [_const_index_value(v) for v in operands[:3]]
        return None if any(b is None for b in bounds) else tuple(bounds)

    def _process_dram_indices(self, value, loop_index_list, indirect_box):
        if not _is_block_arg(value) and value.owner.name == "affine.apply":
            apply_op = value.owner
            amap = ir.AffineMapAttr(apply_op.attributes["map"]).value
            operands = list(apply_op.operands)
            if "indirect_access" in apply_op.attributes:
                indirect_box[0] = True
            for op_v in operands:
                if _is_block_arg(op_v):
                    expr = amap.results[0]
                    index_pos = operands.index(op_v)
                    coeff = _get_coefficient_from_dim(expr, index_pos)
                    key = self.loop_var_name[_value_key(op_v)]
                    loop_index_list.append((key, coeff))
                else:
                    self._process_dram_indices(op_v, loop_index_list, indirect_box)
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
        """Walk an index built by arith, accumulating each loop variable's scale."""
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
        if name in SKIP_OPS:
            return

        if name in _LOOP_OPS:
            oper = op.operation
            bounds = (self._affine_for_bounds(op) if name == "affine.for"
                      else self._scf_for_bounds(op))
            if bounds is None or not _replayed_loop(op):
                if node is not None:
                    self._handle_compute(op, node)
                return

            start, end, step = bounds
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

        self._handle_compute(op, node)

    def _handle_compute(self, op, node):
        if _vcix_name(op) in _VCIX_PUSHES:
            if (self.matmul_fsm == MatmulFsm.MMEpilogue
                    and self.current_matmul_compute_node):
                self._finish_matmul_block()
            opc = self._vcix_iv_opcode(op)
            if opc is not None:
                if opc == 1:
                    if self.matmul_fsm == MatmulFsm.Idle:
                        prepend = self._steal_leading_transfer_read(node)
                        cn = TOGComputeNode("ComputeNode", 0, MATMUL_PRELOAD)
                        for o in prepend:
                            cn.operations.append(o)
                        cn.operations.append(op)
                        cn.op = cn.operations[0]
                        node.add_child(cn)
                        cn.add_parent(node)
                        self.compute_nodes.append(cn)
                        self.current_preload_node = cn
                        self.matmul_fsm = MatmulFsm.Preload
                    elif (self.matmul_fsm == MatmulFsm.Preload
                          and self.current_preload_node):
                        self.current_preload_node.operations.append(op)
                    else:
                        raise RuntimeError("vcix.iv opcode=1 invalid state")
                    return
                if opc == 0:
                    if (self.matmul_fsm == MatmulFsm.Preload
                            and self.current_preload_node):
                        cn = TOGComputeNode("ComputeNode", 0, MATMUL_COMPUTE)
                        cn.op = op
                        cn.operations.append(op)
                        node.add_child(cn)
                        cn.add_parent(node)
                        self.compute_nodes.append(cn)
                        self.current_preload_node = None
                        self.current_matmul_compute_node = cn
                        self.matmul_fsm = MatmulFsm.MMPush
                        self.mm_iv_op0_count = 1
                        return
                    if (self.matmul_fsm == MatmulFsm.MMPush
                            and self.current_matmul_compute_node):
                        self.current_matmul_compute_node.operations.append(op)
                        self.mm_iv_op0_count += 1
                        return
                    raise RuntimeError("vcix.iv opcode=0 invalid state")
            if (self.matmul_fsm == MatmulFsm.MMEpilogue
                    and self.current_matmul_compute_node):
                self._append_mm_with_write_count(op)
                return
            self._append_vector_compute(node, op)
            return

        if _vcix_name(op) == "vcix.v.i":
            viopc = self._vcix_vi_opcode(op)
            if viopc is not None and viopc == 2:
                if (self.matmul_fsm == MatmulFsm.MMPush
                        and self.current_matmul_compute_node):
                    self.mm_expected_vpop = self.mm_iv_op0_count
                    self.mm_expected_transfer_writes = self.mm_expected_vpop
                    self.mm_transfer_writes_seen = 0
                    self.matmul_fsm = MatmulFsm.MMVpop
                    self.mm_vpop_seen = 1
                    self.current_matmul_compute_node.operations.append(op)
                    if self.mm_vpop_seen >= self.mm_expected_vpop:
                        self._enter_epilogue_or_finish()
                    return
                if (self.matmul_fsm == MatmulFsm.MMVpop
                        and self.current_matmul_compute_node):
                    self.current_matmul_compute_node.operations.append(op)
                    self.mm_vpop_seen += 1
                    if self.mm_vpop_seen >= self.mm_expected_vpop:
                        self._enter_epilogue_or_finish()
                    return
            if viopc is not None and viopc == 2 and self._reopen_for_pop():
                self._append_mm_with_write_count(op)
                return
            if (self.matmul_fsm == MatmulFsm.MMEpilogue
                    and self.current_matmul_compute_node):
                self._append_mm_with_write_count(op)
                return
            self._append_vector_compute(node, op)
            return

        if self.matmul_fsm == MatmulFsm.Preload and self.current_preload_node:
            self.current_preload_node.operations.append(op)
            return
        if (self.matmul_fsm == MatmulFsm.MMEpilogue
                and self.current_matmul_compute_node):
            self._append_mm_with_write_count(op)
            return
        if (self.matmul_fsm in (MatmulFsm.MMPush, MatmulFsm.MMVpop)
                and self.current_matmul_compute_node):
            self._append_mm_with_write_count(op)
            return
        self._append_vector_compute(node, op)

    @staticmethod
    def _transfer_is_load(op):
        """True if a `torchsim.transfer` is a load (DRAM -> SRAM), False for a store.
        From `dma_kind` alone, which its ODS requires: MVIN* load, MVOUT store."""
        return op.operation.opview.dma_kind.value.upper().startswith("MVIN")

    def _dma_start_fields(self, op):
        """A `torchsim.transfer` in the legacy src/dst view, its operands read by
        the names its ODS gives them; the offset is the gather index, if any."""
        tx = op.operation.opview
        dram, dram_idx, sram, sram_idx = tx.dram, tx.dram_idx, tx.sram, tx.sram_idx
        tag_idx, offset = tx.tag_idx, tx.index

        if self._transfer_is_load(op):
            src, src_idx = dram, dram_idx
            dst, dst_idx = sram, sram_idx
        else:
            src, src_idx = sram, sram_idx
            dst, dst_idx = dram, dram_idx
        return {
            "src": src, "src_type": src.type, "src_indices": [src_idx],
            "dst": dst, "dst_type": dst.type, "dst_indices": [dst_idx],
            "tag_indices": [tag_idx],
            "offset": offset,
        }

    def _handle_dma_start(self, op, node):
        oper = op.operation
        f = self._dma_start_fields(op)
        dma_async = bool(self._get_async(oper))

        dst_space = _memref_space(f["dst_type"])
        src_space = _memref_space(f["src_type"])

        subtile = _int_array_attr(oper, "subtile_size")

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
        tile_shape = subtile if subtile else list(tile_mt.shape)
        tile_size = [int(x) for x in tile_shape]

        tile_stride = []
        ds = _int_array_attr(oper, "dram_stride")
        if ds:
            tile_stride = list(ds)

        loop_index_map = []
        indirect_box = [False]
        self._process_dram_indices(dram_indices[0], loop_index_map, indirect_box)

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
                              indirect_box[0])
        dma_node.op = op
        self.dma_nodes.append(dma_node)
        node.add_child(dma_node)
        dma_node.add_parent(node)

    def _handle_dma_wait(self, op, node):
        oper = op.operation
        operands = list(oper.operands)
        slot = operands[0]
        tag_indices = operands[0:1]

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

        address = "arg"
        for use in slot.uses:
            user = use.owner
            if user.name == TRANSFER:
                f = self._dma_start_fields(user)
                dst_space = _memref_space(f["dst_type"])
                src_space = _memref_space(f["src_type"])
                dram_memref = None
                if dst_space == 0 and src_space == 1:
                    dram_memref = f["dst"]
                elif dst_space == 1 and src_space == 0:
                    dram_memref = f["src"]
                if "dram_arg" in user.attributes:
                    address += str(ir.IntegerAttr(user.attributes["dram_arg"]).value)
                elif dram_memref is not None and _is_block_arg(dram_memref):
                    address += str(ir.BlockArgument(dram_memref).arg_number)

        if len(tag_stride_list) == 0:
            tag_stride_list.append(1)
            tag_divider_list.append(1)

        wait_node = TOGDMAWaitNode("DMAWaitNode", tag_index_list, tag_stride_list,
                                   tag_divider_list, address)
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
    if op.operation.name != "affine.for":
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


def _make_inline_asm(ctx, ip, asm_string, compute_type=None):
    """Build an llvm.inline_asm compute marker identical to the C++ pass."""
    i64 = ir.IntegerType.get_signless(64)
    attrs = {
        "has_side_effects": ir.UnitAttr.get(),
        "asm_dialect": ir.IntegerAttr.get(i64, 0),
        "asm_string": ir.StringAttr.get(asm_string),
        "constraints": ir.StringAttr.get("~{a0},~{memory}"),
    }
    if compute_type is not None:
        attrs["compute_type"] = ir.StringAttr.get(compute_type)
    return ir.Operation.create(
        "llvm.inline_asm", results=[], operands=[], attributes=attrs,
        loc=ir.Location.unknown(ctx), ip=ip)


def _containing_block(op):
    """Return the Block that directly contains `op`."""
    parent = op.operation.parent
    for region in parent.regions:
        for block in region.blocks:
            for sib in block.operations:
                if sib.operation == op.operation:
                    return block
    return None


def _next_sibling(op):
    """Return the op immediately following `op` in its block, or None."""
    block = _containing_block(op)
    siblings = list(block.operations)
    for i, sib in enumerate(siblings):
        if sib.operation == op.operation:
            return siblings[i + 1] if i + 1 < len(siblings) else None
    return None


_ASM_START = ".insn r CUSTOM_3, 0, 0x40, x0, x0, x0"
_ASM_END = ".insn r CUSTOM_3, 0, 0x41, x0, x0, x0"


def _rewrite_loop_steps(builder):
    """Sample mode: rewrite each loop node's affine.for step to its end.

    Done after the graph is built so the printed graph keeps the original step.
    """
    index_t = ir.IndexType.get()
    for node in builder.loop_nodes:
        forop = node.op.operation
        if forop.name == "scf.for":
            forop.operands[1] = forop.operands[2]
            continue
        once = max(node.loop_end - node.loop_start, node.loop_step)
        forop.attributes["step"] = ir.IntegerAttr.get(index_t, once)


def _insert_compute_markers(builder):
    """Insert inline-asm start/end markers around each compute node's ops."""
    for node in builder.compute_nodes:
        ops = node.operations
        if not ops:
            continue
        front = ops[0]
        back = ops[-1]
        ctx = front.operation.context
        ctype_name = _COMPUTE_TYPE_NAME[node.compute_type]
        after = _next_sibling(back)
        if after is not None:
            end_ip = ir.InsertionPoint(after)
        else:
            end_ip = ir.InsertionPoint(_containing_block(back))
        _make_inline_asm(ctx, end_ip, _ASM_END)
        _make_inline_asm(ctx, ir.InsertionPoint(front), _ASM_START, ctype_name)


def _find_kernel(module):
    """The kernel function: named `kernel` in PyTorchSim's codegen, else the
    module's only func.func (pytorchsim-triton-compiler carries the Triton kernel's own name).
    Declines when there is more than one -- the intent would be a guess."""
    funcs = [op for op in module.body.operations
             if op.operation.name == "func.func"]
    for op in funcs:
        if ir.StringAttr(op.operation.attributes["sym_name"]).value == "kernel":
            return op
    return funcs[0] if len(funcs) == 1 else None


def _replayed_loop(op):
    """Whether the trace replays this loop, so gem5 must sample one trip of it."""
    found = [False]

    def walk(o):
        if found[0]:
            return
        if (o.operation.name in (TRANSFER, WAIT)
                or _vcix_name(o) in _VCIX_PUSHES):
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
    """Scalar index/integer math (DMA offsets, mask extents) and the terminator."""
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
    """Build the graph and return its display string, populating `builder`."""
    func_op = _find_kernel(module)
    if func_op is None:
        return ""

    _erase_empty_affine_for(func_op)

    block = func_op.regions[0].blocks[0]
    out = []
    root = TOGNode("root")
    builder._reset_matmul_fsm()
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
    if sample_mode:
        _rewrite_loop_steps(builder)
    _insert_compute_markers(builder)
    return result


def run_tog(in_path, tog_out_path, custom_out_path, sample_mode=True, vectorlane=128):
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
    parser.add_argument("--vectorlane", type=int, default=128)
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
