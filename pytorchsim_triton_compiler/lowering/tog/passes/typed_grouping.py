"""DEPRECATED: the typed grouping, kept runnable so its cycles can be compared with the segment
grouping's. It groups compute by the unit and opcode that runs it (compute_type 1 matmul, 2 preload,
3 cross-lane) and is what --tog-grouping typed selects; build_tog.py imports this only then."""

from .build_tog import (
    TRANSFER,
    WAIT,
    VECTOR_COMPUTE,
    TOGComputeNode,
    ir,
    _holds,
    _holds_boundary,
    _is_block_arg,
    _op_name,
)

MATMUL_COMPUTE = 1
MATMUL_PRELOAD = 2
CROSS_LANE_COMPUTE = 3

COMPUTE_TYPE_NAME = {
    VECTOR_COMPUTE: "VectorCompute",
    MATMUL_COMPUTE: "MatmulCompute",
    MATMUL_PRELOAD: "MatmulPreload",
    CROSS_LANE_COMPUTE: "CrossLaneCompute",
}

SA_WEIGHTS = "__SA_WEIGHTS__"
"""The virtual buffer a preload writes and its matmul reads."""

XLU_STATE = "__XLU_STATE__"
"""The virtual buffer a cross-lane push writes and a pop reads."""


_VCIX_OF_INTRINSIC = {
    "llvm.riscv.sf.vc.iv.se": "vcix.iv",
    "llvm.riscv.sf.vc.ivv.se": "vcix.ivv",
    "llvm.riscv.sf.vc.i.se": "vcix.i",
    "llvm.riscv.sf.vc.v.i.se": "vcix.v.i",
}

_VCIX_PUSHES = ("vcix.iv", "vcix.ivv")


def _vcix_name(op):
    name = _op_name(op)
    if name != "llvm.call_intrinsic":
        return name
    attrs = op.operation.attributes
    if "intrin" not in attrs:
        return name
    return _VCIX_OF_INTRINSIC.get(ir.StringAttr(attrs["intrin"]).value, name)


def _vcix_opcode(op):
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
    from ...ext.vcix_tpuv6.lib_xlu import XLU_POP_OPCODE, XLU_PUSH_OPCODE
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


class MatmulFsm:
    Idle = 0
    Preload = 1
    MMPush = 2
    MMVpop = 3
    MMEpilogue = 4


class TypedGrouping:
    """Compute grouped by unit: a preload node, a matmul node from its first push to its last
    pop's store, a cross-lane node, and a vector node for what is left."""

    name = "typed"
    type_names = COMPUTE_TYPE_NAME

    def __init__(self, builder):
        self.builder = builder
        self.reset()

    def reset(self):
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
        cn = self.current_matmul_compute_node or self.closed_matmul_compute_node
        if (cn is None or not self.builder.compute_nodes or self.builder.compute_nodes[-1] is not cn
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
            if last in self.builder.compute_nodes:
                self.builder.compute_nodes.remove(last)
        return prepend

    def _append_vector_compute(self, parent, op):
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
        self.builder.compute_nodes.append(cn)

    def _steal_straight_tail(self, parent):
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
            self.builder.compute_nodes.remove(last)
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

    def carries_on(self, loop):
        """A loop met while a matmul's pushes still wait for their pops carries that stream on:
        it is part of the matmul node, not a loop the TOG holds."""
        if self.current_matmul_compute_node is None:
            return False
        if self.matmul_fsm not in (MatmulFsm.MMPush, MatmulFsm.MMVpop):
            return False
        if _holds_boundary(loop):
            raise NotImplementedError(
                f"a loop inside an open matmul stream holds a DMA or a wait: {loop.operation.name}")
        return True

    def append(self, op, node):
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
                        self.builder.compute_nodes.append(cn)
                        self.current_preload_node = cn
                        self.matmul_fsm = MatmulFsm.Preload
                    elif (self.matmul_fsm == MatmulFsm.Preload
                          and self.current_preload_node):
                        self.current_preload_node.operations.append(op)
                    elif (self.matmul_fsm in (MatmulFsm.MMPush, MatmulFsm.MMVpop)
                          and self.current_matmul_compute_node):
                        self.current_matmul_compute_node.operations.append(op)
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
                        self.builder.compute_nodes.append(cn)
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

    def replays(self, loop):
        """A loop the TOG holds and the sample runs once: one holding a DMA, a wait or a unit
        push."""
        return _holds(loop, lambda o: o.operation.name in (TRANSFER, WAIT)
                      or _vcix_name(o) in _VCIX_PUSHES)

    def refuse_unreplayed(self, loop):
        pass

    def buffers(self, cn, reads, writes):
        """`reads` and `writes` with the virtual edges: SA_WEIGHTS from a preload to its matmul,
        XLU_STATE from a push to a pop."""
        if cn.compute_type == MATMUL_COMPUTE and SA_WEIGHTS not in reads:
            reads.append(SA_WEIGHTS)
        elif cn.compute_type == MATMUL_PRELOAD and SA_WEIGHTS not in writes:
            writes.append(SA_WEIGHTS)
        elif cn.compute_type == CROSS_LANE_COMPUTE:
            roles = set().union(*(_xlu_roles(op) for op in cn.operations))
            if "pop" in roles:
                reads.append(XLU_STATE)
            if "push" in roles:
                writes.append(XLU_STATE)
        return reads, writes

    def check_markers(self):
        pass
