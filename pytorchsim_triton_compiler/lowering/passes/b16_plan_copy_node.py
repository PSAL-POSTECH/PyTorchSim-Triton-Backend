"""pytorchsim-plan-copy-node: a load of a whole kernel argument at the kernel's top (in no loop or branch)
into a buffer a pytorchsim.dot reads is a copy node, timed on its own before the kernel, whose TOG reads that
buffer held. Stamps COPY_NODE on the module; the program is not changed."""
import math

from ...dialects.pytorchsim import TransferOp, is_load, moved_shape, view_root
from .lib_ir import DOT_OP, funcs, walk
from . import pass_name

NAME = pass_name(__name__)
MARKERS = (DOT_OP,)

COPY_NODE = "pytorchsim.copy_node"
"""The module's {why: str, args: [i64]}: the kernel arguments a copy node loads whole, and why (or why none).
"""


def run(module, **_):
    """Stamp COPY_NODE on `module`: every whole-argument load at its top that feeds a dot, or why there is
    none."""
    from mlir.ir import ArrayAttr, DictAttr, IntegerAttr, IntegerType, StringAttr
    args, why = plan(module)
    i64 = IntegerType.get_signless(64)
    module.operation.attributes[COPY_NODE] = DictAttr.get({
        "why": StringAttr.get(why), "args": ArrayAttr.get([IntegerAttr.get(i64, a) for a in args])})
    return module


def held_args(path):
    """The arguments COPY_NODE of the module at `path` names, none when it states none."""
    from mlir.ir import ArrayAttr, Context, DictAttr, IntegerAttr, Module
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, open(path) as f:
        attrs = Module.parse(f.read()).operation.attributes
        return ([IntegerAttr(a).value for a in ArrayAttr(DictAttr(attrs[COPY_NODE])["args"])]
                if COPY_NODE in attrs else [])


def plan(module):
    """(the argument whose one load is the copy node, as a list, why): of the dot operands' loads `whole_load`
    takes, the one copy_node.order puts first (XLA prefetches one operand, never two)."""
    from mlir.ir import IntegerAttr
    from ...contract import copy_node
    whole, said = {}, []
    for func in funcs(module):
        transfers = [op for op in walk(func.operation) if op.name == TransferOp.OPERATION_NAME]
        dotted = [view_root(op.operands[k]) for op in walk(func.operation) if op.name == DOT_OP for k in (0, 1)]
        for op in transfers:
            arg = IntegerAttr(op.attributes["dram_arg"]).value
            if not is_load(op) or arg < 0 or view_root(op.opview.sram) not in dotted:
                continue
            why = whole_load(func, op, transfers)
            if why is None:
                whole[arg] = _moved_count(op)
            else:
                said.append(f"arg {arg}: {why}")
    if not whole:
        return [], "; ".join(said) or "no dot operand is loaded from an argument"
    first, *rest = copy_node.order(whole)
    said = [f"arg {first}: loaded whole once at the kernel's top, the copy node"] + \
           [f"arg {a}: loaded whole too, read as loaded (XLA prefetches one operand)" for a in rest] + said
    return [first], "; ".join(said)


def whole_load(func, op, transfers):
    """None when the load `op` moves its whole argument, once, in `func`'s entry block, and no other of
    `transfers` moves that argument; else why not."""
    from mlir.ir import IntegerAttr, MemRefType
    arg = IntegerAttr(op.attributes["dram_arg"]).value
    if op.block != func.regions[0].blocks[0]:
        return "loaded inside a loop or a branch, not once"
    if any(t != op and IntegerAttr(t.attributes["dram_arg"]).value == arg for t in transfers):
        return "moved by more than one transfer"
    dram = MemRefType(op.opview.dram.type)
    moved = _moved_count(op)
    if not dram.has_static_shape or moved != math.prod(dram.shape) or _const(op.opview.dram_idx) != 0:
        return f"the load moves {moved} elements of {dram} from offset {_const(op.opview.dram_idx)}, not all of it"
    return None


def _moved_count(op):
    """The elements `op` moves: its moved shape, each masked axis held to its constant clamp; None when a
    clamp is not a constant."""
    from mlir.ir import IntegerAttr
    tx = op.opview
    shape = moved_shape(op)
    if tx.masked_axes is None:
        return math.prod(shape)
    clamp = [_const(v) for v in tx.clamp]
    axes = [IntegerAttr(a).value for a in tx.masked_axes]
    if None in clamp or len(clamp) != 2 * len(axes):
        return None
    for i, a in enumerate(axes):
        shape[a] = max(min(clamp[2 * i + 1], shape[a]) - max(clamp[2 * i], 0), 0)
    return math.prod(shape)


def _const(v):
    """The value of an arith.constant `v`, else None."""
    from mlir.ir import BlockArgument, IntegerAttr
    if isinstance(v, BlockArgument) or v.owner.name != "arith.constant":
        return None
    return IntegerAttr(v.owner.attributes["value"]).value
