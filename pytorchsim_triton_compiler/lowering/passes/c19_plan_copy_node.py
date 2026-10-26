"""pytorchsim-plan-copy-node: a load of a whole kernel argument at the kernel's top (in no loop or branch)
into a buffer bufferize stamped PREFETCH_OPERAND (a matmul operand or a reduction input) is a copy node,
timed on its own before the kernel, whose TOG reads that buffer held. Runs on the legalized transfers, before
mark-runtime-attributes: a load another op writes beside or that shares its semaphore is not held, nor a
tensor contract/copy_node.resident_fits refuses. Stamps COPY_NODE on the module, changing nothing else."""
import math

from ...contract import copy_node
from ...dialects.pytorchsim import TransferOp, effects_on, is_load, moved_shape, view_root
from ...stage import NotAPrefetchCandidate, SpadOverflow
from .a06_bufferize import PREFETCH_OPERAND, PREFETCH_OPERAND_NOT
from .lib_bank_layout import in_scratchpad
from .lib_ir import funcs, is_view, op_of, walk
from .target_kind import target_kind
from . import pass_name

NAME = pass_name(__name__)

COPY_NODE = "pytorchsim.copy_node"
"""The module's {why: str, args: [i64]}: the kernel arguments a copy node loads whole, and why (or why none).
"""


class TooManyPrefetches(ValueError):
    """KernelSpec.graph_prefetch names more arguments than the graph holds
    (copy_node.graph_prefetches)."""


def run(module, target=None, arg_tensors=None, graph_prefetch=None, **_):
    """Stamp COPY_NODE on `module`: the whole-argument loads at its top into a prefetch operand that
    `graph_prefetch` (KernelSpec's; None: `target`'s kind's own order) holds, or why there is none;
    `arg_tensors` each kernel argument's (shape, element bytes)."""
    from mlir.ir import ArrayAttr, DictAttr, IntegerAttr, IntegerType, StringAttr
    if target is None:
        from ... import machine
        target = machine.target()
    args, why = plan(module, target_kind(target), target, arg_tensors, graph_prefetch)
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


def plan(module, kind, target, arg_tensors, graph_prefetch=None):
    """(the arguments whose one load is a copy node, why): of the prefetch operands' loads `whole_load`,
    `held_alone` and `_unfit` take (of `arg_tensors`, (shape, element bytes) each), each a
    copy_node.Candidate (its tensor's bytes, its uses' output elements bufferize stamped, its buffer's latest
    read, the argument as buffer id), those `choose` takes."""
    from mlir.ir import IntegerAttr
    whole, said, over = {}, [], {}
    for func in funcs(module):
        last = _latest_reads(func)
        transfers = [op for op in walk(func.operation) if isinstance(op.opview, TransferOp)]
        for op in transfers:
            arg = IntegerAttr(op.attributes["dram_arg"]).value
            stamp = _stamp(view_root(op.opview.sram))
            if not is_load(op) or arg < 0 or stamp is None:
                continue
            why = stamp or whole_load(func, op, transfers) or held_alone(func, op, transfers)
            unfit = None if why else _unfit(target, arg_tensors, arg)
            if unfit:
                over[arg] = arg_tensors[arg]
            if why or unfit:
                said.append((arg, why or unfit))
            else:
                root = view_root(op.opview.sram)
                whole[arg] = copy_node.Candidate(math.prod(arg_tensors[arg][0]) * arg_tensors[arg][1],
                                                 _use_elements(root), last.get(root, -1), arg)
    taken, unheld = choose(whole, said, over, kind, target, graph_prefetch)
    if not whole:
        none = "no matmul operand or reduction input is loaded from an argument"
        return [], "; ".join(f"arg {a}: {w}" for a, w in said) or none
    held = [f"arg {a}: loaded whole once at the kernel's top, the copy node" for a in taken]
    read = [f"arg {a}: loaded whole too, read as loaded ({unheld})" for a in kind.prefetch_order(whole)
            if a not in taken]
    return taken, "; ".join(held + read + [f"arg {a}: {w}" for a, w in said])


def choose(whole, said, over, kind, target, graph_prefetch):
    """(the arguments held, why the other candidates are not): the first copy_node.graph_prefetches(target) of
    kind.prefetch_order(`whole`) when `graph_prefetch` is None, else the arguments it names: no more than
    the graph holds (TooManyPrefetches), each a candidate, else SpadOverflow for a tensor `over` the bank
    and NotAPrefetchCandidate for any other reason `said` gives. An argument is a spec.args index."""
    n = copy_node.graph_prefetches(target)
    if graph_prefetch is None:
        return (kind.prefetch_order(whole)[:n] if whole else []), f"the graph holds {n}"
    asked = list(graph_prefetch)
    if len(asked) > copy_node.graph_prefetches(target):
        raise TooManyPrefetches(f"{NAME}: graph_prefetch {tuple(asked)} names {len(asked)} arguments; the "
                                f"graph holds at most {copy_node.graph_prefetches(target)}")
    for a in asked:
        if a in over:
            usage, budget = copy_node.resident_lane_bytes(target, *over[a]), copy_node.resident_budget(target)
            raise SpadOverflow("lower", f"{NAME}: graph_prefetch holds argument {a}, whose tensor takes "
                                        f"{usage} bytes a lane held whole, over {budget}", usage=usage,
                               budget=budget)
        if a not in whole:
            why = "; ".join(w for b, w in said if b == a) or "it loads no matmul operand or reduction input"
            msg = f"{NAME}: graph_prefetch holds argument {a}, which no copy node can hold here: {why}"
            raise NotAPrefetchCandidate("lower", msg)
    return asked, f"graph_prefetch is {tuple(asked)}"


def held_alone(func, op, transfers):
    """None when no op of `func` but the load `op` writes the buffer it fills and no other async transfer of
    `transfers` signals its semaphore; else why not: the TOG holds what the copy node loaded, unchanged."""
    root = view_root(op.opview.sram)
    for o in walk(func.operation):
        if o == op or is_view(o):
            continue
        for v in o.operands:
            if in_scratchpad(v.type) and view_root(v) == root and effects_on(o, v) & {"write", "unknown"}:
                return f"its buffer is written by {o.name} at {o.location} too"
    if _is_async(op) and any(t != op and _is_async(t) and _sem(t) == _sem(op) for t in transfers):
        return f"its semaphore {_sem(op)} is signalled by another transfer too"
    return None


def _unfit(target, arg_tensors, arg):
    """Why copy_node.resident_fits does not hold argument `arg`'s tensor (of `arg_tensors`) on `target`;
    None when it does."""
    if arg_tensors is None:
        raise ValueError(f"{NAME} needs the kernel's arg_tensors to size argument {arg}'s tensor")
    shape, elem_bytes = arg_tensors[arg]
    if copy_node.resident_fits(target, shape, elem_bytes):
        return None
    if copy_node.graph_prefetches(target) == 0:
        return "the graph holds no resident operand on this target (contract/copy_node.graph_prefetches)"
    return (f"a {list(shape)} tensor of {elem_bytes}-byte elements takes "
            f"{copy_node.resident_lane_bytes(target, shape, elem_bytes)} bytes a lane held whole, over "
            f"{copy_node.resident_budget(target)} (contract/copy_node.resident_fits)")


def _is_async(op):
    """Whether the transfer `op` is async, signalling its semaphore."""
    from mlir.ir import BoolAttr
    return "async" in op.attributes and BoolAttr(op.attributes["async"]).value


def _sem(op):
    """The semaphore an async transfer signals: its constant id, else the value."""
    sem = op.opview.sem
    got = _const(sem)
    return sem if got is None else got


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


def _stamp(root):
    """What bufferize stamped on the op allocating the buffer `root`: "" for PREFETCH_OPERAND,
    PREFETCH_OPERAND_NOT's why, None for neither (no dot or reduction reads it)."""
    from mlir.ir import OpResult, StringAttr
    if not isinstance(root, OpResult):
        return None
    attrs = root.owner.attributes
    if PREFETCH_OPERAND_NOT in attrs:
        return StringAttr(attrs[PREFETCH_OPERAND_NOT]).value
    return "" if PREFETCH_OPERAND in attrs else None


def _use_elements(root):
    """The elements the uses of the buffer `root` produce, as bufferize stamped them (PREFETCH_OPERAND)."""
    from mlir.ir import IntegerAttr
    return IntegerAttr(root.owner.attributes[PREFETCH_OPERAND]).value


def _latest_reads(func):
    """{buffer: the position, in `func`'s program order, of the last op that reads it through any view}."""
    got = {}
    for k, o in enumerate(walk(func.operation)):
        if is_view(o):
            continue
        for v in o.operands:
            if in_scratchpad(v.type) and "read" in effects_on(o, v):
                got[view_root(v)] = k
    return got


def _const(v):
    """The value of an arith.constant `v`, else None."""
    from mlir.dialects import arith
    from mlir.ir import BlockArgument, IntegerAttr
    if isinstance(v, BlockArgument) or not isinstance(op_of(v.owner).opview, arith.ConstantOp):
        return None
    return IntegerAttr(v.owner.attributes["value"]).value
