"""lower_convert: every `pytorchsim.convert_layout` onto one move of the cross-lane unit (tpu::Xlu), or
into the DMA load it copies when that load's pieces cost less; one from a buffer every lane holds is each
lane's copy from its own bank (lower_in_bank). A convert no XLU op carries is refused -- the
model has no other carrier for it."""

import math

from ...dialect import pytorchsim
from ...passes.utils.bank_layout import EVERY_LANE, is_uniform, view_lanes
from ...passes.utils.layout_convert import kind_of, pairs_of, physical_groups
from ...passes.utils.ir_utils import funcs, name_of, op_of, walk
from .lower_in_bank import lower_in_bank


def run(module, lanes, **_opts):
    """Each convert, in the order it stands, onto its carrier."""
    from ...passes.utils.ir_utils import LANES_ATTR, lanes_of_module
    per_cycle = lanes_of_module(module) if LANES_ATTR in module.operation.attributes else None
    for f in funcs(module):
        for op in [op_of(o) for o in walk(f.operation)
                   if name_of(o) == pytorchsim.ConvertLayoutOp.OPERATION_NAME]:
            _rewrite(op, lanes, per_cycle)
    return module


def _rewrite(op, lanes, per_cycle):
    """A convert from a UNIFORM buffer, which every lane holds, is each lane's copy from its own bank
    when it writes each destination row it writes in every lane the destination holds there; any
    other convert is one move of the cross-lane unit."""
    from .xlu import XLU_PERMUTE, choose_mode, reaches_every_lane
    if lower_in_bank(op, lanes):
        return
    kind = kind_of(op)
    src, dst = op.operands[0], op.operands[1]
    s_shape, d_shape = list(src.type.shape), list(dst.type.shape)
    pairs = pairs_of(op, kind, lanes)
    src_uniform = is_uniform(src)
    modes = {"modes": (XLU_PERMUTE,)} if "into_window" in op.attributes else {}
    chosen = choose_mode(pairs, s_shape, d_shape, lanes, src_uniform, view_lanes(src), **modes)
    if chosen is None:
        raise NotImplementedError(f"a {kind} convert_layout no XLU move carries: {op}")
    if not src_uniform and _absorbed_into_its_load(
            op, _move_cost(chosen, src, dst, lanes, len(pairs), per_cycle)):
        return
    _build_move(op, chosen, src, dst, reaches_every_lane(pairs, lanes))
    op.operation.erase()


def _move_cost(chosen, src, dst, lanes, moved, per_cycle):
    """The unit's ideal cycles for the move, from the subtarget's vcix keys and the runs the permute
    emits (xlu.permute_runs), a permute's tile through the move stages (tpu_xlu_move_latency_cycles,
    else tpu_xlu_latency_cycles, as the unit reads them); another XLU op's elements over the lanes a
    cycle. The core's own instructions around the pushes and pops are not counted (#291). None when a
    fact is unsaid."""
    from mlir.ir import MemRefType
    from ...driver import machine
    from ...passes.utils.ir_utils import elem_bits
    from .xlu import XLU_PERMUTE, crossing_bits, crossings_of, permute_runs
    mode, plan = chosen
    if mode != XLU_PERMUTE:
        return None if per_cycle is None else -(-moved // per_cycle)
    t = machine.subtarget()
    if "tpu_xlu_elements_per_cycle" not in t.vcix or "tpu_xlu_latency_cycles" not in t.vcix:
        return None
    width = int(t.vcix["tpu_xlu_elements_per_cycle"])
    latency = int(t.vcix.get("tpu_xlu_move_latency_cycles", t.vcix["tpu_xlu_latency_cycles"]))
    elt = MemRefType(src.type).element_type
    crossings, _ = crossings_of(elem_bits(elt))
    if not crossings:
        return None
    runs = permute_runs(plan, min(view_lanes(dst) or lanes, lanes), t.vlen_bits // crossing_bits(elt))
    return latency + 1 + crossings * sum(-(-n // math.gcd(w, width)) + -(-n // w)
                                         for _, (*_, n), w, _ in runs)


def _absorbed_into_its_load(op, move_cost):
    """The copy done by the one plain DMA load that fills its source, restated into its destination
    (renumbered_per_axis), when that load's extra pieces at dma_load_latency_cycles each cost less
    than the move; True when it was. The source buffer and its load go."""
    from mlir.ir import InsertionPoint, Operation
    from ...driver import machine
    from ...passes.utils.descriptor import read_cut, runs_of
    from ...passes.utils.ir_utils import attrs_of, int_array, producer
    latency = machine.subtarget().dma_load_latency_cycles
    src, dst = op.operands[0], op.operands[1]
    groups = physical_groups(op)
    load = _sole_plain_load(op) if groups is not None else None
    if move_cost is None or latency is None or load is None:
        return False
    try:
        restated = pytorchsim.renumbered_per_axis(load, groups, list(src.type.shape))
    except NotImplementedError:
        return False
    attrs = {k: v for k, v in dict(attrs_of(load), **restated).items() if v is not None}
    operands = list(load.operands)
    operands[2] = dst
    with InsertionPoint(op):
        new = Operation.create(pytorchsim.TransferOp.OPERATION_NAME, operands=operands,
                               attributes=attrs)
    cut, shape = read_cut(new) or {}, list(dst.type.shape)
    stride = list(int_array(new.attributes["dram_stride"]))
    pieces = math.prod(len(runs_of(digits, shape[d], stride[d])) for d, digits in cut.items())
    if (pieces - 1) * latency >= move_cost:
        new.erase()
        return False
    alloc = producer(src)
    for u in [_operation(u.owner) for u in src.uses if name_of(u.owner) == "memref.dealloc"]:
        u.erase()
    load.operation.erase()
    op.operation.erase()
    if not list(alloc.results[0].uses):
        alloc.operation.erase()
    return True


_LOAD_ONLY = ("cut", "indirect_dim", "subtile_size", "dyn_stride_axes", "offset_stride", "piece_box")


def _sole_plain_load(op):
    """The DMA load that is the only writer of the copy's fresh source, before it in its block, with no
    cut, index tile or run-time stride (a mask renumbers or refuses); None for any other source."""
    from ...passes.utils.ir_utils import producer
    src = op.operands[0]
    if name_of(producer(src)) != "memref.alloc":
        return None
    users = [_operation(u.owner) for u in src.uses]
    loads = [u for u in users if name_of(u) == pytorchsim.TransferOp.OPERATION_NAME]
    rest = [u for u in users if name_of(u) not in (pytorchsim.TransferOp.OPERATION_NAME,
                                                     "memref.dealloc")]
    if len(loads) != 1 or len(rest) != 1 or rest[0] != op.operation:
        return None
    load = loads[0]
    if (load.operands[2] != src or not pytorchsim.is_load(load)
            or any(a in load.attributes for a in _LOAD_ONLY) or any(
                pytorchsim.operand_number(load, a) != pytorchsim.operand_number(load, b)
                for a, b in (("index", "clamp"), ("dyn_strides", "dyn_offset_stride")))
            or pytorchsim.operand_number(load, "dyn_offset_stride") != len(load.operands)
            or load.block != op.operation.block):
        return None
    order = [_operation(o) for o in op.operation.block.operations]
    at = lambda o: next(k for k, x in enumerate(order) if x == o)
    return load.opview if at(load) < at(op.operation) else None


def _operation(x):
    return getattr(x, "operation", x)


def _build_move(op, chosen, src, dst, every_lane):
    """The target's xlu_move, stamped with its mode and its rounds; EVERY_LANE when its pairs reach every
    lane."""
    from mlir.ir import ArrayAttr, DenseI64ArrayAttr, InsertionPoint, UnitAttr
    from .pytorchsim_tpu import XluMoveOp
    mode, plan = chosen
    with InsertionPoint(op):
        move = XluMoveOp(src, dst, mode, rounds=ArrayAttr.get([DenseI64ArrayAttr.get(list(r)) for r in plan])).operation
    if every_lane:
        move.attributes[EVERY_LANE] = UnitAttr.get()
    return move
