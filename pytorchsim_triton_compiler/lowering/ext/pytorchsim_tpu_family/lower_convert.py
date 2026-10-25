"""lower_convert: every `pytorchsim.convert_layout` onto one move of the cross-lane unit (tpu::Xlu), or
into the DMA load it copies when that load's pieces cost less (rvv_generic.reread_from_its_load); one from
a buffer every lane holds is each lane's copy from its own bank (rvv_generic.lower_convert). A convert no XLU op carries is refused -- the
model has no other carrier for it."""

import math

from ....dialects import pytorchsim
from ...passes.lib_bank_layout import EVERY_LANE, is_uniform, view_lanes
from ...passes.lib_convert import kind_of, pairs_of
from ...passes.lib_ir import funcs, name_of, op_of, walk
from ...passes.transform import LOWER_CONVERT
from ..rvv_generic.lower_convert import lower_in_bank, reread_from_its_load

NAME = "pytorchsim-lower-convert"

SLOT = LOWER_CONVERT.name

FORBIDS = LOWER_CONVERT.lowers


def run(module, lanes, **_opts):
    """Each convert, in the order it stands, onto its carrier."""
    from ...passes.lib_ir import LANES_ATTR, lanes_of_module
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
    from .lib_xlu import XLU_PERMUTE, choose_xlu_op, reaches_every_lane
    if lower_in_bank(op, lanes):
        return
    kind = kind_of(op)
    src, dst = op.operands[0], op.operands[1]
    s_shape, d_shape = list(src.type.shape), list(dst.type.shape)
    pairs = pairs_of(op, kind, lanes)
    src_uniform = is_uniform(src)
    kinds = (XLU_PERMUTE,) if "into_window" in op.attributes else None
    chosen = choose_xlu_op(pairs, s_shape, d_shape, lanes, src_uniform, view_lanes(src), kinds=kinds)
    if chosen is None:
        raise NotImplementedError(f"a {kind} convert_layout no XLU op carries: {op}")
    if not src_uniform and _cheaper_from_its_load(
            op, _move_cost(chosen, src, dst, lanes, len(pairs), per_cycle)):
        return
    _build_move(op, chosen, src, dst, reaches_every_lane(pairs, lanes))
    op.operation.erase()


def _move_cost(chosen, src, dst, lanes, moved, per_cycle):
    """The unit's ideal cycles for the move, from the target's vcix keys and the runs the permute
    emits (lib_xlu.permute_runs), a permute's tile through the move stages (tpu_xlu_move_latency_cycles,
    else tpu_xlu_latency_cycles, as the unit reads them); another XLU op's elements over the lanes a
    cycle. The core's own instructions around the pushes and pops are not counted (#291). None when a
    fact is unsaid."""
    from mlir.ir import MemRefType
    from .... import machine
    from ...passes.lib_ir import elem_bits
    from .lib_xlu import XLU_PERMUTE, crossing_bits, crossings_of, emitter_of, permute_runs
    xlu_op, plan = chosen
    if emitter_of(xlu_op) != XLU_PERMUTE:
        return None if per_cycle is None else -(-moved // per_cycle)
    t = machine.target()
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


def _cheaper_from_its_load(op, move_cost):
    """The copy done by the DMA load that fills its source (rvv_generic.reread_from_its_load), when that
    load's extra pieces at dma_load_latency_cycles each cost less than the move; True when it was."""
    from .... import machine
    latency = machine.target().dma_load_latency_cycles
    if move_cost is None or latency is None:
        return False
    return reread_from_its_load(op, lambda pieces: (pieces - 1) * latency < move_cost)


def _build_move(op, chosen, src, dst, every_lane):
    """The pack's xlu_move, stamped with its XLU op, its rounds and, for whoever counts moves without
    knowing this unit, the kind of move that op is; EVERY_LANE when its pairs reach every lane."""
    from mlir.ir import (ArrayAttr, DenseI64ArrayAttr, InsertionPoint, IntegerAttr, IntegerType,
                         Operation, StringAttr, UnitAttr)
    from .lib_xlu import emitter_of
    from .lib_xlu_move import KIND_ATTR, ROUNDS_ATTR, XLU_MOVE_OP, XLU_OP_ATTR
    xlu_op, plan = chosen
    with InsertionPoint(op):
        move = Operation.create(XLU_MOVE_OP, operands=[src, dst], attributes={
            XLU_OP_ATTR: IntegerAttr.get(IntegerType.get_signless(64), xlu_op),
            KIND_ATTR: StringAttr.get(emitter_of(xlu_op)),
            ROUNDS_ATTR: ArrayAttr.get([DenseI64ArrayAttr.get(list(r)) for r in plan])})
    if every_lane:
        move.attributes[EVERY_LANE] = UnitAttr.get()
    return move
