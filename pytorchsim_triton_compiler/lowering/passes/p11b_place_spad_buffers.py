"""tritonpytorchsim-place-spad-buffers: each scratchpad alloc stamped OFFSET, its slot's per-lane byte offset
in the one arena, and each function PEAK, the bytes its slots reach. Slots are laid end to end in entry
order, each slot_bytes long from a vector-register boundary; one shared only as SLOT_ATTR says, none if only freed."""

from .lib_bank_layout import scratchpad_allocs, slot_bytes
from .lib_ir import funcs, op_of
from .p11a_share_spad_buffers import SLOT_ATTR, only_freed

NAME = "tritonpytorchsim-place-spad-buffers"

MARKERS = ("memref.alloc",)

OFFSET = "torchsim.spad_offset"

PEAK = "torchsim.spad_peak"


class SpadPlacementRefused(Exception):
    """A scratchpad buffer this pass cannot give one static slot: a compile error, never a guess."""


def run(module, *, target, **_):
    """Every function's slots, the module's last function first (the order its buffers' storage had),
    each after the last one's so no two functions' slots meet."""
    from mlir.ir import IntegerAttr, IntegerType
    i64 = IntegerType.get_signless(64)
    align = target.vlen_bits // 8
    reach = 0
    for func in reversed(funcs(module)):
        for at, members in _slots(func, target.vector_lanes):
            start = -(-reach // align) * align
            for op in members:
                op.attributes[OFFSET] = IntegerAttr.get(i64, start)
            reach = start + at
        func.operation.attributes[PEAK] = IntegerAttr.get(i64, reach)
    return module


def _slots(func, vectorlane):
    """(bytes, allocs) of each slot of `func`, in the entry order of the alloc whose slot it is."""
    from mlir.ir import IntegerAttr
    entry = func.operation.regions[0].blocks[0]
    allocs = scratchpad_allocs(func)
    by = {}
    for k, op in enumerate(allocs):
        if op_of(op).block != entry or len(op.results) != 1 or len(op.operands):
            raise SpadPlacementRefused(f"a scratchpad alloc that is not one static buffer at its "
                                       f"function's entry has no one slot: {op}")
        if only_freed(op):
            continue
        key = IntegerAttr(op.attributes[SLOT_ATTR]).value if SLOT_ATTR in op.attributes else k
        if not 0 <= key < len(allocs) or only_freed(allocs[key]) or SLOT_ATTR in allocs[key].attributes \
                and IntegerAttr(allocs[key].attributes[SLOT_ATTR]).value != key:
            raise SpadPlacementRefused(f"{SLOT_ATTR} = {key} names no alloc that keeps its own slot: {op}")
        by.setdefault(key, []).append(op_of(op))
    return [(max(slot_bytes(m.results[0].type, vectorlane) for m in by[key]), by[key])
            for key in sorted(by)]
