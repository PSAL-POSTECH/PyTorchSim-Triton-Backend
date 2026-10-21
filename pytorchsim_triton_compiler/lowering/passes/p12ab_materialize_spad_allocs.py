"""tritonpytorchsim-materialize-spad-allocs: each slot the placement gave (OFFSET) becomes a memref.global
in space 1, which p13_lower_to_llvm reserves in .spad in module order, the slots' own; every alloc on it a
get_global where it stood, its deallocs gone. Two buffers whose slots meet must not live at once."""

from ...dialects.torchsim import effects_on
from .lib_bank_layout import scratchpad_allocs, scratchpad_lives_left
from .lib_ir import bounds_lives_only, funcs, op_of
from .p11a_share_spad_buffers import SLOT_ATTR, lives, only_freed
from .p11b_place_spad_buffers import OFFSET, PEAK

NAME = "tritonpytorchsim-materialize-spad-allocs"


class SpadAllocRefused(Exception):
    """A scratchpad alloc this pass cannot give its slot's storage, or one left after it: a compile
    error, never a heap allocation."""


def run(module, *, vectorlane=None, **_):
    """Every function's placed scratchpad allocs as globals, one a slot, the module's slots in order."""
    from mlir.ir import IntegerAttr
    taken = _symbols(module)
    count = 0
    slots = []
    for func in funcs(module):
        allocs = scratchpad_allocs(func)
        if vectorlane is not None:
            _refuse_meeting_slots_alive_at_once(func, allocs, vectorlane)
        names = {}
        for op in reversed(allocs):
            while f"spad{count}" in taken:
                count += 1
            names[op] = f"spad{count}"
            count += 1
        by, idle = {}, []
        for op in allocs:
            if OFFSET in op.attributes:
                by.setdefault(IntegerAttr(op.attributes[OFFSET]).value, []).append(op)
            elif only_freed(op):
                idle.append(op)
            else:
                raise SpadAllocRefused(f"a scratchpad alloc no placement gave an offset "
                                       f"(tritonpytorchsim-place-spad-buffers): {op}")
        for at, members in by.items():
            m = members[0]
            keeper = allocs[IntegerAttr(m.attributes[SLOT_ATTR]).value] if SLOT_ATTR in m.attributes else m
            ty = keeper.results[0].type
            if any(op.results[0].type != ty for op in members):
                raise SpadAllocRefused(f"buffers of more than one type share the slot at {at}, which "
                                       f"one global of one type cannot hold: {members}")
            name = names[keeper]
            taken.add(name)
            slots.append((at, name, ty))
            for op in members:
                _materialize(op, name)
        for op in idle:
            _materialize(op, None)
        if PEAK in func.operation.attributes:
            del func.operation.attributes[PEAK]
    _mint_the_slots(module, sorted(slots, key=lambda s: s[0]))
    left = scratchpad_lives_left(module)
    if left:
        raise SpadAllocRefused(f"a scratchpad buffer is still allocated or freed after {NAME}: {left[0]}")
    return module


def _symbols(module):
    from mlir.ir import StringAttr
    return {StringAttr(o.operation.attributes["sym_name"]).value
            for o in module.operation.regions[0].blocks[0].operations
            if "sym_name" in o.operation.attributes}


def _mint_the_slots(module, slots):
    from mlir.ir import InsertionPoint, Operation, StringAttr, TypeAttr
    top = module.operation.regions[0].blocks[0]
    for _, name, ty in reversed(slots):
        with InsertionPoint.at_block_begin(top):
            Operation.create("memref.global", results=[], operands=[],
                             attributes={"sym_name": StringAttr.get(name), "type": TypeAttr.get(ty)})


def _materialize(op, name):
    """`op`'s buffer as a get_global of its slot's global `name`, its frees erased; nothing when only
    freed (`name` None)."""
    from mlir.dialects import memref
    from mlir.ir import InsertionPoint
    v = op.results[0]
    ends = [op_of(u.owner) for u in v.uses]
    ends = [e for e in ends if "free" in effects_on(e, v)]
    if any(not bounds_lives_only(e) for e in ends):
        raise SpadAllocRefused(f"an op frees a scratchpad buffer and does more, so it cannot simply "
                               f"go: {ends[0]}")
    for e in ends:
        e.erase()
    if name is not None:
        with InsertionPoint(op):
            got = memref.GetGlobalOp(v.type, name).result
        v.replace_all_uses_with(got)
    op_of(op).erase()


def _refuse_meeting_slots_alive_at_once(func, allocs, vectorlane):
    """Two allocs whose bytes meet in the arena must not both be read or written at once (lives)."""
    from mlir.ir import IntegerAttr
    from .lib_bank_layout import slot_bytes
    spans = sorted(((IntegerAttr(op.attributes[OFFSET]).value, k) for k, op in enumerate(allocs)
                    if OFFSET in op.attributes))
    spans = [(lo, lo + slot_bytes(allocs[k].results[0].type, vectorlane), allocs[k]) for lo, k in spans]
    meets = [(a, b) for i, a in enumerate(spans) for b in spans[i + 1:] if b[0] < a[1]]
    if not meets:
        return
    life = lives(func.operation)
    if isinstance(life, str):
        raise SpadAllocRefused(f"two scratchpad buffers share bytes and their lives cannot be told "
                               f"({life}): {meets[0][0][2]}")
    for (_, _, a), (_, _, b) in meets:
        la, lb = life.get(a.results[0]), life.get(b.results[0])
        if la is not None and lb is not None and not (la[1] < lb[0] or lb[1] < la[0]):
            raise SpadAllocRefused(f"two scratchpad buffers share bytes and live at once "
                                   f"({la} and {lb}): {a} and {b}")
