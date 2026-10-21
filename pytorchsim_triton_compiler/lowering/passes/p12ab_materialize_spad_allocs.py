"""tritonpytorchsim-materialize-spad-allocs: the module's one scratchpad arena, a memref.global of i8 in space
1 as long as the functions' PEAK and named by the module's SPAD_ARENA, and each placed alloc a memref.view
of it at its OFFSET; its deallocs go. Two buffers whose slots meet must not live at once."""

from ...dialects.torchsim import effects_on
from .lib_bank_layout import SPAD_ARENA, SPAD_SPACE, scratchpad_allocs, scratchpad_lives_left
from .lib_ir import bounds_lives_only, funcs, op_of
from .p11a_share_spad_buffers import lives, only_freed
from .p11b_place_spad_buffers import OFFSET, PEAK

NAME = "tritonpytorchsim-materialize-spad-allocs"

ARENA_NAME = "spad_arena"


class SpadAllocRefused(Exception):
    """A scratchpad alloc this pass cannot put in the arena, or one left after it: a compile error,
    never a heap allocation."""


def run(module, *, vectorlane=None, **_):
    """Every function's placed scratchpad allocs as views of the arena."""
    from mlir.ir import IntegerAttr
    placed = [(f, scratchpad_allocs(f)) for f in funcs(module)]
    peak = max([IntegerAttr(f.operation.attributes[PEAK]).value for f, allocs in placed
                if allocs and PEAK in f.operation.attributes] or [0])
    arena = _mint_the_arena(module, peak) if peak else None
    for func, allocs in placed:
        if vectorlane is not None:
            _refuse_meeting_slots_alive_at_once(func, allocs, vectorlane)
        base = None
        for op in allocs:
            if OFFSET not in op.attributes and not only_freed(op):
                raise SpadAllocRefused(f"a scratchpad alloc no placement gave an offset "
                                       f"(tritonpytorchsim-place-spad-buffers): {op}")
            if OFFSET in op.attributes and base is None:
                if arena is None or PEAK not in func.operation.attributes:
                    raise SpadAllocRefused(f"a placed scratchpad alloc in a function the placement "
                                           f"gave no {PEAK}: {op}")
                base = _arena_in(func, arena)
            _materialize(op, base)
        if PEAK in func.operation.attributes:
            del func.operation.attributes[PEAK]
    left = scratchpad_lives_left(module)
    if left:
        raise SpadAllocRefused(f"a scratchpad buffer is still allocated or freed after {NAME}: {left[0]}")
    return module


def _mint_the_arena(module, peak):
    from mlir.ir import (FlatSymbolRefAttr, InsertionPoint, IntegerAttr, IntegerType, MemRefType,
                         Operation, StringAttr, TypeAttr)
    top = module.operation.regions[0].blocks[0]
    taken = {StringAttr(o.operation.attributes["sym_name"]).value for o in top.operations
             if "sym_name" in o.operation.attributes}
    name, n = ARENA_NAME, 0
    while name in taken:
        n += 1
        name = f"{ARENA_NAME}{n}"
    ty = MemRefType.get([peak], IntegerType.get_signless(8),
                        memory_space=IntegerAttr.get(IntegerType.get_signless(64), SPAD_SPACE))
    with InsertionPoint.at_block_begin(top):
        Operation.create("memref.global", results=[], operands=[],
                         attributes={"sym_name": StringAttr.get(name), "type": TypeAttr.get(ty)})
    module.operation.attributes[SPAD_ARENA] = FlatSymbolRefAttr.get(name)
    return name, ty


def _arena_in(func, arena):
    from mlir.dialects import memref
    from mlir.ir import InsertionPoint
    name, ty = arena
    with InsertionPoint.at_block_begin(func.operation.regions[0].blocks[0]):
        return memref.GetGlobalOp(ty, name).result


def _materialize(op, base):
    """`op`'s buffer as a view of the arena at its offset, its frees erased; nothing when only freed.
    The view's descriptor carries no allocated pointer of its own, so no free may read it."""
    from mlir.dialects import arith, memref
    from mlir.ir import IndexType, InsertionPoint, IntegerAttr
    v = op.results[0]
    ends = [op_of(u.owner) for u in v.uses]
    ends = [e for e in ends if "free" in effects_on(e, v)]
    if any(not bounds_lives_only(e) for e in ends):
        raise SpadAllocRefused(f"an op frees a scratchpad buffer and does more, so it cannot simply "
                               f"go: {ends[0]}")
    for e in ends:
        e.erase()
    if OFFSET in op.attributes:
        at = IntegerAttr(op.attributes[OFFSET]).value
        with InsertionPoint(op):
            shift = arith.ConstantOp(IndexType.get(), at).result
            got = memref.ViewOp(v.type, base, shift, []).result
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
