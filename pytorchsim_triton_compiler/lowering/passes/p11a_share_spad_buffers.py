"""torchsim-share-spad-buffers (#227): scratchpad buffers of one type whose lives do not overlap take one
slot -- each alloc stamped SLOT_ATTR, the entry position of the alloc whose slot it takes. It runs once every
pass that tells buffers apart by identity is done; tritonpytorchsim-place-spad-buffers reads the stamps."""

from ...dialects import torchsim
from .lib_bank_layout import BANK_VIEW_OP, in_scratchpad, is_a_scratchpad_buffer, scratchpad_allocs
from .lib_ir import VIEW_OPS, bounds_lives_only, funcs, op_of
from .lib_pack import PACK_OP

NAME = "torchsim-share-spad-buffers"

MARKERS = ("memref.alloc",)

SHARED_NOT = "torchsim.spad_shared_not"

SLOT_ATTR = "torchsim.spad_slot"

_LOOPS = ("affine.for", "scf.for", "scf.while", "scf.parallel", "affine.parallel")

_CHASED = tuple(VIEW_OPS) + (BANK_VIEW_OP, PACK_OP, "memref.memory_space_cast")


def run(module, **_):
    """Every function's scratchpad allocs of one type packed by their lives onto the fewest slots."""
    from mlir.ir import IntegerAttr, IntegerType, StringAttr
    i64 = IntegerType.get_signless(64)
    for func in funcs(module):
        allocs = scratchpad_allocs(func)
        got = lives(func.operation)
        if isinstance(got, str):
            func.operation.attributes[SHARED_NOT] = StringAttr.get(got)
            continue
        for k, slot in _share(allocs, got).items():
            allocs[k].attributes[SLOT_ATTR] = IntegerAttr.get(i64, slot)
    return module


def only_freed(op):
    """Is every use of the buffer `op` allocates a free: a buffer no op reads, writes or views?"""
    return all(bounds_lives_only(op_of(u.owner)) for u in op.results[0].uses)


def buffer_of(v):
    """The alloc result `v` views, or None when its buffer cannot be told."""
    from mlir.ir import OpResult
    while isinstance(v, OpResult):
        if is_a_scratchpad_buffer(v):
            return v
        o = op_of(v.owner)
        if o.name not in _CHASED:
            return None
        v = o.operands[0]
    return None


def _number(func):
    """(every op under `func` in program order, its position, the position of the last op inside it)."""
    ops, index, last = [], {}, {}

    def rec(o):
        index[o] = len(ops)
        ops.append(o)
        for region in o.regions:
            for block in region.blocks:
                for inner in block.operations:
                    rec(inner.operation)
        last[o] = len(ops) - 1
    rec(func)
    return ops, index, last


def lives(func):
    """{buffer: (first, last) position} of what reads or writes each buffer through its views (a loop
    touching it is its life whole, an async transfer's buffer lives to its waits or the end), or why
    a scratchpad value's buffer cannot be told."""
    from mlir.ir import BoolAttr
    ops, index, last = _number(func)
    got = {}

    def touch(g, lo, hi):
        old = got.get(g)
        got[g] = (lo, hi) if old is None else (min(old[0], lo), max(old[1], hi))

    def life_at(o):
        lo, hi = index[o], index[o]
        p = o.parent
        while p is not None:
            p = op_of(p)
            if p == func:
                break
            if p.name in _LOOPS:
                lo, hi = min(lo, index[p]), max(hi, last[p])
            p = p.parent
        return lo, hi

    waits = [o for o in ops if o.name == torchsim.WaitOp.OPERATION_NAME]
    for o in ops[1:]:
        for v in list(o.operands) + list(o.results):
            if not in_scratchpad(v.type):
                continue
            g = buffer_of(v)
            if g is None:
                return f"{o.name} holds a scratchpad value whose buffer is not an alloc's view"
            if o.name not in _CHASED and not bounds_lives_only(o):
                touch(g, *life_at(o))
        if o.name == torchsim.TransferOp.OPERATION_NAME and "async" in o.attributes \
                and BoolAttr(o.attributes["async"]).value:
            g = buffer_of(o.operands[torchsim.operand_number(o, "sram")])
            sem = o.operands[torchsim.operand_number(o, "sem")]
            mine = [w for w in waits if w.operands[0] == sem]
            for w in mine:
                touch(g, *life_at(w))
            if not mine:
                touch(g, index[o], len(ops) - 1)
    return got


def _share(allocs, life):
    """{k: the position in `allocs` whose slot allocs[k] takes}: by type, in order of life, each buffer
    into the first slot whose last life ended before it starts; one no op touches into its type's first."""
    by_type, idle = {}, []
    for k, op in enumerate(allocs):
        if only_freed(op):
            continue
        v = op.results[0]
        if v in life:
            by_type.setdefault(v.type, []).append((life[v], -k))
        else:
            idle.append(k)
    slot_of, kept = {}, {}
    for ty, members in by_type.items():
        slots = []
        for (lo, hi), nk in sorted(members):
            j = next((j for j, (end, _) in enumerate(slots) if end < lo), None)
            if j is None:
                slots.append((hi, -nk))
                slot_of[-nk] = -nk
            else:
                slot_of[-nk] = slots[j][1]
                slots[j] = (hi, slots[j][1])
        kept[ty] = slots[0][1]
    for k in sorted(idle, reverse=True):
        ty = allocs[k].results[0].type
        slot_of[k] = kept.setdefault(ty, k)
    return slot_of
