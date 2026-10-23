"""torchsim-share-spad-buffers: scratchpad buffers of one type whose lives do not overlap share one
global. It runs once the passes that tell buffers apart by their global (the matmul and its epilogue)
are done, and before the DMA lowering, while transfers and their waits are still ops.

A buffer lives from its function's first op that touches it to its last; a loop that touches it lives
whole (its next trip may touch it again); an async transfer's buffer lives to its waits, or to the
function's end when it has none; one no op reads or writes has no life and takes any global of its
type. A static loop Triton unrolls (attention's query blocks) leaves one set of buffers a trip, each
dead once its trip has stored, and the next trip's set takes their place. Each kept global is reserved
once (p13_lower_to_llvm.reserve_spad_storage), so a kernel's scratchpad is what is live at once, not
every buffer it ever made.

A buffer an op allocates keeps its own storage, and a dealloc is no touch. A function where a
scratchpad value's buffer cannot be told (a memref carried through a loop or a region, or a function
argument) shares nothing, and says why on SHARED_NOT."""

from ...dialects import torchsim
from .lib_bank_layout import BANK_VIEW_OP, in_scratchpad, is_a_scratchpad_buffer
from .lib_ir import VIEW_OPS, bounds_lives_only, funcs
from .lib_pack import PACK_OP

NAME = "torchsim-share-spad-buffers"

SHARED_NOT = "torchsim.spad_shared_not"

_LOOPS = ("affine.for", "scf.for", "scf.while", "scf.parallel", "affine.parallel")


def run(module, **_):
    """Every function's scratchpad buffers of one type packed by their lives onto the fewest globals."""
    from mlir.ir import StringAttr
    top = module.operation.regions[0].blocks[0]
    for func in funcs(module):
        why = _share(func.operation, top)
        if why:
            func.operation.attributes[SHARED_NOT] = StringAttr.get(why)
    return module


def _global_of(v):
    """The scratchpad global `v` views (its name), the value an op allocates it as when it views
    one of those, or None when its buffer cannot be told."""
    from mlir.ir import FlatSymbolRefAttr, OpResult
    while True:
        if not isinstance(v, OpResult):
            return None
        o = v.owner
        o = o.operation if hasattr(o, "operation") else o
        if o.name == "memref.get_global":
            return FlatSymbolRefAttr(o.attributes["name"]).value
        if is_a_scratchpad_buffer(v):
            return v
        if o.name not in VIEW_OPS and o.name not in (BANK_VIEW_OP, PACK_OP, "memref.memory_space_cast"):
            return None
        v = o.operands[0]


_VIEWS = tuple(VIEW_OPS) + (BANK_VIEW_OP, PACK_OP, "memref.memory_space_cast", "memref.get_global")


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


def _share(func, top):
    """Fold `func`'s buffers (_global_of) onto shared globals; why not, or None. A view makes no use:
    what touches the buffer is what reads or writes through it, or the wait of a transfer into it."""
    from mlir.ir import BoolAttr, FlatSymbolRefAttr, StringAttr, TypeAttr
    ops, index, last = _number(func)
    lives = {}

    def touch(g, lo, hi):
        old = lives.get(g)
        lives[g] = (lo, hi) if old is None else (min(old[0], lo), max(old[1], hi))

    def life_at(o):
        lo, hi = index[o], index[o]
        p = o.parent
        while p is not None:
            p = p.operation if hasattr(p, "operation") else p
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
            g = _global_of(v)
            if g is None:
                return f"{o.name} holds a scratchpad value whose buffer is not a global's view"
            if o.name not in _VIEWS and not bounds_lives_only(o):
                touch(g, *life_at(o))
        if o.name == torchsim.TransferOp.OPERATION_NAME and "async" in o.attributes \
                and BoolAttr(o.attributes["async"]).value:
            g = _global_of(o.operands[torchsim.operand_number(o, "sram")])
            sem = o.operands[torchsim.operand_number(o, "sem")]
            mine = [w for w in waits if w.operands[0] == sem]
            for w in mine:
                touch(g, *life_at(w))
            if not mine:
                touch(g, index[o], len(ops) - 1)
    globals_ = {StringAttr(o.operation.attributes["sym_name"]).value: o.operation
                for o in top.operations if o.operation.name == "memref.global"}
    by_type = {}
    for g, life in lives.items():
        if isinstance(g, str) and g in globals_:
            by_type.setdefault(TypeAttr(globals_[g].attributes["type"]).value, []).append((life, g))
    rename, kept = {}, {}
    for ty, members in by_type.items():
        slots = []      # (end of its last life, global it keeps)
        for (lo, hi), g in sorted(members):
            k = next((k for k, (end, _) in enumerate(slots) if end < lo), None)
            if k is None:
                slots.append((hi, g))
            else:
                rename[g] = slots[k][1]
                slots[k] = (hi, slots[k][1])
        kept[ty] = slots[0][1]
    # A global no op reads or writes (a dot's C whose pops feed the epilogue alone) has no life:
    # any global of its type takes it.
    named = {FlatSymbolRefAttr(o.attributes["name"]).value for o in ops if o.name == "memref.get_global"}
    for g in sorted(named - {g for g in lives if isinstance(g, str)}):
        if g in globals_:
            ty = TypeAttr(globals_[g].attributes["type"]).value
            if ty in kept:
                rename[g] = kept[ty]
            else:
                kept[ty] = g
    if not rename:
        return None
    for o in ops:
        if o.name == "memref.get_global":
            g = FlatSymbolRefAttr(o.attributes["name"]).value
            if g in rename:
                o.attributes["name"] = FlatSymbolRefAttr.get(rename[g])
    for g in rename:
        globals_[g].erase()
    return None
