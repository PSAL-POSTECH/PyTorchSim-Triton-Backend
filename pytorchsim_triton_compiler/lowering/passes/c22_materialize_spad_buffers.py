"""pytorchsim-materialize-spad-buffers: each scratchpad alloc becomes a memref.global in space 1, which c24
reserves in .spad, read by a get_global where the alloc stood (its dealloc goes; the globals keep the module
order a06 gave them, the last minted first); then buffers of one type whose lives do not overlap share one."""

from ...dialects import pytorchsim
from ...dialects.pytorchsim import effects_on, is_view_like
from .lib_bank_layout import SPAD_SPACE, in_scratchpad, is_a_scratchpad_buffer, scratchpad_lives_left
from .lib_ir import bounds_lives_only, funcs, is_get_global, is_global, op_of, walk
from . import pass_name, verify_spad_containment

NAME = pass_name(__name__)

SHARED_NOT = "pytorchsim.spad_shared_not"

_LOOPS = ("affine.for", "scf.for", "scf.while", "scf.parallel", "affine.parallel")


class SpadAllocRefused(Exception):
    """A scratchpad alloc this pass cannot make a global, or one left after it: a compile error, never
    a heap allocation."""


def run(module, **opts):
    """Every access first checked inside its buffer's slot (verify_spad_containment, before anything is
    placed), then every function's scratchpad allocs as globals, the entry's last first, each at the
    module's top, then shared by their lives (share_globals)."""
    verify_spad_containment.run(module, **opts)
    taken = _symbols(module)
    count = 0
    for func in funcs(module):
        for op in reversed(_allocs_of(func)):
            count = _materialize(module, op, taken, count)
    left = scratchpad_lives_left(module)
    if left:
        raise SpadAllocRefused(f"a scratchpad buffer is still allocated or freed after {NAME}: {left[0]}")
    share_globals(module)
    return module


def share_globals(module):
    """Every function's scratchpad buffers of one type packed by their lives onto the fewest globals; a
    function where a buffer cannot be told shares nothing and says why on SHARED_NOT."""
    from mlir.ir import StringAttr
    top = module.operation.regions[0].blocks[0]
    for func in funcs(module):
        why = _share(func.operation, top)
        if why:
            func.operation.attributes[SHARED_NOT] = StringAttr.get(why)


def _symbols(module):
    from mlir.ir import StringAttr
    return {StringAttr(o.operation.attributes["sym_name"]).value
            for o in module.operation.regions[0].blocks[0].operations
            if "sym_name" in o.operation.attributes}


def _allocs_of(func):
    """`func`'s scratchpad allocs (each op stating it allocates one) in the entry block's order; one
    anywhere else, or of a size known only at run time, is refused."""
    entry = func.operation.regions[0].blocks[0]
    got = []
    for op in walk(func.operation):
        if not any(in_scratchpad(r.type) and "allocate" in effects_on(op, r) for r in op.results):
            continue
        if op.operation.block != entry or len(op.results) != 1 or len(op.operands):
            raise SpadAllocRefused(f"a scratchpad alloc that is not one static buffer at its "
                                   f"function's entry has no one global to stand for it: {op}")
        got.append(op)
    return got


def _materialize(module, op, taken, count):
    from mlir.ir import InsertionPoint, StringAttr, TypeAttr, Operation
    from mlir.dialects import memref
    v = op.results[0]
    ends = [op_of(u.owner) for u in v.uses]
    ends = [e for e in ends if "free" in effects_on(e, v)]
    if any(not bounds_lives_only(e) for e in ends):
        raise SpadAllocRefused(f"an op frees a scratchpad buffer and does more, so it cannot simply "
                               f"go: {ends[0]}")
    while f"spad{count}" in taken:
        count += 1
    name = f"spad{count}"
    taken.add(name)
    with InsertionPoint.at_block_begin(module.operation.regions[0].blocks[0]):
        Operation.create("memref.global", results=[], operands=[],
                         attributes={"sym_name": StringAttr.get(name), "type": TypeAttr.get(v.type)})
    with InsertionPoint(op):
        got = memref.GetGlobalOp(v.type, name).result
    for e in ends:
        e.erase()
    v.replace_all_uses_with(got)
    op.erase()
    return count + 1


def _is_spad(ty):
    from mlir.ir import IntegerAttr, MemRefType
    try:
        mt = MemRefType(ty)
    except (TypeError, ValueError):
        return False
    return mt.memory_space is not None and IntegerAttr(mt.memory_space).value == SPAD_SPACE


def _global_of(v):
    """The scratchpad global `v` views (its name), through every ViewLikeOpInterface op, the value an
    op allocates it as when it views one of those, or None when its buffer cannot be told."""
    from mlir.ir import FlatSymbolRefAttr, OpResult
    while True:
        if not isinstance(v, OpResult):
            return None
        o = v.owner
        o = o.operation if hasattr(o, "operation") else o
        if is_get_global(o):
            return FlatSymbolRefAttr(o.attributes["name"]).value
        if is_a_scratchpad_buffer(v):
            return v
        if not is_view_like(o):
            return None
        v = o.operands[0]


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
    what touches the buffer is what reads or writes through it, or the wait of a transfer into it. A global
    no op reads or writes (a dot's C whose pops feed the epilogue alone) has no life: any global of its
    type takes it."""
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

    waits = [o for o in ops if o.name == pytorchsim.WaitOp.OPERATION_NAME]
    for o in ops[1:]:
        for v in list(o.operands) + list(o.results):
            if not _is_spad(v.type):
                continue
            g = _global_of(v)
            if g is None:
                return f"{o.name} holds a scratchpad value whose buffer is not a global's view"
            if not is_view_like(o) and not is_get_global(o) and not bounds_lives_only(o):
                touch(g, *life_at(o))
        if o.name == pytorchsim.TransferOp.OPERATION_NAME and "async" in o.attributes \
                and BoolAttr(o.attributes["async"]).value:
            g = _global_of(o.operands[pytorchsim.operand_number(o, "sram")])
            sem = o.operands[pytorchsim.operand_number(o, "sem")]
            mine = [w for w in waits if w.operands[0] == sem]
            for w in mine:
                touch(g, *life_at(w))
            if not mine:
                touch(g, index[o], len(ops) - 1)
    globals_ = {StringAttr(o.operation.attributes["sym_name"]).value: o.operation
                for o in top.operations if is_global(o)}
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
    named = {FlatSymbolRefAttr(o.attributes["name"]).value for o in ops if is_get_global(o)}
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
        if is_get_global(o):
            g = FlatSymbolRefAttr(o.attributes["name"]).value
            if g in rename:
                o.attributes["name"] = FlatSymbolRefAttr.get(rename[g])
    for g in rename:
        globals_[g].erase()
    return None
