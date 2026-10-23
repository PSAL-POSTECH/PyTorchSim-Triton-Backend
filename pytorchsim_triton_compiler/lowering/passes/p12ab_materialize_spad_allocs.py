"""tritonpytorchsim-materialize-spad-allocs: each scratchpad alloc becomes a memref.global in space 1,
which p13_lower_to_llvm reserves in .spad, read by a get_global where the alloc stood; its dealloc goes.
The globals keep the module order p04 once gave them, the last minted first."""

from ...dialects.torchsim import effects_on
from .lib_bank_layout import in_scratchpad, scratchpad_lives_left
from .lib_ir import bounds_lives_only, funcs, op_of, symbols, walk

NAME = "tritonpytorchsim-materialize-spad-allocs"


class SpadAllocRefused(Exception):
    """A scratchpad alloc this pass cannot make a global, or one left after it: a compile error, never
    a heap allocation."""


def run(module, **_):
    """Every function's scratchpad allocs as globals, the entry's last first, each at the module's top."""
    taken = symbols(module)
    count = 0
    for func in funcs(module):
        for op in reversed(_allocs_of(func)):
            count = _materialize(module, op, taken, count)
    left = scratchpad_lives_left(module)
    if left:
        raise SpadAllocRefused(f"a scratchpad buffer is still allocated or freed after {NAME}: {left[0]}")
    return module




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
