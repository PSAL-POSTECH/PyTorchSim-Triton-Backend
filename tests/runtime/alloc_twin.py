"""A fixture's alloc twin: every scratchpad global it names becomes a space-1 memref.alloc where its
get_global stood, freed by a memref.dealloc before its function returns -- the IR #230 moves the
pipeline to -- so a pass test can hold the two forms to one answer."""


def to_allocs(module):
    """`module` in place: each scratchpad get_global an alloc of its type, its global gone, and one
    dealloc of each alloc ahead of its function's terminator. Returns the allocs in walk order."""
    from mlir.dialects import memref
    from mlir.ir import InsertionPoint, MemRefType, TypeAttr
    top = module.operation.regions[0].blocks[0]
    made = []
    for fn in [o.operation for o in top.operations if o.operation.name == "func.func"]:
        mine = []
        for get in [o for o in _walk(fn) if o.name == "memref.get_global"
                    and in_scratchpad(o.results[0].type)]:
            with InsertionPoint(get):
                a = memref.AllocOp(MemRefType(get.results[0].type), [], []).result
            get.results[0].replace_all_uses_with(a)
            get.erase()
            mine.append(a)
        if mine:
            entry = fn.regions[0].blocks[0]
            with InsertionPoint(entry.operations[len(entry.operations) - 1]):
                for a in mine:
                    memref.DeallocOp(a)
        made += mine
    for g in [o.operation for o in top.operations if o.operation.name == "memref.global"
              and in_scratchpad(TypeAttr(o.operation.attributes["type"]).value)]:
        g.erase()
    return made


def in_scratchpad(ty):
    from mlir.ir import IntegerAttr, MemRefType
    mt = MemRefType(ty)
    return mt.memory_space is not None and IntegerAttr(mt.memory_space).value == 1


def _walk(op):
    out = []
    for region in op.regions:
        for block in region.blocks:
            for o in block.operations:
                out.append(o.operation)
                out += _walk(o.operation)
    return out
