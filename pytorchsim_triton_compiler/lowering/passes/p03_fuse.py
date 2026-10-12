"""torchsim-fuse: collapse the elementwise linalg chain into one op, AFTER the layouts are decided,
as upstream fuses after layout assignment -- a broadcast included. A cast that several ops read is
re-made at each reader, and one a reduce reads goes into the reduce's body, so a widened copy of a
tile is never a walk of the scratchpad on its own."""

from .lib_ir import (attrs_of, funcs, int_array, name_of, op_of, parallel_iterator, rank_of,
                     walk)

FENCE = "torchsim.fusion_fence"

FORBIDS = (FENCE,)

_ARRAY_OPS = ("torchsim.dot",)

_RESHAPES = ("tensor.collapse_shape", "tensor.expand_shape", "tensor.reshape")

_CASTS = ("arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp", "arith.fptosi",
          "arith.fptoui", "arith.extsi", "arith.extui", "arith.trunci")


def run(module, **_opts):
    """The table of contents: fence, let upstream fuse, unfence, lower what fusion left."""
    from mlir.ir import Location
    from mlir.passmanager import PassManager

    with Location.unknown():
        for f in funcs(module):
            _rewrite_broadcasts_to_generics(f.operation)
            _rewrite_array_fences(f.operation)
            _rewrite_reshape_fences(f.operation)
            _rewrite_casts_into_readers(f.operation)
            PassManager.parse("func.func(linalg-fuse-elementwise-ops)").run(f.operation)
            _erase_fences(f.operation)
    _clean_up(module)
    return module


def _rewrite_broadcasts_to_generics(func):
    from mlir.dialects import linalg
    from mlir.ir import (AffineDimExpr, AffineMap, AffineMapAttr, ArrayAttr, DenseI32ArrayAttr,
                         InsertionPoint, Operation, RankedTensorType)
    for op in [o for o in walk(func) if name_of(o) == "linalg.broadcast"]:
        src, init = op.operands[0], op.operands[1]
        rank = rank_of(init)
        added = set(int_array(op.attributes["dimensions"]))
        kept = [AffineDimExpr.get(d) for d in range(rank) if d not in added]
        with InsertionPoint(op_of(op)):
            gen = Operation.create(
                "linalg.generic", results=[init.type], operands=[src, init],
                attributes={"indexing_maps": ArrayAttr.get([
                                AffineMapAttr.get(AffineMap.get(rank, 0, kept)),
                                AffineMapAttr.get(AffineMap.get_identity(rank))]),
                            "iterator_types": ArrayAttr.get([parallel_iterator()] * rank),
                            "operandSegmentSizes": DenseI32ArrayAttr.get([1, 1])},
                regions=1)
        blk = gen.regions[0].blocks.append(RankedTensorType(src.type).element_type,
                                           RankedTensorType(init.type).element_type)
        with InsertionPoint(blk):
            linalg.YieldOp([blk.arguments[0]])
        op.results[0].replace_all_uses_with(gen.results[0])
        op_of(op).erase()


def _rewrite_casts_into_readers(func):
    """A cast-only generic leaves no tile of its own: a reduce reading it takes the cast into its
    body, and every elementwise reader gets a copy of its own, which upstream then fuses."""
    from mlir.ir import InsertionPoint
    for cast in [o for o in walk(func) if _is_cast(o)]:
        for owner, n in [(op_of(u.owner), u.operand_number) for u in cast.results[0].uses]:
            if name_of(owner) == "linalg.reduce" and len(owner.operands) == 2 and n == 0:
                _rewrite_reduce_over_cast(owner, cast)
            elif name_of(owner) == "linalg.generic" and n < len(owner.operands) - len(owner.results):
                owner.operands[n] = cast.clone(ip=InsertionPoint(owner)).results[0]
        if not any(True for _ in cast.results[0].uses):
            cast.erase()


def _is_cast(op):
    """An elementwise generic of one tile whose body is casts of its element and nothing else."""
    from mlir.ir import AffineMap, AffineMapAttr, ArrayAttr
    if name_of(op) != "linalg.generic" or len(op.operands) != 2 or len(op.results) != 1:
        return False
    same = AffineMap.get_identity(rank_of(op.results[0]))
    if any(AffineMapAttr(m).value != same for m in ArrayAttr(op.attributes["indexing_maps"])):
        return False
    blk = op.regions[0].blocks[0]
    cur, steps = blk.arguments[0], [i.operation for i in blk.operations]
    for st in steps[:-1]:
        if st.name not in _CASTS or st.operands[0] != cur:
            return False
        cur = st.results[0]
    return len(steps) > 1 and steps[-1].name == "linalg.yield" and steps[-1].operands[0] == cur


def _rewrite_reduce_over_cast(red, cast):
    """`red` again, reading the cast's own input and casting each element before it combines."""
    from mlir.ir import InsertionPoint, Operation
    src, init = cast.operands[0], red.operands[1]
    old = red.regions[0].blocks[0]
    with InsertionPoint(red):
        new = Operation.create("linalg.reduce", results=[r.type for r in red.results],
                               operands=[src, init], attributes=attrs_of(red), regions=1,
                               loc=red.location)
    blk = new.regions[0].blocks.append(cast.regions[0].blocks[0].arguments[0].type,
                                       old.arguments[1].type)
    env = {cast.regions[0].blocks[0].arguments[0]: blk.arguments[0]}
    with InsertionPoint(blk):
        for body, last in ((cast.regions[0].blocks[0], False), (old, True)):
            for inner in body.operations:
                o = inner.operation
                if o.name == "linalg.yield":
                    if last:
                        Operation.create("linalg.yield", operands=[env[v] for v in o.operands])
                    else:
                        env[old.arguments[0]], env[old.arguments[1]] = env[o.operands[0]], blk.arguments[1]
                    break
                got = Operation.create(o.name, results=[r.type for r in o.results],
                                       operands=[env.get(v, v) for v in o.operands],
                                       attributes=attrs_of(o), loc=o.location)
                env.update(zip(o.results, got.results))
    for a, b in zip(red.results, new.results):
        a.replace_all_uses_with(b)
    red.erase()


def _rewrite_array_fences(func):
    for op in [o for o in walk(func) if name_of(o) in _ARRAY_OPS]:
        for k in range(len(op.operands) - len(op.results)):
            if rank_of(op.operands[k]) is not None:
                _rewrite_operand_fenced(op, k)
        for r in op.results:
            if rank_of(r) is not None:
                _rewrite_result_fenced(r)


def _rewrite_reshape_fences(func):
    for op in [o for o in walk(func) if name_of(o) in _RESHAPES]:
        _rewrite_operand_fenced(op, 0)
        _rewrite_result_fenced(op.results[0])


def _rewrite_operand_fenced(op, k):
    from mlir.ir import InsertionPoint, Operation
    v = op.operands[k]
    with InsertionPoint(op_of(op)):
        f = Operation.create(FENCE, results=[v.type], operands=[v])
    op.operands[k] = f.results[0]


def _rewrite_result_fenced(r):
    from mlir.ir import InsertionPoint, Operation
    uses = [(u.owner, u.operand_number) for u in r.uses]
    if not uses:
        return
    with InsertionPoint.after(op_of(r.owner)):
        f = Operation.create(FENCE, results=[r.type], operands=[r])
    for owner, n in uses:
        owner.operands[n] = f.results[0]


def _erase_fences(func):
    for f in [o for o in walk(func) if name_of(o) == FENCE]:
        f.results[0].replace_all_uses_with(f.operands[0])
        op_of(f).erase()


def _clean_up(module):
    from mlir.passmanager import PassManager
    PassManager.parse("builtin.module(func.func(canonicalize,cse))").run(module.operation)
