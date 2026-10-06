"""torchsim-fuse: collapse the elementwise linalg chain into one op,
AFTER the layouts are decided, as upstream fuses after layout assignment -- a broadcast
included. What the layouts decided -- a matmul and a reshape -- is fenced."""

from .lib_ir import funcs, int_array, name_of, op_of, parallel_iterator, rank_of, walk

FENCE = "torchsim.fusion_fence"

FORBIDS = (FENCE,)

_ARRAY_OPS = ("torchsim.dot",)

_RESHAPES = ("tensor.collapse_shape", "tensor.expand_shape", "tensor.reshape")


def run(module, **_opts):
    """The table of contents: fence, let upstream fuse, unfence, lower what fusion left."""
    from mlir.ir import Location
    from mlir.passmanager import PassManager

    with Location.unknown():
        for f in funcs(module):
            _rewrite_broadcasts_to_generics(f.operation)
            _rewrite_array_fences(f.operation)
            _rewrite_reshape_fences(f.operation)
            PassManager.parse("func.func(linalg-fuse-elementwise-ops)").run(f.operation)
            _erase_fences(f.operation)
    _lower_the_affine_apply_fusion_left(module)
    return module


def _rewrite_broadcasts_to_generics(func):
    """Every `linalg.broadcast` as the generic it means, so fusion folds it into its reader:
    upstream fuses only generics, and p02 made each broadcast explicit to see the lanes."""
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


def _rewrite_array_fences(func):
    """Fence every shaped input and result of a contraction, not the `outs`: upstream fuses
    into a matmul from the input side and expands one from the consumer side."""
    for op in [o for o in walk(func) if name_of(o) in _ARRAY_OPS]:
        for k in range(len(op.operands) - len(op.results)):
            if rank_of(op.operands[k]) is not None:
                _rewrite_operand_fenced(op, k)
        for r in op.results:
            if rank_of(r) is not None:
                _rewrite_result_fenced(r)


def _rewrite_reshape_fences(func):
    """Fence every reshape on both sides: after layout assignment it is a decided view or move."""
    for op in [o for o in walk(func) if name_of(o) in _RESHAPES]:
        _rewrite_operand_fenced(op, 0)
        _rewrite_result_fenced(op.results[0])


def _rewrite_operand_fenced(op, k):
    """`op` reads operand `k` through a fence just before it."""
    from mlir.ir import InsertionPoint, Operation
    v = op.operands[k]
    with InsertionPoint(op_of(op)):
        f = Operation.create(FENCE, results=[v.type], operands=[v])
    op.operands[k] = f.results[0]


def _rewrite_result_fenced(r):
    """Every reader of `r` reads it through a fence just after its maker."""
    from mlir.ir import InsertionPoint, Operation
    uses = [(u.owner, u.operand_number) for u in r.uses]
    if not uses:
        return
    with InsertionPoint.after(op_of(r.owner)):
        f = Operation.create(FENCE, results=[r.type], operands=[r])
    for owner, n in uses:
        owner.operands[n] = f.results[0]


def _erase_fences(func):
    """Take every fence back out, found by name: fusion erases ops a kept handle may name."""
    for f in [o for o in walk(func) if name_of(o) == FENCE]:
        f.results[0].replace_all_uses_with(f.operands[0])
        op_of(f).erase()


def _lower_the_affine_apply_fusion_left(module):
    """Fusing through a reshape leaves an `affine.apply`, which lower_to_dma has no case for;
    lower-affine turns it into the arith it means."""
    from mlir.passmanager import PassManager
    PassManager.parse(
        "builtin.module(func.func(canonicalize,cse),lower-affine,"
        "func.func(canonicalize,cse))").run(module.operation)
