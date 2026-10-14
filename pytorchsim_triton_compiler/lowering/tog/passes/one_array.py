"""The sample gem5 times drives the matrix unit's first array alone. A target's `matmul_units`
arrays work in the same cycles -- the TPU v6e's bundles carry mxu0's and mxu1's instructions side by
side -- and the core gem5 models sends the unit one instruction a cycle, so the arrays' instructions
in one stream would be timed one after the other. The program Spike runs keeps them all."""
import mlir.ir as ir

from ...passes.lib_ir import walk

# Which array a matrix unit's instruction is for: sf.vc.iv's rd, sf.vc.v.i's rs2 (opcode 2, the vpop).
VPOP = 2
# What feeds an erased instruction and nothing else: its register's read, and the cast of it.
FEEDS = ("vector.transfer_read", "arith.bitcast")


def _int(op, name):
    return ir.IntegerAttr(op.attributes[name]).value if name in op.attributes else 0


def _unused(op):
    return all(not list(r.uses) for r in op.results)


def keep_first_array(module):
    """Erases every vmatprep, vmatpush and vmatmul of an array past the first, with the reads only
    they took, and stands zeros in for each vpop of one: the add that took its rows stays, as the
    vector unit does that add whatever the arrays do. How many instructions went."""
    from mlir.dialects import arith
    gone = 0
    for op in list(walk(module.operation)):
        if op.name == "vcix.iv" and _int(op, "rd"):
            feeds = [v.owner for v in op.operands if isinstance(v.owner, (ir.Operation, ir.OpView))]
            op.erase()
            gone += 1
            while feeds:
                f = feeds.pop()
                f = f.operation if isinstance(f, ir.OpView) else f
                if f.name in FEEDS and _unused(f):
                    feeds += [v.owner for v in f.operands
                              if isinstance(v.owner, (ir.Operation, ir.OpView))]
                    f.erase()
        elif op.name == "vcix.v.i" and _int(op, "opcode") == VPOP and _int(op, "rs2"):
            ty = ir.VectorType(op.results[0].type)
            with ir.InsertionPoint(op), op.location:
                zero = arith.ConstantOp(ty, ir.DenseElementsAttr.get_splat(
                    ty, ir.IntegerAttr.get(ty.element_type, 0))).result
            op.results[0].replace_all_uses_with(zero)
            op.erase()
            gone += 1
    return gone
