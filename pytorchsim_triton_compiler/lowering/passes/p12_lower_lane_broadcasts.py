"""pytorchsim-lower-lane-broadcasts: a vector.broadcast of a one-element vector -- a value each
lane holds on its own -- becomes an RVV vrgather of element 0, never a trip through the one scalar
register file every lane shares. An unused one is erased; one with no register group is refused."""
from .lib_ir import walk

NAME = "pytorchsim-lower-lane-broadcasts"

MARKERS = ("vector.broadcast",)

GROUPS = (1, 2, 4, 8)


class LaneBroadcastRefused(Exception):
    """A lane value's broadcast this pass has no lowering for: refused, never scalarized."""


def is_lane_broadcast(op):
    """A vector.broadcast whose source is a vector of one element."""
    from mlir.ir import VectorType
    if op.name != "vector.broadcast" or not isinstance(op.operands[0].type, VectorType):
        return False
    src = VectorType(op.operands[0].type)
    return all(d == 1 for d in src.shape)


def _integer_view(elem):
    """(the integer element a lane value of `elem` is gathered as, the cast there, the cast back)."""
    from mlir.dialects import arith
    from mlir.ir import BF16Type, F16Type, F32Type, F64Type, IndexType, IntegerType
    if isinstance(elem, IndexType):
        return IntegerType.get_signless(64), arith.IndexCastOp, arith.IndexCastOp
    if isinstance(elem, IntegerType) and elem.width == 1:
        return IntegerType.get_signless(8), arith.ExtUIOp, arith.TruncIOp
    if isinstance(elem, IntegerType) and elem.width in (8, 16, 32, 64):
        return elem, None, None
    if isinstance(elem, (BF16Type, F16Type, F32Type, F64Type)):
        return IntegerType.get_signless(elem.width), arith.BitcastOp, arith.BitcastOp
    raise LaneBroadcastRefused(f"a lane value of element type {elem} has no integer view to gather")


def _group(n, bits, vlen):
    whole = 1 << (n - 1).bit_length()
    m = next((m for m in GROUPS if whole * bits <= m * vlen), None)
    if m is None:
        raise LaneBroadcastRefused(
            f"a lane value broadcast to {n} x i{bits} needs {whole * bits // vlen} registers at "
            f"VLEN {vlen}, above one LMUL-{GROUPS[-1]} group")
    return whole, m


def _gather0(src, n, elem, vlen):
    """`src`'s element 0 in all `n` positions: a vrgather over the register group of next_pow2(n),
    extracted at that power of two, then narrowed to `n`."""
    from mlir.dialects import arith, llvm
    from mlir.ir import IntegerAttr, IntegerType, Operation, VectorType
    ity, there, back = _integer_view(elem)
    whole, m = _group(n, ity.width, vlen)
    i64 = IntegerType.get_signless(64)
    one = there(VectorType.get([1], ity), src).result if there else src
    sc = VectorType.get([m * 64 // ity.width], ity, scalable=[True])
    poison = llvm.PoisonOp(sc).result
    put = Operation.create("llvm.intr.vector.insert", results=[sc], operands=[poison, one],
                           attributes={"pos": IntegerAttr.get(i64, 0)}).results[0]
    wide = llvm.call_intrinsic(sc, "llvm.riscv.vrgather.vx",
                               [poison, put, arith.ConstantOp(i64, 0).result,
                                arith.ConstantOp(i64, n).result], [], [])
    for k in dict.fromkeys([whole, n]):
        wide = Operation.create("llvm.intr.vector.extract", results=[VectorType.get([k], ity)],
                                operands=[wide],
                                attributes={"pos": IntegerAttr.get(i64, 0)}).results[0]
    return back(VectorType.get([n], elem), wide).result if back else wide


def run(module, *, target, **_):
    """Lower every lane-value broadcast of `module`; refuse one to rank above 1 or wider than a
    register group, and any this pass leaves."""
    from mlir.ir import InsertionPoint, VectorType
    for op in [o for o in walk(module.operation) if is_lane_broadcast(o)]:
        res = VectorType(op.results[0].type)
        if not any(True for _ in op.results[0].uses):
            op.erase()
            continue
        if len(res.shape) != 1:
            raise LaneBroadcastRefused(f"a lane value broadcast to rank {len(res.shape)}: {op}")
        if res.shape[0] == 1:
            wide = op.operands[0]
        else:
            with InsertionPoint(op), op.location:
                wide = _gather0(op.operands[0], res.shape[0], res.element_type, target.vlen_bits)
        op.results[0].replace_all_uses_with(wide)
        op.erase()
    left = [o for o in walk(module.operation) if is_lane_broadcast(o)]
    if left:
        raise LaneBroadcastRefused(f"{NAME} left a lane value's broadcast: {left[0]}")
