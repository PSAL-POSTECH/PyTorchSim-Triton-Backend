"""pytorchsim-lower-lane-reductions: a vector.multi_reduction of vector<1xN> into vector<1> -- one
lane's own row folded to a value it keeps -- becomes one RVV vred*/vfred* whose answer stays in
element 0 of a register, never a trip through the scalar register file every lane shares."""
from .lib_ir import walk
from . import pass_name

NAME = pass_name(__name__)

MARKERS = ("vector.multi_reduction",)

GROUPS = (1, 2, 4, 8)

FLOAT_REDUCE = {"add": "vfredusum", "maxnumf": "vfredmax", "minnumf": "vfredmin"}

INT_REDUCE = {"add": "vredsum", "maxsi": "vredmax", "minsi": "vredmin", "maxui": "vredmaxu",
              "minui": "vredminu", "and": "vredand", "or": "vredor", "xor": "vredxor"}

FRM_DYNAMIC = 7


class LaneReductionRefused(Exception):
    """A lane reduction this pass has no RVV instruction for: refused, never scalarized."""


def _kind(op):
    from .lib_ir import enum_of
    return enum_of(op.attributes["kind"], "#vector.kind<%s>",
                   sorted(set(FLOAT_REDUCE) | set(INT_REDUCE) | {"mul", "maximumf", "minimumf"}))


def instruction(kind, elem):
    """The RVV reduction that folds `kind` over `elem`, or None when the V extension has none."""
    from mlir.ir import F32Type, F64Type, IntegerType
    if isinstance(elem, (F32Type, F64Type)):
        return FLOAT_REDUCE.get(kind)
    if isinstance(elem, IntegerType) and elem.width in (8, 16, 32, 64):
        return INT_REDUCE.get(kind)
    return None


def is_lane_reduction(op):
    """A vector.multi_reduction of vector<1xN> over dim 1 into vector<1>."""
    from mlir.ir import VectorType
    if op.name != "vector.multi_reduction" or not isinstance(op.results[0].type, VectorType):
        return False
    src, res = VectorType(op.operands[0].type), VectorType(op.results[0].type)
    return list(src.shape)[:1] == [1] and len(src.shape) == 2 and list(res.shape) == [1]


def _scalable(n_regs, elem):
    from mlir.ir import VectorType
    return VectorType.get([n_regs * 64 // elem.width], elem, scalable=[True])


def _lower(op, vlen):
    """`op` as its RVV reduction over the first n elements of the register group holding its row,
    padded to the next power of two; the answer read back from element 0."""
    from mlir.dialects import arith, llvm
    from mlir.ir import DenseI64ArrayAttr, IntegerAttr, IntegerType, Operation, VectorType
    src_t = VectorType(op.operands[0].type)
    n, elem = src_t.shape[1], src_t.element_type
    kind = _kind(op)
    insn = instruction(kind, elem)
    if insn is None:
        raise LaneReductionRefused(f"no RVV reduction folds <{kind}> over {elem}: {op}")
    whole = 1 << (n - 1).bit_length()
    m = next((m for m in GROUPS if whole * elem.width <= m * vlen), None)
    if m is None:
        raise LaneReductionRefused(f"a lane reduction over {n} x {elem} is wider than one "
                                   f"LMUL-{GROUPS[-1]} group at VLEN {vlen}: {op}")
    i64 = IntegerType.get_signless(64)
    flat = Operation.create("vector.shape_cast", results=[VectorType.get([n], elem)],
                            operands=[op.operands[0]]).results[0]
    if whole != n:
        flat = Operation.create("vector.shuffle", results=[VectorType.get([whole], elem)],
                                operands=[flat, flat],
                                attributes={"mask": DenseI64ArrayAttr.get(
                                    list(range(n)) + [-1] * (whole - n))}).results[0]
    group, one = _scalable(m, elem), _scalable(1, elem)
    src = Operation.create("llvm.intr.vector.insert", results=[group],
                           operands=[llvm.PoisonOp(group).result, flat],
                           attributes={"pos": IntegerAttr.get(i64, 0)}).results[0]
    acc = Operation.create("llvm.intr.vector.insert", results=[one],
                           operands=[llvm.PoisonOp(one).result, op.operands[1]],
                           attributes={"pos": IntegerAttr.get(i64, 0)}).results[0]
    tail = [arith.ConstantOp(i64, n).result]
    if insn == "vfredusum":
        tail = [arith.ConstantOp(i64, FRM_DYNAMIC).result] + tail
    got = llvm.call_intrinsic(one, f"llvm.riscv.{insn}",
                              [llvm.PoisonOp(one).result, src, acc] + tail, [], [])
    return Operation.create("llvm.intr.vector.extract", results=[op.results[0].type],
                            operands=[got], attributes={"pos": IntegerAttr.get(i64, 0)}).results[0]


def run(module, *, target, **_):
    """Lower every lane reduction of `module` to its RVV instruction; refuse one the V extension
    has no instruction for, and any this pass leaves."""
    from mlir.ir import InsertionPoint
    for op in [o for o in walk(module.operation) if is_lane_reduction(o)]:
        with InsertionPoint(op), op.location:
            got = _lower(op, target.vlen_bits)
        op.results[0].replace_all_uses_with(got)
        op.erase()
    left = [o for o in walk(module.operation) if is_lane_reduction(o)]
    if left:
        raise LaneReductionRefused(f"{NAME} left a lane reduction: {left[0]}")
