"""lower_round_convert: `arith.fptosi` of a clamp to integral bounds of `math.roundeven` -- torch's quantize, round
then clamp then cast -- as the clamp of the unrounded value and one conversion that rounds to nearest even
(llvm.intr.lrint), as the v6e's vclampa and vcvt.f32.s32 do; RISC-V otherwise rounds by switching the rounding mode
around a vfcvt and back each trip. The two agree: rounding is monotone and the bounds are integers. A clamp to
the narrower integer's own range is done after the conversion, on integers, where it takes no f32 constant."""

from ...passes.utils.ir_utils import walk
from ...passes import pass_name  # noqa: E402

NAME = pass_name(__name__)

CLAMPS = ("arith.maximumf", "arith.minimumf")

MARKERS = ("math.roundeven",)


def run(module, **_):
    """Each roundeven whose one reader starts such a clamp and cast, rewritten where it stands."""
    for op in [o for o in walk(module.operation) if o.name == "math.roundeven"]:
        _rewrite(op)
    return module


def _only_user(value):
    users = [u.owner for u in value.uses]
    return users[0] if len(users) == 1 else None


def _integral_splat(value):
    """Whether `value` is an arith.constant splat of an integer-valued float."""
    from mlir.ir import DenseFPElementsAttr
    owner = getattr(value, "owner", None)
    owner = getattr(owner, "operation", owner)
    if getattr(owner, "name", None) != "arith.constant":
        return False
    try:
        attr = DenseFPElementsAttr(owner.attributes["value"])
    except (TypeError, ValueError):
        return False
    return attr.is_splat and float(attr.get_splat_value().value).is_integer()


def _bound(op, x):
    """The constant operand of clamp op `op` whose other operand is `x`, or None."""
    a, b = op.operands
    other = b if a == x else a if b == x else None
    return other if other is not None and _integral_splat(other) else None


def _whole_range(first, lo, second, hi, width):
    """(min, max) of a `width`-bit signed integer below 32 bits when the clamp is to exactly that range, its
    maximum by the minimum and its minimum by the maximum; else None."""
    from mlir.ir import DenseFPElementsAttr
    if width >= 32:
        return None
    by = {op.name: float(DenseFPElementsAttr(b.owner.attributes["value"]).get_splat_value().value)
          for op, b in ((first, lo), (second, hi))}
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    if by.get("arith.maximumf") == low and by.get("arith.minimumf") == high:
        return low, high
    return None


def _rewrite(rnd):
    from mlir.dialects import arith
    from mlir.ir import DenseElementsAttr, InsertionPoint, IntegerAttr, IntegerType, Operation, VectorType
    first = _only_user(rnd.results[0])
    if first is None or first.name not in CLAMPS:
        return
    second = _only_user(first.results[0])
    if second is None or second.name not in CLAMPS or second.name == first.name:
        return
    cast = _only_user(second.results[0])
    if cast is None or cast.name != "arith.fptosi":
        return
    lo, hi = _bound(first, rnd.results[0]), _bound(second, first.results[0])
    try:
        vt, out = VectorType(rnd.results[0].type), VectorType(cast.results[0].type)
    except (TypeError, ValueError):
        return
    if lo is None or hi is None or IntegerType(out.element_type).width > 32:
        return
    width = IntegerType(out.element_type).width
    i32 = VectorType.get(vt.shape, IntegerType.get_signless(32))
    whole = _whole_range(first, lo, second, hi, width)
    with InsertionPoint(cast):
        if whole:
            got = Operation.create("llvm.intr.lrint", results=[i32], operands=[rnd.operands[0]]).results[0]
            for op, v in ((arith.MaxSIOp, whole[0]), (arith.MinSIOp, whole[1])):
                got = op(got, arith.ConstantOp(i32, DenseElementsAttr.get_splat(
                    i32, IntegerAttr.get(i32.element_type, v))).result).result
        else:
            a = Operation.create(first.name, results=[vt], operands=[rnd.operands[0], lo]).results[0]
            b = Operation.create(second.name, results=[vt], operands=[a, hi]).results[0]
            got = Operation.create("llvm.intr.lrint", results=[i32], operands=[b]).results[0]
        if width < 32:
            got = arith.TruncIOp(out, got).result
    cast.results[0].replace_all_uses_with(got)
    for op in (cast, second, first, rnd):
        op.erase()
