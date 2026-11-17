"""lower_extremes: every f32 vector `arith.minimumf` / `arith.maximumf` onto the Misc unit's minimum / maximum
(misc.minimum, misc.maximum), one instruction that carries a NaN as the v6e's vmin.f32 / vmax.f32 do, where
RISC-V's vfmin / vfmax drop it and LLVM adds a compare and merge to each. A scalar, or another element type,
stays as it is."""

from ...passes.utils.ir_utils import walk
from ...passes.utils.vcix import bitcast, from_unit, to_unit, unit_vector
from .pytorchsim_tpu import MiscMaximumOp, MiscMinimumOp
from ...passes import pass_name  # noqa: E402

NAME = pass_name(__name__)

#: Each op, and the Misc unit's op it becomes.
EXTREMES = {"arith.minimumf": MiscMinimumOp, "arith.maximumf": MiscMaximumOp}

MARKERS = tuple(EXTREMES)


def run(module, *, subtarget, **_):
    """Each f32 rank-1 vector minimumf / maximumf no wider than a register group, replaced where it stands."""
    from ...driver.machine import ISA_MAX_LMUL
    for op in [o for o in walk(module.operation) if o.name in EXTREMES]:
        _rewrite(op, subtarget.vlen_bits * ISA_MAX_LMUL)
    return module


def _rewrite(op, bits):
    """`op` as its Misc op on the f32 words of both operands, if it is an f32 vector of at most `bits`."""
    from mlir.dialects import arith
    from mlir.ir import F32Type, IntegerAttr, IntegerType, InsertionPoint, VectorType
    try:
        vt = VectorType(op.results[0].type)
    except (TypeError, ValueError):
        return
    if len(vt.shape) != 1 or not isinstance(vt.element_type, F32Type) or vt.shape[0] * 32 > bits:
        return
    n = vt.shape[0]
    i32, i64 = IntegerType.get_signless(32), IntegerType.get_signless(64)
    words, unit = VectorType.get([n], i32), VectorType.get([unit_vector(n)], i32)
    with InsertionPoint(op):
        vl = arith.ConstantOp(i64, IntegerAttr.get(i64, n)).result
        lhs, rhs = (to_unit(bitcast(v, words), n) for v in op.operands)
        got = bitcast(from_unit(EXTREMES[op.name](unit, lhs, rhs, vl).result, n), vt)
    op.results[0].replace_all_uses_with(got)
    op.erase()
