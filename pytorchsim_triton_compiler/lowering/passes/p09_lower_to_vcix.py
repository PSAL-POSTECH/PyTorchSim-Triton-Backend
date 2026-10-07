"""torchsim-lower-to-vcix: the lane number, the transcendental math ops and a kernel's own
torchsim.vcix_elem -> VCIX."""

from ...contract.sfu import SFU
from .lib_ir import int_attr, walk
from .lib_math import VCIX_MATH
from .lib_vcix import i64, lane_number, vcix

VCIX_ELEM = "torchsim.vcix_elem"

FORBIDS = ("torchsim.vlane_idx", VCIX_ELEM) + VCIX_MATH

MARKERS = ("torchsim.vlane_idx", VCIX_ELEM) + VCIX_MATH

_MATH_SFU = {
    "math.exp":  "vexp",
    "math.erf":  "verf",
    "math.tanh": "vtanh",
    "math.sin":  "vsin",
    "math.cos":  "vcos",
    "math.log":  "vlog",
    "math.atan": "vatan",
}


def _get_viv_table(names, by_op):
    missing = [n for n in names if by_op.get(n) not in SFU]
    orphan = sorted(set(by_op) - set(names))
    if missing or orphan:
        raise ImportError(f"VCIX_MATH and _MATH_SFU disagree: {missing} have no SFU instruction, "
                          f"{orphan} name no op")
    return {n: SFU[by_op[n]] for n in names}


_MATH_VIV = _get_viv_table(VCIX_MATH, _MATH_SFU)


def run(module, *, vectorlane, vlen, **_):
    """Every lane number, then every transcendental and kernel-named instruction, onto its VCIX op
    in place."""
    for op in [o for o in walk(module.operation) if o.name == "torchsim.vlane_idx"]:
        _rewrite_vlane_idx(op)
    for op in [o for o in walk(module.operation) if o.name in _MATH_VIV or o.name == VCIX_ELEM]:
        _rewrite_to_viv(op, vlen)
    return module


def _rewrite_vlane_idx(op):
    from mlir.dialects import arith
    from mlir.ir import InsertionPoint, IntegerAttr, IntegerType, VectorType
    vt = VectorType(op.results[0].type)
    i64_ty = IntegerType.get_signless(64)
    with InsertionPoint(op):
        rvl = arith.ConstantOp(i64_ty, IntegerAttr.get(i64_ty, vt.shape[0])).result
        lane = lane_number(rvl, vt.shape[0])
        idx = arith.IndexCastOp(vt, lane).result
    op.results[0].replace_all_uses_with(idx)
    op.erase()


def _rewrite_to_viv(op, vlen):
    from mlir.ir import InsertionPoint, VectorType
    from ...machine import ISA_MAX_LMUL
    opcode, imm = (_MATH_VIV[op.name] if op.name in _MATH_VIV
                   else (int_attr(op, "opcode"), int_attr(op, "imm")))
    if len(op.operands) != 1:
        raise NotImplementedError(f"{op.name} with {len(op.operands)} operands: sf.vc.v.iv takes "
                                  f"one vector: {op}")
    try:
        vt = VectorType(op.results[0].type)
    except (TypeError, ValueError):
        raise NotImplementedError(f"{op.name} on a scalar: VCIX takes a vector register, and "
                                  f"legalize reissues every scalar transcendental as one: {op}")
    sew = _get_sew(vt.element_type)
    if len(vt.shape) != 1 or not sew:
        raise NotImplementedError(f"{op.name} on {vt}: the unit takes a rank-1 vector of "
                                  f"f32 or f64: {op}")
    if vt.shape[0] * sew > vlen * ISA_MAX_LMUL:
        raise NotImplementedError(f"{vt} is wider than one register group at LMUL "
                                  f"{ISA_MAX_LMUL}; legalize caps it: {op}")
    with InsertionPoint(op):
        new = _build_viv_padded(op.operands[0], vt, opcode, imm)
    op.results[0].replace_all_uses_with(new)
    op.erase()


def _get_sew(elt_ty):
    from mlir.ir import F32Type, F64Type, IndexType, IntegerType
    if isinstance(elt_ty, F32Type):
        return 32
    if isinstance(elt_ty, (F64Type, IndexType)):
        return 64
    if isinstance(elt_ty, IntegerType):
        return IntegerType(elt_ty).width
    return 0


def _build_viv_padded(operand, vt, opcode, imm):
    from mlir.ir import DenseI64ArrayAttr, IntegerAttr, IntegerType, Operation, VectorType

    def viv(v, ty):
        return vcix("vcix.v.iv", [v], [ty], {"opcode": i64(opcode), "imm": IntegerAttr.get(
            IntegerType.get_signless(32), int(imm))}).results[0]

    def shuffle(v, ty, mask):
        return Operation.create("vector.shuffle", results=[ty], operands=[v, v],
                                attributes={"mask": DenseI64ArrayAttr.get(mask)}).results[0]

    w = vt.shape[0]
    if not w & (w - 1):
        return viv(operand, vt)
    wide_ty = VectorType.get([1 << (w - 1).bit_length()], vt.element_type)
    wide = shuffle(operand, wide_ty, [i if i < w else 0 for i in range(wide_ty.shape[0])])
    return shuffle(viv(wide, wide_ty), vt, list(range(w)))
