"""torchsim-lower-to-vcix: the lane number and the transcendental math ops -> VCIX. The ops are
`vcix.*` DIALECT ops whose own LLVM translation names the intrinsic; lib_vcix mints them."""

from .lib_ir import walk
from .lib_math import VCIX_MATH
from .lib_vcix import i64, lane_number, vcix

FORBIDS = ("torchsim.vlane_idx",) + VCIX_MATH

MARKERS = ("torchsim.vlane_idx",) + VCIX_MATH

_VIV_ENCODING = {
    "exp":  (0b000011, 0),
    "erf":  (0b000000, 0),
    "tanh": (0b000001, 0),
    "sin":  (0b000010, 0),
    "cos":  (0b000010, 1),
    "log":  (0b000010, 2),
    "atan": (0b000010, 3),
}


def _get_viv_table(names, encoding):
    """math op name -> (opcode, imm), iterating lib_math's VCIX_MATH so no second copy of the
    vocabulary lives here. A name with no opcode, or an opcode with no name, stops the import:
    that op would fall to one libm call per lane, a wrong kernel nothing downstream flags."""
    bare = {n: n.split(".", 1)[1] for n in names}
    table = {n: encoding[b] for n, b in bare.items() if b in encoding}
    missing = [n for n in names if n not in table]
    orphan = sorted(set(encoding) - set(bare.values()))
    if missing or orphan:
        raise ImportError(f"VCIX_MATH and _VIV_ENCODING disagree: {missing} have no opcode, "
                          f"{orphan} name no op")
    return table


_MATH_VIV = _get_viv_table(VCIX_MATH, _VIV_ENCODING)


def run(module, *, vectorlane, vlen, **_):
    """Every lane number, then every transcendental, onto its VCIX op in place. The matmul
    pass before this one mints lane numbers too, which is why a matmul is a marker here."""
    for op in [o for o in walk(module.operation) if o.name == "torchsim.vlane_idx"]:
        _rewrite_vlane_idx(op)
    for op in [o for o in walk(module.operation) if o.name in _MATH_VIV]:
        _rewrite_transcendental(op, vlen)
    return module


def _rewrite_vlane_idx(op):
    """torchsim.vlane_idx -> the VCIX op reading the lane number (opcode 0, `vl` the result's
    length): a tile banked on axis 0 puts element i in lane i."""
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


def _rewrite_transcendental(op, vlen):
    """One transcendental onto `vcix.v.iv`, refused on anything but a rank-1 vector of an
    element the unit takes, no wider than one register group -- legalize guarantees both."""
    from mlir.ir import InsertionPoint, VectorType
    from ...machine import ISA_MAX_LMUL
    opcode, imm = _MATH_VIV[op.name]
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
    """The vtype's element width for this element type, 0 for one it has no width for."""
    from mlir.ir import F32Type, F64Type, IndexType, IntegerType
    if isinstance(elt_ty, F32Type):
        return 32
    if isinstance(elt_ty, (F64Type, IndexType)):
        return 64
    if isinstance(elt_ty, IntegerType):
        return IntegerType(elt_ty).width
    return 0


def _build_viv_padded(operand, vt, opcode, imm):
    """A `vcix.v.iv` at the current IP, at a width no register type holds padded to the next
    power of two and cut back -- llc widens no VCIX intrinsic."""
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
