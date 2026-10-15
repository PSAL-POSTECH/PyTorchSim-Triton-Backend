"""tritonpytorchsim-lower-vector-functions-tpuv6e: the lane number and the math ops of SFU_FUNCTIONS,
the one statement of what vcix_tpuv6e's SFU computes, onto its units (libtpuv6e.so): vcix.v.iv per
function, padded to a power-of-two width, and the lane-index unit's vcix.v.i."""

from ...passes.lib_ir import walk
from ...passes.lib_vcix import i64, vcix
from ...passes.lower import LOWER_VECTOR_FUNCTIONS
from .lib_units import lane_number

NAME = "tritonpytorchsim-lower-vector-functions-tpuv6e"

REQUIRES = ("vcix_tpuv6e",)

SLOT = LOWER_VECTOR_FUNCTIONS.name

SFU_FUNCTIONS = {
    "math.exp":  (0b000011, 0),
    "math.erf":  (0b000000, 0),
    "math.tanh": (0b000001, 0),
    "math.sin":  (0b000010, 0),
    "math.cos":  (0b000010, 1),
    "math.log":  (0b000010, 2),
    "math.atan": (0b000010, 3),
}

FORBIDS = LOWER_VECTOR_FUNCTIONS.lowers + tuple(SFU_FUNCTIONS)

MARKERS = FORBIDS


def run(module, *, target, **_):
    """Every lane number, then every transcendental, onto its VCIX op in place."""
    vlen = target.vlen_bits
    for op in [o for o in walk(module.operation) if o.name == "torchsim.vlane_idx"]:
        _rewrite_vlane_idx(op)
    for op in [o for o in walk(module.operation) if o.name in SFU_FUNCTIONS]:
        _rewrite_transcendental(op, vlen, SFU_FUNCTIONS[op.name])
    return module


def _rewrite_vlane_idx(op):
    from mlir.dialects import arith
    from mlir.ir import InsertionPoint, IntegerAttr, IntegerType, VectorType
    from mlir.ir import DenseI64ArrayAttr, Operation
    vt = VectorType(op.results[0].type)
    i64_ty = IntegerType.get_signless(64)
    w = vt.shape[0]
    wide = w if not w & (w - 1) else 1 << (w - 1).bit_length()   # the unit takes a power of two
    with InsertionPoint(op):
        rvl = arith.ConstantOp(i64_ty, IntegerAttr.get(i64_ty, wide)).result
        lane = lane_number(rvl, wide)
        if wide != w:
            lane = Operation.create(
                "vector.shuffle", results=[VectorType.get([w], VectorType(lane.type).element_type)],
                operands=[lane, lane],
                attributes={"mask": DenseI64ArrayAttr.get(list(range(w)))}).results[0]
        idx = arith.IndexCastOp(vt, lane).result
    op.results[0].replace_all_uses_with(idx)
    op.erase()


def _rewrite_transcendental(op, vlen, encoding):
    from mlir.ir import InsertionPoint, VectorType
    from ....machine import ISA_MAX_LMUL
    opcode, imm = encoding
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
