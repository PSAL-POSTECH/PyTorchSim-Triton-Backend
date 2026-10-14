"""tritonpytorchsim-lower-vector-functions-tpuv6: the lane number and the transcendental math ops
onto vcix_tpuv6's units (libtpuv6.so) -- the SFU's vcix.v.iv per function, padded to a power-of-two
width, and the lane-index unit's vcix.v.i."""

from ...passes.lib_ir import walk
from ...passes.lib_math import VCIX_MATH
from ...passes.lib_vcix import i64, vcix
from .lib_units import lane_number

NAME = "tritonpytorchsim-lower-vector-functions-tpuv6"

REQUIRES = ("vcix_tpuv6",)

SLOT = "lower-vector-functions"

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


def _get_viv_table(names=VCIX_MATH, encoding=None):
    """Each op of `names` to the SFU's (funct6, imm); refused, naming this pass, where the table and
    the shared VCIX_MATH disagree -- at run, so another model's table cannot break this one's import."""
    encoding = _VIV_ENCODING if encoding is None else encoding
    bare = {n: n.split(".", 1)[1] for n in names}
    table = {n: encoding[b] for n, b in bare.items() if b in encoding}
    missing = [n for n in names if n not in table]
    orphan = sorted(set(encoding) - set(bare.values()))
    if missing or orphan:
        raise NotImplementedError(f"{NAME}: VCIX_MATH and _VIV_ENCODING disagree: {missing} have "
                                  f"no opcode, {orphan} name no op")
    return table


def run(module, *, target, **_):
    """Every lane number, then every transcendental, onto its VCIX op in place."""
    vlen = target.vlen_bits
    table = _get_viv_table()
    for op in [o for o in walk(module.operation) if o.name == "torchsim.vlane_idx"]:
        _rewrite_vlane_idx(op)
    for op in [o for o in walk(module.operation) if o.name in table]:
        _rewrite_transcendental(op, vlen, table[op.name])
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
