"""tritonpytorchsim-lower-vector-functions-tpuv6e: the lane number and the math ops of SFU_FUNCTIONS,
the one statement of what vcix_tpuv6e's SFU computes, onto its units (libtpuv6e.so): vcix.v.iv per
function, padded to a power-of-two width, and the lane-index unit's vcix.v.i."""

from ...passes.lib_ir import walk
import os

from ...passes.lib_vcix import bitcast, i64, vcix
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
    if os.environ.get("TORCHSIM_VCIX_PACK"):
        for st in [o for o in walk(module.operation) if o.name == "affine.vector_store"]:
            _rewrite_bf16_store(st)
        for ld in [o for o in walk(module.operation) if o.name == "affine.vector_load"]:
            _rewrite_bf16_load(ld)
    return module


# The misc unit's bf16 pack (a model with it: run/mxu/sm/prefix-unpack2's libtpuv6e.so, TORCHSIM_VCIX_PACK=1):
# one vcix.v.sv (sf.vc.v.vv) opcode 2, as the v6e's vpack takes two registers: bf16 of vs2's f32 then vs1's,
# rounded to nearest even and packed two to a 32-bit word.
PACK = 0b000010
# ... and its unpack (prefix-unpack's libtpuv6e.so): imm 8 / 9 give the first / last vl of vs2's 2 * vl bf16 as f32.
UNPACK_LO, UNPACK_HI = 8, 9


def _word_view(mem, so, mt):
    """`mem` (bf16, last dim unit-stride and whole words) as 32-bit words: the same bytes, its type alone."""
    from mlir.ir import IntegerType, MemRefType, Operation, StridedLayoutAttr
    from ...passes.lib_pack import PACK_OP
    i32 = IntegerType.get_signless(32)
    words = MemRefType.get(list(mt.shape[:-1]) + [mt.shape[-1] // 2], i32,
                           StridedLayoutAttr.get(so[1] // 2, [s // 2 for s in so[0][:-1]] + [1]),
                           mt.memory_space)
    return Operation.create(PACK_OP, results=[words], operands=[mem]).results[0]


def _halved_map(r):
    from mlir.ir import AffineConstantExpr, AffineFloorDivExpr, AffineMap, AffineMapAttr
    ident = AffineMap.get_identity(r).results
    return AffineMapAttr.get(AffineMap.get(r, 0, [ident[i] for i in range(r - 1)] + [
        AffineFloorDivExpr.get(ident[r - 1], AffineConstantExpr.get(2))]))


def _packable(mem, amap, n):
    """(strides, offset) of a bf16 buffer whose last dim the words can halve, read at identity indices."""
    from mlir.ir import AffineMap, MemRefType, ShapedType
    from ...passes.lib_ir import strides_and_offset
    mt = MemRefType(mem.type)
    so = strides_and_offset(mem)
    dyn = ShapedType.get_dynamic_stride_or_offset()
    if (n < 2 or n % 2 or so is None or dyn in list(so[0]) + [so[1]] or so[0][-1] != 1 or so[1] % 2
            or any(s % 2 for s in so[0][:-1]) or mt.shape[-1] % 2
            or amap != AffineMap.get_identity(len(mt.shape))):
        return None
    return so


def _rewrite_bf16_load(ld):
    """A bf16 vector loaded and only widened to f32 becomes words loaded and the two halves through the
    unpack, the f32 put end to end: no 16-bit vtype for the load and the widening."""
    from mlir.dialects import affine, vector
    from mlir.ir import (AffineMapAttr, BF16Type, DenseI64ArrayAttr, F32Type, InsertionPoint,
                         IntegerAttr, IntegerType, MemRefType, Operation, VectorType)
    val, mem = ld.results[0], ld.operands[0]
    vt = VectorType(val.type)
    users = [u.owner for u in val.uses]
    if (len(vt.shape) != 1 or not isinstance(vt.element_type, BF16Type) or not users
            or any(getattr(u, "name", None) != "arith.extf"
                   or not isinstance(VectorType(u.results[0].type).element_type, F32Type)
                   for u in users)):
        return
    n = vt.shape[0]
    so = _packable(mem, AffineMapAttr(ld.attributes["map"]).value, n)
    if so is None:
        return
    mt = MemRefType(mem.type)
    i32 = IntegerType.get_signless(32)
    half = VectorType.get([n // 2], i32)
    f32h, f32 = VectorType.get([n // 2], F32Type.get()), VectorType.get([n], F32Type.get())
    imm = lambda k: IntegerAttr.get(IntegerType.get_signless(32), k)  # noqa: E731
    with InsertionPoint(ld):
        view = _word_view(mem, so, mt)
        w = affine.AffineVectorLoadOp(half, view, list(ld.operands[1:]), _halved_map(len(mt.shape))).result
        lo = vcix("vcix.v.iv", [w], [half], {"opcode": i64(PACK), "imm": imm(UNPACK_LO)}).results[0]
        hi = vcix("vcix.v.iv", [w], [half], {"opcode": i64(PACK), "imm": imm(UNPACK_HI)}).results[0]
        wide = Operation.create("vector.shuffle", results=[f32], operands=[bitcast(lo, f32h), bitcast(hi, f32h)],
                                attributes={"mask": DenseI64ArrayAttr.get(list(range(n)))}).results[0]
    for u in users:
        u.results[0].replace_all_uses_with(wide)
        u.erase()
    ld.erase()


def _rewrite_bf16_store(st):
    """An f32 vector cast to bf16 and stored becomes the two halves through the pack, stored as
    32-bit words: no 16-bit vtype for the cast and the store, so no e32/e16 vsetvli each trip."""
    from mlir.dialects import affine, vector
    from mlir.ir import (AffineConstantExpr, AffineFloorDivExpr, AffineMap, AffineMapAttr, BF16Type, F32Type, InsertionPoint, IntegerAttr,
                         IntegerType, MemRefType, Operation, ShapedType, StridedLayoutAttr,
                         VectorType)
    from ...passes.lib_ir import strides_and_offset
    from ...passes.lib_pack import PACK_OP
    val, mem = st.operands[0], st.operands[1]
    cast = getattr(val.owner, "name", None) == "arith.truncf" and val.owner
    if not cast or len(list(val.uses)) != 1:
        return
    vt, src = VectorType(val.type), VectorType(cast.operands[0].type)
    mt = MemRefType(mem.type)
    n = vt.shape[0] if len(vt.shape) == 1 else 0
    if not (isinstance(vt.element_type, BF16Type) and isinstance(src.element_type, F32Type)
            and n >= 2 and n % 2 == 0):
        return
    so = strides_and_offset(mem)
    dyn = ShapedType.get_dynamic_stride_or_offset()
    amap = AffineMapAttr(st.attributes["map"]).value
    if (so is None or dyn in list(so[0]) + [so[1]] or so[0][-1] != 1 or so[1] % 2
            or any(s % 2 for s in so[0][:-1]) or mt.shape[-1] % 2
            or amap != AffineMap.get_identity(len(mt.shape))):
        return
    i32 = IntegerType.get_signless(32)
    half = VectorType.get([n // 2], i32)
    words = MemRefType.get(list(mt.shape[:-1]) + [mt.shape[-1] // 2], i32,
                           StridedLayoutAttr.get(so[1] // 2, [s // 2 for s in so[0][:-1]] + [1]),
                           mt.memory_space)
    imm = lambda k: IntegerAttr.get(IntegerType.get_signless(32), k)  # noqa: E731
    with InsertionPoint(st):
        x = cast.operands[0]
        f32h = VectorType.get([n // 2], src.element_type)
        lo = vector.ExtractStridedSliceOp(f32h, x, [0], [n // 2], [1]).result
        hi = vector.ExtractStridedSliceOp(f32h, x, [n // 2], [n // 2], [1]).result
        got = vcix("vcix.v.sv", [bitcast(lo, half), bitcast(hi, half)], [half],
                   {"opcode": i64(PACK)}).results[0]
        view = Operation.create(PACK_OP, results=[words], operands=[mem]).results[0]
        r = len(mt.shape)
        wmap = AffineMap.get(r, 0, [AffineMap.get_identity(r).results[i] for i in range(r - 1)]
                             + [AffineFloorDivExpr.get(AffineMap.get_identity(r).results[r - 1],
                                                        AffineConstantExpr.get(2))])
        affine.AffineVectorStoreOp(got, view, list(st.operands[2:]), AffineMapAttr.get(wmap))
    st.erase()
    cast.erase()


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
