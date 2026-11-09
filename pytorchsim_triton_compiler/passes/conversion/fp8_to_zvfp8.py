"""fp8 -> Zvfp8: every convert to or from an 8-bit float as Zvfofp8min's instruction (format in
vtype.altfmt, e8 = E4M3, e8alt = E5M2), then every fp8 left in a type retyped to i8, as LLVM IR has no
fp8. fp8 is storage, convert and matrix-unit input only: arithmetic on it is refused."""
from ...contract.stop import Reason, Refusal

_LMUL = {1: "mf8", 2: "mf4", 4: "mf2", 8: "m1", 16: "m2", 32: "m4", 64: "m8"}

_UNARY_ASM = "li t0, {n}\nvsetvli zero, t0, {sew}, {lmul}, ta, ma\n{op} $0, $1"
_UNARY_CONSTRAINTS = "=&^vr,^vr,~{t0}"

_CARRIERS = ("arith.select", "arith.bitcast", "arith.constant")


class Fp8Refused(NotImplementedError):
    """An fp8 op this lowering has no Zvfp8 instruction for."""


class Fp8ArithmeticRefused(NotImplementedError, Refusal):
    """Arithmetic on an 8-bit float: the vector unit only converts fp8, it does not compute on it."""
    reason = Reason.DTYPE_NOT_RUN


def run(module, *, subtarget):
    """Lower every fp8 op in `module` in place; a module without fp8 is left as it is."""
    from ..utils.ir_utils import walk
    if not any(_has_fp8(o) for o in walk(module.operation)):
        return module
    vlen = subtarget.vlen_bits
    _rewrite_converts(module, vlen)
    _rewrite_constants(module)
    _check_no_fp8_arithmetic(module)
    _rewrite_storage(module)
    return module


def _rewrite_converts(module, vlen):
    """Every truncf/extf across an fp8 -> one Zvfp8 convert: f32 -> fp8 in one rounding
    (vfncvt.f.f.q), f16 -> fp8 (vfncvt.f.f.w), fp8 -> f16 (vfwcvt.f.f.v) then extf to the rest."""
    import mlir.ir as ir
    from mlir.dialects import arith
    from ..utils.ir_utils import walk
    todo = [o for o in walk(module.operation) if o.name in ("arith.truncf", "arith.extf")
            and (_get_fp8_vector(o.results[0]) or _get_fp8_vector(o.operands[0]))]
    for op in todo:
        res, src = op.results[0], op.operands[0]
        with ir.InsertionPoint(op), op.location:
            if _get_fp8_vector(res):
                n, sew = _get_fp8_vector(res)
                wide = ir.VectorType(src.type).element_type
                if _is_a(wide, ir.F32Type):
                    bits = _build_convert(src, n, vlen, sew, "vfncvt.f.f.q", 4)
                elif _is_a(wide, ir.F16Type):
                    bits = _build_convert(src, n, vlen, sew, "vfncvt.f.f.w", 2)
                else:
                    raise Fp8Refused(f"{wide} -> fp8 has no one-rounding Zvfp8 convert: {op}")
                new = _build_from_i8(bits, res.type)
            else:
                n, sew = _get_fp8_vector(src)
                half = _build_convert(_build_as_i8(src), n, vlen, sew, "vfwcvt.f.f.v", 2,
                                      to=ir.F16Type.get())
                new = half if half.type == res.type else arith.ExtFOp(res.type, half).result
        res.replace_all_uses_with(new)
        op.erase()


def _rewrite_constants(module):
    """Every fp8 zero constant, scalar or vector -> the i8 zero, bitcast back. Any other value would be
    rounding to the format in the compiler, which is what the instruction is for, so it is refused."""
    import mlir.ir as ir
    from mlir.dialects import arith
    from ..utils.ir_utils import walk
    for op in [o for o in walk(module.operation) if o.name == "arith.constant"]:
        res = op.results[0]
        got = _get_fp8_vector(res)
        if got is None and _get_fp8_sew(res.type) is None:
            continue
        if not _is_zero(op.attributes["value"]):
            raise Fp8Refused(f"a non-zero fp8 constant would be rounded by the compiler: {op}")
        i8 = ir.IntegerType.get_signless(8)
        with ir.InsertionPoint(op), op.location:
            if got is None:
                new = arith.BitcastOp(res.type, arith.ConstantOp(i8, ir.IntegerAttr.get(i8, 0)).result).result
            else:
                ty = ir.VectorType.get([got[0]], i8)
                splat = ir.DenseElementsAttr.get_splat(ty, ir.IntegerAttr.get(i8, 0))
                zero = arith.ConstantOp(ty, splat).result
                new = _build_from_i8(zero, res.type)
        res.replace_all_uses_with(new)
        op.erase()


def _check_no_fp8_arithmetic(module):
    """Refuse an arith or math op still reading or making an fp8 once the converts are gone: retyped
    to i8 it would compute on the bits."""
    from ..utils.ir_utils import walk
    for op in walk(module.operation):
        if op.name.split(".")[0] in ("arith", "math") and op.name not in _CARRIERS and _has_fp8(op):
            raise Fp8ArithmeticRefused(f"{op.name} on an 8-bit float: fp8 is converted, not computed "
                                       f"on; widen it first: {op}")


def _rewrite_storage(module):
    """Every fp8 left in a type -> i8, in place, through the bindings: results, block arguments and
    type attributes (a function's signature). What is left is memory and the moves of it, which do
    not read the format; an fp8 value still held by an attribute is refused."""
    import mlir.ir as ir
    from ..utils.ir_utils import walk
    for op in walk(module.operation):
        for r in op.results:
            r.set_type(_get_retyped(r.type))
        for region in op.regions:
            for block in region.blocks:
                for a in block.arguments:
                    a.set_type(_get_retyped(a.type))
        for name in [getattr(a, "name", a) for a in op.attributes]:
            attr = op.attributes[name]
            try:
                ty = ir.TypeAttr(attr).value
            except (TypeError, ValueError):
                ty = None
            if ty is not None:
                op.attributes[name] = ir.TypeAttr.get(_get_retyped(ty))
            elif _holds_fp8(getattr(attr, "type", None)):
                raise Fp8Refused(f"an fp8 value in attribute {name} has no i8 spelling here: {op}")


def _build_convert(src, n, vlen, sew, op, ratio, to=None):
    """One Zvfp8 convert of `src` (n elements); `ratio` is the wide side's bits over 8, and the
    result is i8 unless `to` names the wide element type."""
    return _build_asm([src], n, vlen, sew, op, 64 // ratio, to=to)


def _build_asm(operands, n, vlen, sew, op, max_eighths, to=None):
    """`op` under `vsetvli n, sew` as inline asm over scalable registers: each fixed operand inserted
    at element 0, the fixed result extracted back. THE VSETVLI IS THE i8 SIDE's: a wide operand is the
    same scalable count at a wider LMUL, so `max_eighths` keeps the widest within m8."""
    import mlir.ir as ir
    from mlir.dialects import llvm
    eighths = _get_eighths(n, vlen)
    if eighths > max_eighths:
        raise Fp8Refused(f"an {n}-element fp8 {op} needs LMUL {eighths}/8 on the e8 side, and its "
                         f"widest operand would pass m8")
    i64 = ir.IntegerType.get_signless(64)
    big = []
    for v in operands:
        sc = ir.VectorType.get([eighths], ir.VectorType(v.type).element_type, scalable=[True])
        big.append(ir.Operation.create(
            "llvm.intr.vector.insert", results=[sc], operands=[llvm.PoisonOp(sc).result, v],
            attributes={"pos": ir.IntegerAttr.get(i64, 0)}).results[0])
    out_elt = to if to is not None else ir.IntegerType.get_signless(8)
    out_sc = ir.VectorType.get([eighths], out_elt, scalable=[True])
    asm = _UNARY_ASM.format(n=n, sew=sew, lmul=_LMUL[eighths], op=op)
    got = llvm.InlineAsmOp(out_sc, big, asm, _UNARY_CONSTRAINTS,
                           has_side_effects=False, asm_dialect=0).res
    return ir.Operation.create(
        "llvm.intr.vector.extract", results=[ir.VectorType.get([n], out_elt)], operands=[got],
        attributes={"pos": ir.IntegerAttr.get(i64, 0)}).results[0]


def _get_eighths(n, vlen):
    """The e8 side's LMUL in eighths for n elements: the scalable count k with vscale * k >= n."""
    vscale, k = vlen // 64, 1
    while vscale * k < n:
        k *= 2
    return k


def _build_as_i8(v):
    """`v`, an fp8 vector, as the i8 vector of its bits."""
    import mlir.ir as ir
    from mlir.dialects import vector
    vt = ir.VectorType(v.type)
    return vector.BitCastOp(ir.VectorType.get(list(vt.shape), ir.IntegerType.get_signless(8)), v).result


def _build_from_i8(v, ty):
    """The i8 vector `v` as `ty`, the fp8 vector it holds the bits of."""
    from mlir.dialects import vector
    return vector.BitCastOp(ty, v).result


def _get_fp8_sew(ty):
    """The vsetvli element spelling of an 8-bit float type -- e8 for E4M3, e8alt for E5M2 -- else None."""
    import mlir.ir as ir
    if ty is None:
        return None
    for cls, sew in ((ir.Float8E4M3FNType, "e8"), (ir.Float8E5M2Type, "e8alt")):
        if _is_a(ty, cls):
            return sew
    return None


def _get_fp8_vector(v):
    """(element count, vsetvli element) when `v` is a 1-D vector of an 8-bit float, else None."""
    import mlir.ir as ir
    if not _is_a(v.type, ir.VectorType):
        return None
    vt = ir.VectorType(v.type)
    sew = _get_fp8_sew(vt.element_type)
    if sew is None:
        return None
    if vt.rank != 1 or vt.scalable:
        raise Fp8Refused(f"an fp8 vector of rank {vt.rank} reached the e8 lowering: {v.type}")
    return vt.shape[0], sew


def _holds_fp8(ty):
    """Is `ty` an 8-bit float, or shaped (vector, memref, tensor) over one?"""
    import mlir.ir as ir
    if ty is None:
        return False
    if _is_a(ty, ir.ShapedType):
        return _get_fp8_sew(ir.ShapedType(ty).element_type) is not None
    return _get_fp8_sew(ty) is not None


def _has_fp8(op):
    """Does any operand or result of `op` hold an 8-bit float?"""
    return any(_holds_fp8(v.type) for v in list(op.operands) + list(op.results))


def _get_retyped(ty):
    """`ty` with its 8-bit float element (or itself, if a scalar one) as i8; a function type
    argument by argument. Every other type unchanged."""
    import mlir.ir as ir
    i8 = ir.IntegerType.get_signless(8)
    if _get_fp8_sew(ty) is not None:
        return i8
    if _is_a(ty, ir.FunctionType):
        ft = ir.FunctionType(ty)
        return ir.FunctionType.get([_get_retyped(t) for t in ft.inputs],
                                   [_get_retyped(t) for t in ft.results])
    if not _holds_fp8(ty):
        return ty
    if _is_a(ty, ir.VectorType):
        vt = ir.VectorType(ty)
        return ir.VectorType.get(list(vt.shape), i8, scalable=list(vt.scalable_dims))
    if _is_a(ty, ir.MemRefType):
        mt = ir.MemRefType(ty)
        return ir.MemRefType.get(list(mt.shape), i8, mt.layout, mt.memory_space)
    if _is_a(ty, ir.UnrankedMemRefType):
        return ir.UnrankedMemRefType.get(i8, ir.UnrankedMemRefType(ty).memory_space)
    if _is_a(ty, ir.RankedTensorType):
        rt = ir.RankedTensorType(ty)
        return ir.RankedTensorType.get(list(rt.shape), i8, rt.encoding)
    raise Fp8Refused(f"a type over an 8-bit float this lowering cannot retype: {ty}")


def _is_a(ty, cls):
    """Is `ty` of the binding type `cls`?"""
    try:
        cls(ty)
        return True
    except (TypeError, ValueError):
        return False


def _is_zero(attr):
    """Is this float constant (scalar or dense) zero everywhere?"""
    import mlir.ir as ir
    try:
        return all(float(x) == 0.0 for x in ir.DenseFPElementsAttr(attr))
    except (TypeError, ValueError):
        pass
    try:
        return float(ir.FloatAttr(attr).value) == 0.0
    except (TypeError, ValueError):
        return False
