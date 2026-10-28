"""lower_vector_functions: the lane number and the math ops of SFU_FUNCTIONS onto the model's units, an
sfu.<function> op a function and misc.lane_idx the lane number (to_vcix writes them as VCIX); an f32 exp
goes to the EUP as the v6e computes it, x * log2(e) pushed (eup.push) and its 2^x popped (eup.pop)."""

from ...passes.utils.ir_utils import walk

from ...dialect.pytorchsim import LaneIdOp
from .pytorchsim_tpu import (
    EupPopOp, EupPushOp, MiscPackOp, MiscUnpackHiOp, MiscUnpackLoOp, SfuAtanOp, SfuCosOp, SfuErfOp, SfuExpOp,
    SfuLogOp, SfuSinOp, SfuTanhOp)
from ...passes.utils.vcix import bitcast, from_unit, to_unit, unit_vector
from .units import lane_number
from ...passes import pass_name  # noqa: E402

NAME = pass_name(__name__)

# Each math op the SFU computes, and its unit op (to_vcix writes the instruction).
SFU_FUNCTIONS = {
    "math.exp": SfuExpOp.OPERATION_NAME,
    "math.erf": SfuErfOp.OPERATION_NAME,
    "math.tanh": SfuTanhOp.OPERATION_NAME,
    "math.sin": SfuSinOp.OPERATION_NAME,
    "math.cos": SfuCosOp.OPERATION_NAME,
    "math.log": SfuLogOp.OPERATION_NAME,
    "math.atan": SfuAtanOp.OPERATION_NAME,
}

# What reaches the SFU as it is.
NATIVE_FUNCTIONS = frozenset(SFU_FUNCTIONS)

FORBIDS = (LaneIdOp.OPERATION_NAME,) + tuple(SFU_FUNCTIONS)

MARKERS = FORBIDS


def run(module, *, subtarget, **_):
    """The selects a loop's trips settle folded and cut apart (_split_by_selects, then
    _hoist_idempotent_folds), the rest made of signs where they can be (_select_by_sign), the
    broadcasts of one element made by register (_broadcast_by_register); then every lane number,
    then every transcendental, onto its VCIX op in place. An f32 exp goes to the EUP when
    the machine states what the EUP holds (schedule_vpu_stream.eup_registers), else to its
    vcix.v.iv, and the function says why on EUP_PIPELINE."""
    from mlir.ir import StringAttr
    from .schedule_vpu_stream import EUP_PIPELINE, LATENCY_KEY, QUEUE_KEY, eup_registers
    vlen = subtarget.vlen_bits
    has_eup = eup_registers(subtarget) is not None
    for lp in [o for o in walk(module.operation) if o.name == "affine.for"]:
        _split_by_selects(lp, subtarget.vector_lanes)
    for lp in [o for o in walk(module.operation) if o.name == "affine.for"]:
        _hoist_idempotent_folds(lp)
    for op in [o for o in walk(module.operation) if o.name == "arith.select"]:
        if not _fold_select(op, subtarget.vector_lanes, {}):
            _select_by_sign(op, subtarget.vector_lanes)
    for op in [o for o in walk(module.operation) if o.name == "vector.broadcast"]:
        _broadcast_by_register(op, vlen)
    for op in [o for o in walk(module.operation) if o.name == "pytorchsim.lane_id"]:
        _rewrite_lane_id(op)
    for op in [o for o in walk(module.operation) if o.name in SFU_FUNCTIONS]:
        if op.name == "math.exp" and _is_f32_vector(op.results[0].type) and has_eup:
            _rewrite_exp_eup(op, vlen)
            continue
        if op.name == "math.exp" and _is_f32_vector(op.results[0].type):
            func = op.parent
            while func is not None and func.name != "func.func":
                func = func.parent
            if func is not None:
                func.attributes[EUP_PIPELINE] = StringAttr.get(
                    f"the subtarget states no {QUEUE_KEY} and {LATENCY_KEY}: exp stays on the SFU's "
                    f"vcix.v.iv")
        _rewrite_transcendental(op, vlen, SFU_FUNCTIONS[op.name])
    for st in [o for o in walk(module.operation) if o.name == "affine.vector_store"]:
        _rewrite_bf16_store(st)
    for ld in [o for o in walk(module.operation) if o.name == "affine.vector_load"]:
        _rewrite_bf16_load(ld)
    return module


# The misc unit's bf16 pack (misc.pack, as the v6e's vpack takes two registers) and its unpack
# (misc.unpack_lo, misc.unpack_hi: the first / last vl of 2 * vl bf16 as f32).


def _word_view(mem, so, mt):
    """`mem` (bf16, last dim unit-stride and whole words) as 32-bit words: the same bytes, its type alone."""
    from mlir.ir import IntegerType, MemRefType, Operation, StridedLayoutAttr
    from .pack import PACK_OP
    i32 = IntegerType.get_signless(32)
    words = MemRefType.get(list(mt.shape[:-1]) + [mt.shape[-1] // 2], i32,
                           StridedLayoutAttr.get(so[1] // 2, [s // 2 for s in so[0][:-1]] + [1]),
                           mt.memory_space)
    return Operation.create(PACK_OP, results=[words], operands=[mem]).results[0]


def _halves(n):
    """Whether `n` elements go through the pack or unpack as two halves of n // 2."""
    return n >= 2 and n % 2 == 0


def _vl(h):
    from mlir.dialects import arith
    from mlir.ir import IntegerAttr, IntegerType
    i64_ty = IntegerType.get_signless(64)
    return arith.ConstantOp(i64_ty, IntegerAttr.get(i64_ty, h)).result


def _packable(mem, amap, n):
    """(strides, offset) of a bf16 buffer whose last dim the words can halve, read at identity indices."""
    from mlir.ir import AffineMap, MemRefType, ShapedType
    from ...passes.utils.ir_utils import strides_and_offset
    mt = MemRefType(mem.type)
    so = strides_and_offset(mem)
    dyn = ShapedType.get_dynamic_stride_or_offset()
    if (not _halves(n) or so is None or dyn in list(so[0]) + [so[1]] or so[0][-1] != 1 or so[1] % 2
            or any(s % 2 for s in so[0][:-1]) or mt.shape[-1] % 2
            or amap != AffineMap.get_identity(len(mt.shape))):
        return None
    return so


def _rewrite_bf16_load(ld):
    """A bf16 vector loaded and only widened to f32 becomes words loaded and the two halves through the
    unpack, the f32 put end to end: no 16-bit vtype for the load and the widening."""
    from mlir.ir import (AffineMapAttr, BF16Type, DenseI64ArrayAttr, F32Type, InsertionPoint,
                         IntegerType, MemRefType, Operation, VectorType)
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
    with InsertionPoint(ld):
        view = _word_view(mem, so, mt)
        # The last index is a whole word's (an even element, at an identity map): halved by a shift,
        # not by affine's floordiv, which lowers to a signed division each trip.
        from mlir.dialects import arith
        from ...passes.utils.ir_utils import const_index, transfer_read, zero_of
        idx = list(ld.operands[1:])
        idx[-1] = arith.ShRUIOp(idx[-1], const_index(1)).result
        w = transfer_read(half, view, idx, zero_of(i32), True)
        h = n // 2
        unit = VectorType.get([unit_vector(h)], i32)
        w, vl = to_unit(w, h), _vl(h)
        lo = from_unit(MiscUnpackLoOp(unit, w, vl).result, h)
        hi = from_unit(MiscUnpackHiOp(unit, w, vl).result, h)
        wide = Operation.create("vector.shuffle", results=[f32], operands=[bitcast(lo, f32h), bitcast(hi, f32h)],
                                attributes={"mask": DenseI64ArrayAttr.get(list(range(n)))}).results[0]
    for u in users:
        u.results[0].replace_all_uses_with(wide)
        u.erase()
    ld.erase()


def _rewrite_bf16_store(st):
    """An f32 vector cast to bf16 and stored becomes the two halves through the pack, stored as
    32-bit words: no 16-bit vtype for the cast and the store, so no e32/e16 vsetvli each trip."""
    from mlir.dialects import vector
    from mlir.ir import (AffineMap, AffineMapAttr, BF16Type, F32Type, InsertionPoint, IntegerType,
                         MemRefType, Operation, ShapedType, StridedLayoutAttr, VectorType)
    from ...passes.utils.ir_utils import strides_and_offset
    from .pack import PACK_OP
    val, mem = st.operands[0], st.operands[1]
    cast = getattr(val.owner, "name", None) == "arith.truncf" and val.owner
    if not cast or len(list(val.uses)) != 1:
        return
    vt, src = VectorType(val.type), VectorType(cast.operands[0].type)
    mt = MemRefType(mem.type)
    n = vt.shape[0] if len(vt.shape) == 1 else 0
    if not (isinstance(vt.element_type, BF16Type) and isinstance(src.element_type, F32Type)
            and _halves(n)):
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
    with InsertionPoint(st):
        x = cast.operands[0]
        f32h = VectorType.get([n // 2], src.element_type)
        lo = vector.ExtractStridedSliceOp(f32h, x, [0], [n // 2], [1]).result
        hi = vector.ExtractStridedSliceOp(f32h, x, [n // 2], [n // 2], [1]).result
        h = n // 2
        unit = VectorType.get([unit_vector(h)], i32)
        got = from_unit(MiscPackOp(unit, to_unit(bitcast(lo, half), h), to_unit(bitcast(hi, half), h),
                                    vl=_vl(h)).result, h)
        view = Operation.create(PACK_OP, results=[words], operands=[mem]).results[0]
        # The last index is a whole word's (an even element, at an identity map): halved by a shift,
        # not by affine's floordiv, which lowers to a signed division each trip.
        from mlir.dialects import arith
        from ...passes.utils.ir_utils import const_index, transfer_write
        idx = list(st.operands[2:])
        idx[-1] = arith.ShRUIOp(idx[-1], const_index(1)).result
        transfer_write(got, view, idx, True)
    st.erase()
    cast.erase()


def _rewrite_lane_id(op):
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
        elem = vt.element_type
        if isinstance(elem, IntegerType) and IntegerType(elem).width < 64:
            idx = arith.TruncIOp(vt, lane).result
        elif isinstance(elem, IntegerType):
            idx = lane
        else:
            idx = arith.IndexCastOp(vt, lane).result
    op.results[0].replace_all_uses_with(idx)
    op.erase()


def _rewrite_transcendental(op, vlen, unit_op):
    from mlir.ir import InsertionPoint, VectorType
    from ...driver.machine import ISA_MAX_LMUL
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
        new = _build_viv_padded(op.operands[0], vt, unit_op)
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


def _build_viv_padded(operand, vt, unit_op):
    from mlir.ir import DenseI64ArrayAttr, Operation, VectorType

    def viv(v, ty):
        return Operation.create(unit_op, results=[ty], operands=[v]).results[0]

    def shuffle(v, ty, mask):
        return Operation.create("vector.shuffle", results=[ty], operands=[v, v],
                                attributes={"mask": DenseI64ArrayAttr.get(mask)}).results[0]

    w = vt.shape[0]
    if not w & (w - 1):
        return viv(operand, vt)
    wide_ty = VectorType.get([1 << (w - 1).bit_length()], vt.element_type)
    wide = shuffle(operand, wide_ty, [i if i < w else 0 for i in range(wide_ty.shape[0])])
    return shuffle(viv(wide, wide_ty), vt, list(range(w)))


# The EUP: eup.push and eup.pop, the pop taking the oldest push's result; nothing in between reads the EUP. A
# push computes 2^x (the v6e's vpow2), so exp(x) is the push of x * log2(e).
LOG2_E = 1.4426950408889634
PUSH, POP = EupPushOp.OPERATION_NAME, EupPopOp.OPERATION_NAME


def _is_f32_vector(ty):
    from mlir.ir import F32Type, VectorType
    try:
        return isinstance(VectorType(ty).element_type, F32Type)
    except (TypeError, ValueError):
        return False


def _times_log2e(v, scale, vt):
    """`v` * log2(e) at the current point. Of `a - b` in a loop that does not make b (attention's row
    max over a row's trips) it is one fma, a * log2(e) - b * log2(e): the subtraction goes, and b's
    product is made once, before the loop; a b made every trip would cost a product each, so stays."""
    from mlir.ir import Operation
    from ...passes.utils.ir_utils import producer
    p = producer(v)
    body = p.block if p is not None else None
    if p is None or p.name != "arith.subf" or len(list(v.uses)) != 1 or \
            body.owner.name not in ("affine.for", "scf.for") or not _defined_outside(p.operands[1], body):
        return Operation.create("arith.mulf", results=[vt], operands=[v, scale]).results[0]
    from mlir.ir import InsertionPoint
    a, b = p.operands[0], p.operands[1]
    with InsertionPoint(body.owner):
        out_scale = Operation.create("arith.constant", results=[vt], attributes={
            "value": scale.owner.attributes["value"]}).results[0]
        bc = Operation.create("arith.mulf", results=[vt], operands=[b, out_scale]).results[0]
        nbc = Operation.create("arith.negf", results=[vt], operands=[bc]).results[0]
    return Operation.create("math.fma", results=[vt], operands=[a, scale, nbc]).results[0]


def _rewrite_exp_eup(op, vlen):
    from mlir.ir import DenseElementsAttr, F32Type, FloatAttr, InsertionPoint, Operation, VectorType
    from ...driver.machine import ISA_MAX_LMUL
    vt = VectorType(op.results[0].type)
    if len(vt.shape) != 1:
        raise NotImplementedError(f"math.exp on {vt}: the EUP takes a rank-1 vector of f32: {op}")
    if vt.shape[0] * 32 > vlen * ISA_MAX_LMUL:
        raise NotImplementedError(f"{vt} is wider than one register group at LMUL {ISA_MAX_LMUL}; "
                                  f"legalize caps it: {op}")
    with InsertionPoint(op):
        scale = Operation.create("arith.constant", results=[vt], attributes={
            "value": DenseElementsAttr.get_splat(vt, FloatAttr.get(F32Type.get(), LOG2_E))}).results[0]
        x = _times_log2e(op.operands[0], scale, vt)
        w = vt.shape[0]
        wide_ty = vt if not w & (w - 1) else VectorType.get([1 << (w - 1).bit_length()], vt.element_type)
        if wide_ty != vt:
            x = _shuffle(x, wide_ty, [i if i < w else 0 for i in range(wide_ty.shape[0])])
        EupPushOp(x)
        got = EupPopOp(wide_ty).result
        if wide_ty != vt:
            got = _shuffle(got, vt, list(range(w)))
    arg = op.operands[0]
    op.results[0].replace_all_uses_with(got)
    op.erase()
    from ...passes.utils.ir_utils import producer
    made = producer(arg)
    if made is not None and made.name == "arith.subf" and not len(list(made.results[0].uses)):
        made.erase()


def _shuffle(v, ty, mask):
    from mlir.ir import DenseI64ArrayAttr, Operation
    return Operation.create("vector.shuffle", results=[ty], operands=[v, v],
                            attributes={"mask": DenseI64ArrayAttr.get(mask)}).results[0]


def _defined_outside(v, body):
    from mlir.ir import BlockArgument
    if isinstance(v, BlockArgument):
        return v.owner != body
    return v.owner.operation.block != body


# arith.cmpi predicates whose answer is the sign of a difference: (lhs, rhs, minus one) such that
# the predicate holds iff lhs - rhs - minus is negative.
_BY_SIGN = {"slt": (0, 1, 0), "ult": (0, 1, 0), "sgt": (1, 0, 0), "ugt": (1, 0, 0),
            "sge": (1, 0, 1), "uge": (1, 0, 1), "sle": (0, 1, 1), "ule": (0, 1, 1)}
_PREDICATES = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]


def _select_by_sign(op, lanes):
    """`op`, an arith.select of 32-bit vectors by an arith.cmpi of two vectors of i32 that
    _int_range bounds so that their difference fits in an i32 (and, for an unsigned compare, both
    non-negative), rewritten as no = the false value's bits, keep = the difference's sign spread
    over the word, and no ^ ((no ^ yes) & keep): the same bits, through no mask register."""
    from mlir.dialects import arith
    from mlir.ir import (DenseElementsAttr, InsertionPoint, IntegerAttr, IntegerType,
                         VectorType)
    from ...passes.utils.ir_utils import elem_bits, producer
    cond, yes, no = op.operands
    cmp = producer(cond)
    if (cmp is None or cmp.name != "arith.cmpi" or not isinstance(yes.type, VectorType)
            or elem_bits(VectorType(yes.type).element_type) != 32):
        return
    pred = _PREDICATES[IntegerAttr(cmp.attributes["predicate"]).value]
    a, b = cmp.operands
    i32 = IntegerType.get_signless(32)
    if (pred not in _BY_SIGN or not isinstance(a.type, VectorType)
            or VectorType(a.type).element_type != i32):
        return
    first, second, minus = _BY_SIGN[pred]
    x, y = (a, b)[first], (a, b)[second]
    rx, ry = _int_range(x, lanes), _int_range(y, lanes)
    if rx is None or ry is None:
        return
    lo, hi = rx[0] - ry[1] - minus, rx[1] - ry[0] - minus
    if lo < -(1 << 31) or hi >= 1 << 31 or (pred[0] == "u" and min(rx[0], ry[0]) < 0):
        return
    vt = VectorType(yes.type)
    wt = VectorType.get(vt.shape, i32)

    def splat(n):
        return arith.ConstantOp(wt, DenseElementsAttr.get_splat(wt, IntegerAttr.get(i32, n))).result

    def bits(v):
        return v if VectorType(v.type).element_type == i32 else arith.BitcastOp(wt, v).result
    with InsertionPoint(op), op.location:
        d = arith.SubIOp(x, y).result
        if minus:
            d = arith.SubIOp(d, splat(minus)).result
        keep = arith.ShRSIOp(d, splat(31)).result
        n, v = bits(no), bits(yes)
        got = arith.XOrIOp(n, arith.AndIOp(arith.XOrIOp(n, v).result, keep).result).result
        if VectorType(no.type).element_type != i32:
            got = arith.BitcastOp(vt, got).result
    op.results[0].replace_all_uses_with(got)
    op.erase()


def _truth(op, lanes, fixed):
    """True or False when the arith.select `op`'s condition, an arith.cmpi of integer vectors, has
    that answer in every element whatever _int_range lets its operands hold (`fixed`: values held
    to a range); None otherwise."""
    from mlir.ir import IntegerAttr, VectorType
    from ...passes.utils.ir_utils import producer
    cmp = producer(op.operands[0])
    if cmp is None or cmp.name != "arith.cmpi" or not isinstance(cmp.operands[0].type, VectorType):
        return None
    pred = _PREDICATES[IntegerAttr(cmp.attributes["predicate"]).value]
    a, b = (_int_range(v, lanes, fixed=fixed) for v in cmp.operands)
    if a is None or b is None or (pred[0] == "u" and min(a[0], b[0]) < 0):
        return None
    if pred in ("slt", "ult", "sge", "uge"):
        lt = True if a[1] < b[0] else False if a[0] >= b[1] else None
        return lt if pred in ("slt", "ult") or lt is None else not lt
    if pred in ("sgt", "ugt", "sle", "ule"):
        gt = True if a[0] > b[1] else False if a[1] <= b[0] else None
        return gt if pred in ("sgt", "ugt") or gt is None else not gt
    return None


def _fold_select(op, lanes, fixed):
    """`op`, an arith.select whose condition _truth settles, replaced by the operand it picks;
    whether it was."""
    t = _truth(op, lanes, fixed)
    if t is None:
        return False
    op.results[0].replace_all_uses_with(op.operands[1] if t else op.operands[2])
    op.erase()
    return True


def _split_by_selects(lp, lanes):
    """An innermost loop of constant bounds whose body selects by a compare _truth settles in each
    trip -- a mask's compare of a loop counter against lane rows, all of whose elements one trip
    answers alike at most rows -- cut into one loop per run of trips that settle every such select
    the same way, each select there folded to what it picks (or kept, where a trip leaves it
    open): a causal mask then masks only the trips on the diagonal."""
    from mlir.dialects import affine
    from mlir.ir import (AffineAddExpr, AffineConstantExpr, AffineDimExpr, AffineMap, AffineMapAttr,
                         InsertionPoint, Operation)
    from ...passes.utils.ir_utils import loop_bounds, trip_count
    body = lp.regions[0].blocks[0]
    ops = [o.operation for o in body.operations]
    term, ops = ops[-1], ops[:-1]
    trips, bounds = trip_count(lp), loop_bounds(lp)
    if not trips or trips < 2 or any(o.regions for o in ops):
        return
    sels = [o for o in ops if o.name == "arith.select"]
    if not sels:
        return
    iv, carried = body.arguments[0], list(body.arguments[1:])
    lb, _, step = bounds
    runs = []
    for t in range(trips):
        at = lb + t * step
        cls = tuple(_truth(o, lanes, {iv: (at, at)}) for o in sels)
        if runs and runs[-1][2] == cls:
            runs[-1][1] = t + 1
        else:
            runs.append([t, t + 1, cls])
    if len(runs) == 1 and all(c is None for c in runs[0][2]):
        return
    from .schedule_vpu_stream import _BOUNDS
    ip = InsertionPoint(lp.operation)
    inits = list(lp.operation.operands)[-len(carried):] if carried else []
    for t0, t1, cls in runs:
        # Each from 0, the counter offset in its body: the TOG's sample takes a loop's first trip
        # to start at 0.
        with ip, lp.location:
            new = affine.AffineForOp(0, (t1 - t0) * step, step, iter_args=inits)
        for name in [lp.attributes[i].name for i in range(len(lp.attributes))
                     if lp.attributes[i].name not in _BOUNDS]:
            new.operation.attributes[name] = lp.attributes[name]
        inside = InsertionPoint(new.body)
        at = new.induction_variable
        if lb + t0 * step:
            with inside, lp.location:
                at = Operation.create("affine.apply", results=[at.type], operands=[at], attributes={
                    "map": AffineMapAttr.get(AffineMap.get(1, 0, [AffineAddExpr.get(
                        AffineDimExpr.get(0), AffineConstantExpr.get(lb + t0 * step))]))}).results[0]
        m = {iv: at, **dict(zip(carried, list(new.inner_iter_args)))}
        folded = {o: c for o, c in zip(sels, cls) if c is not None}
        for o in ops:
            if o in folded:
                m[o.results[0]] = m.get(o.operands[1 if folded[o] else 2],
                                        o.operands[1 if folded[o] else 2])
                continue
            c = o.clone(inside).operation
            for n, v in enumerate(list(c.operands)):
                if v in m:
                    c.operands[n] = m[v]
            m.update(zip(o.results, c.results))
        with inside, lp.location:
            affine.AffineYieldOp([m.get(v, v) for v in term.operands])
        inits = list(new.results)
    for old, v in zip(lp.results, inits):
        old.replace_all_uses_with(v)
    lp.operation.erase()


# Folds whose answer does not change when the same operand is folded in again.
_IDEMPOTENT = ("arith.maxnumf", "arith.minnumf", "arith.maximumf", "arith.minimumf",
               "arith.maxsi", "arith.minsi", "arith.maxui", "arith.minui", "arith.andi", "arith.ori")


def _hoist_idempotent_folds(lp):
    """A loop value `lp` carries that each trip folds, by a chain of one _IDEMPOTENT op and
    nothing else, with values made outside the loop -- a running max of a constant, where a mask
    settled a run of trips to it -- folded with them once before the loop instead: the loop then
    carries it as is."""
    from mlir.ir import InsertionPoint
    from ...passes.utils.ir_utils import loop_bounds, producer
    # Folding before the loop is what the loop does only if it runs at least once.
    b = loop_bounds(lp)
    if b is None or None in b or b[2] <= 0 or b[1] <= b[0]:
        return
    body = lp.regions[0].blocks[0]
    term = [o.operation for o in body.operations][-1]
    n_in = len(lp.operands) - len(lp.results)
    for k, (x, y) in enumerate(zip(list(body.arguments)[1:], list(term.operands))):
        chain, v = [], y
        while v != x:
            f = producer(v)
            if (f is None or f.name not in _IDEMPOTENT or f.block != body
                    or (chain and f.name != chain[0].name) or len(list(v.uses)) != 1):
                chain = None
                break
            inner = [w for w in f.operands if not _defined_outside(w, body)]
            if len(inner) != 1 or len(list(f.operands)) != 2:
                chain = None
                break
            chain.append(f)
            v = inner[0]
        if not chain or len(list(x.uses)) != 1:
            continue
        acc = lp.operands[n_in + k]
        for f in reversed(chain):
            g = f.clone(InsertionPoint(lp)).operation
            inside = [j for j, w in enumerate(f.operands) if not _defined_outside(w, body)][0]
            g.operands[inside] = acc
            acc = g.results[0]
        lp.operands[n_in + k] = acc
        term.operands[k] = x
        for f in chain:
            f.erase()


def _int_range(v, lanes, depth=0, fixed=None):
    """(least, most) that the integer or index value `v`, or every element of the vector `v`,
    can hold, by what makes it: constants, loop counters of constant bounds, the lane number and
    vector.step, and index casts, additions, subtractions and products by constants of those;
    None for anything else."""
    from mlir.ir import (BlockArgument, DenseElementsAttr, IntegerAttr, VectorType)
    from ...passes.utils.ir_utils import loop_bounds, op_of, producer
    if fixed and v in fixed:
        return fixed[v]
    if depth > 32:
        return None
    if isinstance(v, BlockArgument):
        owner = op_of(v.owner.owner) if v.owner.owner is not None else None
        if owner is None or owner.name not in ("affine.for", "scf.for") or v.arg_number != 0:
            return None
        b = loop_bounds(owner)
        if b is None or None in b or b[2] <= 0 or b[1] <= b[0]:
            return None
        return (b[0], b[0] + (-(-(b[1] - b[0]) // b[2]) - 1) * b[2])
    p = producer(v)
    if p is None:
        return None
    r = lambda w: _int_range(w, lanes, depth + 1, fixed)  # noqa: E731
    if p.name == "arith.constant":
        a = p.attributes["value"]
        try:
            n = IntegerAttr(a).value
        except (TypeError, ValueError):
            try:
                d = DenseElementsAttr(a)
                if not d.is_splat:
                    return None
                n = IntegerAttr(d.get_splat_value()).value
            except (TypeError, ValueError):
                return None
        return (n, n)
    if p.name in ("arith.index_cast", "arith.index_castui", "vector.broadcast"):
        return r(p.operands[0])
    if p.name == "vector.step":
        return (0, VectorType(p.results[0].type).shape[0] - 1)
    if p.name == "pytorchsim.lane_id":
        return (0, lanes - 1)
    if p.name in ("arith.addi", "arith.subi", "arith.muli"):
        x, y = r(p.operands[0]), r(p.operands[1])
        if x is None or y is None:
            return None
        if p.name == "arith.addi":
            return (x[0] + y[0], x[1] + y[1])
        if p.name == "arith.subi":
            return (x[0] - y[1], x[1] - y[0])
        c = [x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]]
        return (min(c), max(c))
    return None


def _broadcast_by_register(op, vlen):
    """`op`, a vector.broadcast of a vector of one element to a 1-D vector of more than one
    register, rewritten as the broadcast to one register, doubled by concatenation (vector.shuffle
    of a vector with itself) to the width: llc gathers one register and copies it, where a gather
    over the whole group takes a micro-op for every element of every register."""
    from mlir.ir import InsertionPoint, Operation, VectorType
    from ...passes.utils.ir_utils import elem_bits
    src, res = op.operands[0], op.results[0]
    if not isinstance(src.type, VectorType) or VectorType(src.type).shape != [1]:
        return
    rt = VectorType(res.type)
    if len(rt.shape) != 1:
        return
    per = vlen // elem_bits(rt.element_type)
    n = rt.shape[0]
    if n <= per or n % per or (n // per) & (n // per - 1):
        return
    with InsertionPoint(op), op.location:
        got = Operation.create("vector.broadcast", results=[VectorType.get([per], rt.element_type)],
                               operands=[src]).results[0]
        w = per
        while w < n:
            got = _shuffle(got, VectorType.get([2 * w], rt.element_type), list(range(2 * w)))
            w *= 2
    res.replace_all_uses_with(got)
    op.erase()


def max_lmul(loop):
    """The widest LMUL a vector loop whose own body holds an f32 math.exp may take: the widest whose DEPTH + 1
    groups the EUP holds while schedule_vpu_stream pipelines it (eup_registers), or None when every LMUL fits,
    the machine does not say what the EUP holds, or the loop holds no such exp."""
    from ...driver import machine
    from .schedule_vpu_stream import eup_registers
    return eup_lmul(eup_registers(machine.subtarget())) if is_exp_loop(loop) else None


def is_exp_loop(loop):
    """Is `loop` a vector loop whose own body holds an f32 math.exp this lowering sends to the EUP?"""
    from mlir.dialects import affine, math
    from ...passes.utils.vector_loops import VECTOR_LOOP
    if (math.ExpOp.OPERATION_NAME not in SFU_FUNCTIONS or not isinstance(loop.opview, affine.AffineForOp)
            or VECTOR_LOOP not in loop.attributes):
        return False
    return any(isinstance(o.opview, math.ExpOp) and _is_f32_vector(o.results[0].type)
               for o in loop.regions[0].blocks[0].operations)


def eup_lmul(holds):
    """The widest LMUL whose DEPTH + 1 groups an EUP holding `holds` registers takes, or None when every LMUL
    fits or `holds` is unsaid (None)."""
    from ...driver.machine import ISA_MAX_LMUL
    from .schedule_vpu_stream import DEPTH, UNROLL
    fits = [1 << k for k in range(ISA_MAX_LMUL.bit_length()) if (DEPTH + 1) * UNROLL * (1 << k) <= (holds or 0)]
    return fits[-1] if fits and fits[-1] != ISA_MAX_LMUL else None
