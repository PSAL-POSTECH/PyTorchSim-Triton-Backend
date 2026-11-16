"""What a tts access and the pointer it is made through say: the descriptor (sizes, strides, offsets, wrap, step),
the mask box, the gather's own facts, and the exceptions of the passes that read them. Shared by
pytorchsim-resolve-masks and convert-tts-to-pytorchsim; each answer is read from the op, never decided here."""

from ...contract.stop import Reason, Refusal
from .ir_utils import (
    GS_BASE_SEG, GS_MASK_SEG, GS_OFFSET_SEG, GS_SHAPE_SEG, GS_STRIDE_SEG, elementwise_generic, int_array,
    is_shaped, name_of, op_of, operand_segments)

KDYN = -9223372036854775808
GATHER_PTR_OP = "tts.make_gather_scatter_tptr"
TPTR_BASE_OFFSET_SEG = 4
TPTR_FLAT_SHAPE_SEG = 5
MAX_WRAP_PIECES = 12


class Fatal(Exception):
    pass


class BaseUntraced(Fatal, Refusal):
    """A make_tptr base that does not trace to a function argument."""
    reason = Reason.POINTER_BASE_UNTRACED


def get_dram_arg(base, tensor_args):
    """A pointer base -> the memref arg behind its unrealized_conversion_cast, and that arg's index among
    `tensor_args`."""
    cast = getattr(base, "owner", None)
    cast = op_of(cast)
    if cast is None or cast.name != "builtin.unrealized_conversion_cast":
        raise Fatal(f"make_tptr base is not an unrealized_conversion_cast "
                    f"from a memref arg: {base}")
    memref_arg = get_through_casts(cast.operands[0])
    for i, a in enumerate(tensor_args):
        if a == memref_arg:
            return memref_arg, i
    raise BaseUntraced(f"make_tptr base does not trace to a function arg: "
                       f"{memref_arg}")


def is_buffer(ty):
    """Is `ty` a memref, ranked or not: a tensor argument's type, where a scalar's is no memref."""
    from mlir.ir import MemRefType, UnrankedMemRefType
    return isinstance(ty, (MemRefType, UnrankedMemRefType))


def get_through_casts(value):
    while not str(value.type).startswith("memref<"):
        up = op_of(getattr(value, "owner", None))
        if up is None or up.name != "builtin.unrealized_conversion_cast":
            break
        value = up.operands[0]
    return value


def get_int(v):
    from mlir.ir import IntegerAttr
    if isinstance(v, int):
        return v
    owner = op_of(getattr(v, "owner", None))
    if name_of(owner) != "arith.constant":
        return None
    return IntegerAttr(owner.attributes["value"]).value


def get_ptr_op(op):
    return op_of(op.operands[0].owner)


def get_descriptor(make_tptr_op):
    a = make_tptr_op.attributes
    sizes = int_array(a["sizes"])
    strides = int_array(a["static_strides"])
    static_offsets = int_array(a["static_offsets"])
    shape = int_array(a["static_shape"])
    seg = int_array(a["operandSegmentSizes"])
    ops = list(make_tptr_op.operands)
    p = 1
    stride_ops = ops[p:p + seg[1]]; p += seg[1]
    offset_ops = ops[p:p + seg[2]]; p += seg[2]
    shape_ops = ops[p:p + seg[3]]; p += seg[3]
    wrap = resolve_strides(shape, shape_ops)
    resolved = resolve_strides(strides, stride_ops)
    odyn = iter(offset_ops)
    offsets = [next(odyn) if o == KDYN else o for o in static_offsets]
    return sizes, resolved, offsets, wrap


def get_steps(make_tptr_op):
    a = make_tptr_op.attributes
    n = len(int_array(a["sizes"]))
    return int_array(a["static_step"]) if "static_step" in a else [0] * n


def get_base_offset(make_tptr_op):
    return get_operand_at_segment(make_tptr_op, TPTR_BASE_OFFSET_SEG)


def get_flat_shape(make_tptr_op):
    return get_operand_at_segment(make_tptr_op, TPTR_FLAT_SHAPE_SEG)


def get_operand_at_segment(op, seg):
    segs = operand_segments(op)
    if seg >= len(segs) or not segs[seg]:
        return None
    return list(op.operands)[sum(segs[:seg])]


def split_access_operands(access_op):
    ops = list(access_op.operands)
    seg = int_array(access_op.attributes["operandSegmentSizes"])
    if name_of(access_op) == "tts.load":
        i = seg[0]
        dims = ops[i:i + seg[1]]; i += seg[1]
        lows = ops[i:i + seg[2]]; i += seg[2]
        other = ops[i] if seg[3] else None; i += seg[3]
        return dims, lows, other, (ops[i] if seg[4] else None)
    i = seg[0] + seg[1]
    dims = ops[i:i + seg[2]]; i += seg[2]
    lows = ops[i:i + seg[3]]; i += seg[3]
    return dims, lows, None, (ops[i] if seg[4] else None)


def resolve_strides(static_strides, stride_ops):
    it = iter(stride_ops)
    out = []
    for s in static_strides:
        if s != KDYN:
            out.append(s)
            continue
        v = next(it)
        c = get_int(v)
        out.append(v if c is None else c)
    return out


def get_dram_strides(sizes, strides):
    out, acc = [], 1
    for size, stride in zip(reversed(sizes), reversed(strides)):
        if size == 1 and isinstance(stride, int) and stride == 0:
            out.append(acc)
        else:
            out.append(stride)
        acc *= size
    out.reverse()
    return out


def build_select(mask, a, b, loc, mask_axes=None):
    from mlir.ir import AffineExpr, AffineMap, AffineMapAttr, ShapedType
    from mlir.dialects import arith

    if not is_shaped(a):
        return arith.SelectOp(mask, a, b, loc=loc).result
    st = ShapedType(a.type)
    shape, elem = list(st.shape), st.element_type
    ident = AffineMapAttr.get(AffineMap.get_identity(len(shape)))
    mask_map = ident if mask_axes is None else AffineMapAttr.get(AffineMap.get(
        len(shape), 0, [AffineExpr.get_dim(d) for d in mask_axes]))
    return elementwise_generic(
        [mask, a, b], elem,
        lambda m, x, y: arith.SelectOp(m, x, y).result,
        loc, shape, maps=[mask_map] + [ident] * 3)


def get_mask_window(access_op, sizes):
    """Per axis the (low, high) of the access's static mask, whole axis where it has none."""
    if "static_mask_dims" not in access_op.attributes:
        return [(0, n) for n in sizes]
    hi = int_array(access_op.attributes["static_mask_dims"])
    lo = (int_array(access_op.attributes["static_mask_lows"])
          if "static_mask_lows" in access_op.attributes else [0] * len(hi))
    if len(hi) != len(sizes) or len(lo) != len(sizes):
        return [(0, n) for n in sizes]
    return list(zip(lo, hi))


def get_gather_ptr_facts(gs_op):
    a = gs_op.attributes
    seg = int_array(a["operandSegmentSizes"])
    ops = list(gs_op.operands)
    p = 2
    stride_ops = ops[p:p + seg[GS_STRIDE_SEG]]; p += seg[GS_STRIDE_SEG]
    offset_ops = ops[p:p + seg[GS_OFFSET_SEG]]; p += seg[GS_OFFSET_SEG]
    shape_ops = ops[p:p + seg[GS_SHAPE_SEG]]
    strides = resolve_strides(int_array(a["static_strides"]), stride_ops)
    offsets = resolve_strides(int_array(a["static_offsets"]), offset_ops)
    wrap = resolve_strides(int_array(a["static_shape"]), shape_ops)
    return (int_array(a["sizes"]), offsets, strides, wrap,
            [d for d, w in enumerate(wrap)
             if not (isinstance(w, int) and w == 0)],
            get_operand_at_segment(gs_op, GS_MASK_SEG),
            get_operand_at_segment(gs_op, GS_BASE_SEG))


def build_read_park(base, stride, elem, loc):
    """-floor(base / stride) as `elem`: base + park * stride is then in [0, stride), the tensor's first row,
    where a bare 0 would read at base, before the tensor when base < 0. 0 without a base or a stride."""
    from mlir.dialects import arith
    from mlir.ir import IndexType, IntegerAttr
    if base is None or stride == 0:
        return arith.ConstantOp(elem, 0, loc=loc).result
    index = IndexType.get()
    if isinstance(stride, int):
        stride = arith.ConstantOp(index, IntegerAttr.get(index, stride), loc=loc).result
    rows = arith.FloorDivSIOp(base, stride, loc=loc).result
    zero = arith.ConstantOp(index, IntegerAttr.get(index, 0), loc=loc).result
    return arith.IndexCastOp(elem, arith.SubIOp(zero, rows, loc=loc).result, loc=loc).result
