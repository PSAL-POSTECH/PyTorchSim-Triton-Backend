"""bank_layout: what a LANE's bank holds for a value."""

import functools

from ...dialect.pytorchsim import BankViewOp, effects_on, is_bank_view, is_retyped_view, view_root
from .ir_utils import (
    int_array, is_get_global, is_global, is_view, name_of, op_of, strides_and_offset, view_source)

SPAD_SPACE = 1

#: `pytorchsim.distributed_reduction`(src, dst) {by}: the distributed axis folded away into one bank,
#: each column combined over it by `by` (an arith op) with what the destination held. b07 mints it
#: from a reduce that leaves one bank; the target's LOWER_PASSES lower it.
DISTRIBUTED_REDUCTION_OP = "pytorchsim.distributed_reduction"
REDUCTION_BY = "by"

#: A discardable marker on a target's own move whose pairs leave every lane holding the same row:
#: verify-lane-uniformity takes such a write into a UNIFORM buffer, as it takes a DISTRIBUTED_REDUCTION_OP.
EVERY_LANE = "pytorchsim.every_lane"


def need_per_lane(numel, lane_extent, vectorlane):
    """Elements one lane actually HOLDS of a `numel`-element tile: its rows, once per lane-full of
    a lane axis longer than the machine -- a last, partial lane-full as deep as a whole one, since
    the lanes it leaves idle hold nothing else."""
    if lane_extent and lane_extent > vectorlane:
        return lane_fulls(lane_extent, vectorlane) * (numel // lane_extent)
    used = lane_extent or vectorlane
    return -(-numel // max(used, 1))


def lane_fulls(extent, vectorlane):
    """How many times a lane axis of `extent` wraps the machine's lanes -- the last time partially,
    where `extent` is no multiple of them."""
    return -(-extent // vectorlane)


def is_one_bank(shape):
    """Does a tile of `shape` sit in one bank?"""
    return not shape or shape[0] == 1


def is_uniform(value):
    """Is `value` UNIFORM, its buffer held by one lane: its view_root's shape is one bank, whatever
    a view's own type says."""
    from mlir.ir import MemRefType
    return is_one_bank(list(MemRefType(view_root(value).type).shape))


def bank_layout(value):
    """`(strides, offset)` a lane's bank has for `value`: its type's, since bufferize lays every
    scratchpad buffer out lane-first."""
    from mlir.ir import ShapedType
    dyn = ShapedType.get_dynamic_stride_or_offset()
    v = value
    while True:
        so = strides_and_offset(v)
        if so is not None and dyn not in so[0] and so[1] != dyn:
            return so
        o = op_of(getattr(v, "owner", None))
        if name_of(o) != "memref.cast":
            return row_major(list(ShapedType(value.type).shape)), 0
        v = o.operands[0]


def bank_strides(value):
    """The strides a lane's bank has for `value`, `bank_layout`'s."""
    return bank_layout(value)[0]


def bank_offset(value):
    """Where `value`'s first element sits inside a lane's bank, `bank_layout`'s."""
    v = value
    while True:
        o = op_of(getattr(v, "owner", None))
        if name_of(o) == "memref.subview":
            src = o.operands[0]
            offs = int_array(o.attributes["static_offsets"])
            if (view_lanes(src) or 1) > 1 and offs[0] != 0:
                raise ValueError(
                    f"a subview starts at {offs[0]} along the lane axis "
                    f"0 of its source, which selects a BANK rather than an "
                    f"address inside one -- the lane is chosen by the hardware and "
                    f"there is no index for it: {value}")
        if not is_view(o):
            return bank_layout(value)[1]
        v = view_source(o)


def _at_function_top(op):
    got = getattr(op, "operation", None)
    if got is None:
        return False
    return got.parent is not None and name_of(got.parent) == "func.func"


def bank_cast(value, offset, shape, strides):
    """`value` retyped to `(offset, shape, strides)` by a pytorchsim.bank_view, made ONCE
    beside a buffer the whole function can see; the result type is all the view states."""
    from mlir.ir import InsertionPoint, MemRefType, StridedLayoutAttr
    mt = MemRefType(value.type)
    shape, strides = list(shape), list(strides)
    ty = MemRefType.get(shape, mt.element_type,
                        StridedLayoutAttr.get(offset, strides), mt.memory_space)

    def mint():
        from mlir.ir import Operation
        return Operation.create(BankViewOp.OPERATION_NAME, results=[ty], operands=[value]).results[0]

    owner = op_of(getattr(value, "owner", None))
    if owner is None or not _at_function_top(owner):
        return mint()
    got = _cast_beside(owner, value, ty)
    if got is None:
        with InsertionPoint.after(owner):
            got = mint()
    return got


def _cast_beside(owner, value, ty):
    """The bank view of `value` as `ty` in the run of bank views of it right after `owner`, read
    from the IR, whichever kind made it (a view is the same view); None when that run holds none."""
    after = False
    for got in owner.block.operations:
        got = op_of(got)
        if not after:
            after = got == owner
            continue
        if not is_bank_view(got) or len(got.operands) != 1 or got.operands[0] != value:
            return None
        if got.results[0].type == ty:
            return got.results[0]
    return None


def row_major(shape):
    """Row-major strides for `shape`: the minor axis gets 1."""
    out, acc = [], 1
    for d in reversed(shape):
        out.insert(0, acc)
        acc *= d
    return out


def build_a_scratchpad_alloc(func, shape, elem):
    """A scratchpad buffer of `shape` x `elem`: a memref.alloc in space 1 at the top of `func`'s entry
    block, freed just before its terminator -- the whole function's life, as pytorchsim-bufferize leaves every
    buffer."""
    from mlir.ir import InsertionPoint, IntegerAttr, IntegerType, MemRefType
    from mlir.dialects import memref
    ty = MemRefType.get(list(shape), elem,
                        memory_space=IntegerAttr.get(IntegerType.get_signless(64), SPAD_SPACE))
    entry = func.regions[0].blocks[0]
    with InsertionPoint.at_block_begin(entry):
        buf = memref.AllocOp(ty, [], []).result
    with InsertionPoint.at_block_terminator(entry):
        memref.DeallocOp(buf)
    return buf


def scratchpad_lives_left(module):
    """Every op under `module` that allocates or frees a scratchpad buffer (its effects, never its
    name): what finalize-memref-to-llvm would turn into a malloc or a free."""
    from .ir_utils import walk
    left = []
    for op in walk(module.operation):
        if any(in_scratchpad(r.type) and "allocate" in effects_on(op, r) for r in op.results) or \
                any(in_scratchpad(v.type) and "free" in effects_on(op, v) for v in op.operands):
            left.append(op)
    return left


def lanes_of_shape(shape):
    """How many lanes a scratchpad buffer of `shape` spreads over: 1 in one bank, else its axis 0's
    extent -- the one reading both the reservation and every reader of a buffer's type share."""
    return 1 if is_one_bank(shape) else shape[0]


@functools.lru_cache(maxsize=4096)
def in_scratchpad(ty):
    """Is `ty` a memref of the scratchpad's memory space? Kept per type: every pass asks it of every
    operand."""
    from mlir.ir import IntegerAttr, MemRefType
    try:
        mt = MemRefType(ty)
    except (TypeError, ValueError):
        return False
    return mt.memory_space is not None and IntegerAttr(mt.memory_space).value == SPAD_SPACE


def is_a_scratchpad_buffer(v):
    """Is `v` a buffer itself, not a view of one: what a memref.get_global names, or a scratchpad
    memref the op making it states it allocates (its effects, never its name)?"""
    from mlir.ir import BlockArgument
    if v is None or isinstance(v, BlockArgument):
        return False
    o = op_of(v.owner)
    if is_get_global(o):
        return True
    return in_scratchpad(v.type) and "allocate" in effects_on(o, v)


def lanes_of_a_buffer(op):
    """`lane_extent` for a scratchpad buffer, off the type a `memref.global` declares or an op that
    allocates one makes (is_a_scratchpad_buffer); None for any other op."""
    from mlir.ir import TypeAttr
    op = op_of(op)
    if op is not None and len(op.results) == 1 and not is_get_global(op) \
            and is_a_scratchpad_buffer(op.results[0]):
        return lane_extent(op.results[0])
    if not is_global(op) or "type" not in op.attributes:
        return None
    try:
        return _lane_extent_of_type(TypeAttr(op.attributes["type"]).value)
    except (TypeError, ValueError):
        return None


def lane_split(shape, one_lane, lanes):
    """(trips over axis 0, machine lanes that carry data) for a tile of `shape`; a one-lane
    (UNIFORM) tile walks all of axis 0 as trips in one lane."""
    if one_lane:
        return (shape[0] if shape else 1), 1
    dim = shape[0]
    if dim > lanes and dim % lanes:
        raise ValueError(
            f"tile dim0={dim} is neither <= {lanes} lanes nor a multiple "
            f"of them, and a scalar fallback would read the banked scratchpad "
            f"linearly")
    return (dim // lanes if dim > lanes else 1), min(dim, lanes)


def lane_extent(value):
    """How many lanes the buffer `value` is a view of spreads over: lanes_of_shape of its view_root's
    type, None when it is no scratchpad memref."""
    return _lane_extent_of_type(view_root(value).type)


def view_lanes(value):
    """How many lanes `value`'s own elements ride, off its own type: a view's question, as bank_depth
    is; lane_extent asks it of the buffer."""
    return _lane_extent_of_type(value.type)


def _lane_extent_of_type(ty):
    from mlir.ir import MemRefType
    return lanes_of_shape(list(MemRefType(ty).shape)) if in_scratchpad(ty) else None


def why_no_lane_axis(value):
    """Why `lane_extent` answered None, for a caller's own refusal."""
    return "%s is not a scratchpad buffer, so no lane rides it" % value.type


def bank_view(v, lanes):
    """`v` as the `[1, in-bank]` view a lane's bank really has, a pytorchsim.bank_view,
    from the depth `v` starts at (a view's own offset -- the cast's is absolute), depth as
    need_per_lane defines it."""
    depth = bank_depth(v, lanes)
    if depth is None:
        raise NotImplementedError(
            "a lane's rows of %s are not a run at one step" % v.type)
    return bank_cast(v, bank_offset(v), [1, depth], [depth, 1])


def bank_depth(v, lanes):
    """How many elements one lane's bank holds of `v`, as one run from where `v` starts."""
    from mlir.ir import MemRefType
    shape = list(MemRefType(v.type).shape)
    numel = 1
    for d in shape:
        numel *= d
    owner = op_of(getattr(v, "owner", None))
    if name_of(owner) == "memref.subview" and not _is_one_run_per_lane(owner, lanes):
        return None
    return need_per_lane(numel, view_lanes(v), lanes)


def _is_one_run_per_lane(subview, lanes):
    from mlir.ir import MemRefType
    src, res = subview.operands[0], subview.results[0]
    shape = list(MemRefType(res.type).shape)
    so = strides_and_offset(res)
    if (so is None or len(shape) != len(MemRefType(src.type).shape) or not shape
            or int_array(subview.attributes["static_offsets"])[0] != 0
            or shape[0] > lanes):
        return False
    run = 1
    for size, stride in reversed(list(zip(shape[1:], so[0][1:]))):
        if size != 1 and stride != run:
            return False
        run *= size
    return True


def lower_bank_views(module):
    """Every bank view (an op of the BankView trait) written back as the `memref.reinterpret_cast`
    to its result type, whose strided layout is all it states."""
    from mlir.ir import InsertionPoint, MemRefType, ShapedType
    from mlir.dialects import memref
    from .ir_utils import walk
    made = 0
    for op in list(walk(module.operation)):
        if not is_bank_view(op):
            continue
        res = op.results[0]
        shape = list(ShapedType(res.type).shape)
        so = strides_and_offset(res)
        if so is None or ShapedType.get_dynamic_stride_or_offset() in (*so[0], so[1]):
            raise NotImplementedError(f"{op.name}: {res.type} states no static strided layout")
        strides, offset = so
        with InsertionPoint(op):
            got = memref.ReinterpretCastOp(
                MemRefType(res.type), op.operands[0], [], [], [],
                [offset], shape, strides).result
        res.replace_all_uses_with(got)
        op.erase()
        made += 1
    return made


def lower_retyped_views(module):
    """Every retyped bank view (an op of the RetypedView trait) written back as a change of type alone:
    lowered, a memref of either element is one descriptor, and the result's offset and strides are
    its type's."""
    from mlir.ir import InsertionPoint, Operation
    from .ir_utils import walk
    made = 0
    for op in list(walk(module.operation)):
        if not is_retyped_view(op):
            continue
        res = op.results[0]
        with InsertionPoint(op):
            got = Operation.create("builtin.unrealized_conversion_cast", results=[res.type],
                                   operands=[op.operands[0]]).results[0]
        res.replace_all_uses_with(got)
        op.erase()
        made += 1
    return made
