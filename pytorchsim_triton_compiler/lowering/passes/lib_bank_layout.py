"""lib_bank_layout: what a LANE's bank holds for a value."""

from ...dialects import pytorchsim_rvv, pytorchsim_tpu
from ...dialects.pytorchsim import effects_on, view_root
from .lib_ir import VIEW_OPS, int_array, name_of, op_of, strides_and_offset
from .lib_lanes import lane_fulls, need_per_lane  # noqa: F401 -- defined with no IR

SPAD_SPACE = 1

#: `pytorchsim.distributed_reduction`(src, dst) {by}: the distributed axis folded away into one bank,
#: each column combined over it by `by` (an arith op) with what the destination held. b09 mints it
#: from a reduce that leaves one bank; the pack fills the lower-distributed-reduction slot with its
#: unit for it.
DISTRIBUTED_REDUCTION_OP = "pytorchsim.distributed_reduction"
REDUCTION_BY = "by"

#: A discardable marker on a pack's own move whose pairs leave every lane holding the same row: b15
#: takes such a write into a UNIFORM buffer, as it takes a DISTRIBUTED_REDUCTION_OP.
EVERY_LANE = "pytorchsim.every_lane"


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
        if name_of(o) not in VIEW_OPS:
            return bank_layout(value)[1]
        v = o.operands[0]


def _at_function_top(op):
    got = getattr(op, "operation", None)
    if got is None:
        return False
    return got.parent is not None and name_of(got.parent) == "func.func"


#: The bank views each target kind makes (its dialect's own); convert-pytorchsim-to-llvm lowers them all.
BANK_VIEW_OPS = (pytorchsim_tpu.BANK_VIEW, pytorchsim_rvv.BANK_VIEW)


def bank_cast(value, offset, shape, strides, view_op):
    """`value` retyped to `(offset, shape, strides)` by `view_op` (one of BANK_VIEW_OPS), made ONCE
    beside a buffer the whole function can see."""
    from mlir.ir import InsertionPoint, MemRefType, StridedLayoutAttr
    mt = MemRefType(value.type)
    shape, strides = list(shape), list(strides)
    ty = MemRefType.get(shape, mt.element_type,
                        StridedLayoutAttr.get(offset, strides), mt.memory_space)

    def mint():
        from mlir.ir import DenseI64ArrayAttr, IntegerAttr, IntegerType, Operation
        i64 = IntegerType.get_signless(64)
        return Operation.create(
            view_op, results=[ty], operands=[value],
            attributes={"offset": IntegerAttr.get(i64, offset),
                        "sizes": DenseI64ArrayAttr.get(shape),
                        "strides": DenseI64ArrayAttr.get(strides)}).results[0]

    owner = op_of(getattr(value, "owner", None))
    if owner is None or not _at_function_top(owner):
        return mint()
    got = _cast_beside(owner, value, offset, shape, strides)
    if got is None:
        with InsertionPoint.after(owner):
            got = mint()
    return got


def _cast_beside(owner, value, offset, shape, strides):
    """The cast of `value` to `(offset, shape, strides)` in the run of casts of it right after
    `owner`, read from the IR, whichever kind made it (a view is the same view); None when that run
    holds none."""
    from mlir.ir import IntegerAttr
    ops = [op_of(o) for o in owner.block.operations]
    for got in ops[ops.index(owner) + 1:]:
        if name_of(got) not in BANK_VIEW_OPS or len(got.operands) != 1 or got.operands[0] != value:
            return None
        if (IntegerAttr(got.attributes["offset"]).value == offset
                and int_array(got.attributes["sizes"]) == shape
                and int_array(got.attributes["strides"]) == strides):
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
    block, freed just before its terminator -- the whole function's life, as a06 leaves every buffer."""
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
    from .lib_ir import walk
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


def in_scratchpad(ty):
    """Is `ty` a memref of the scratchpad's memory space?"""
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
    if o.name == "memref.get_global":
        return True
    return in_scratchpad(v.type) and "allocate" in effects_on(o, v)


def lanes_of_a_buffer(op):
    """`lane_extent` for a scratchpad buffer, off the type a `memref.global` declares or an op that
    allocates one makes (is_a_scratchpad_buffer); None for any other op."""
    from mlir.ir import TypeAttr
    op = op_of(op)
    if op is not None and len(op.results) == 1 and op.name != "memref.get_global" \
            and is_a_scratchpad_buffer(op.results[0]):
        return lane_extent(op.results[0])
    if op is None or op.name != "memref.global" or "type" not in op.attributes:
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


def bank_view(v, lanes, view_op):
    """`v` as the `[1, in-bank]` view a lane's bank really has, by `view_op` (one of BANK_VIEW_OPS),
    from the depth `v` starts at (a view's own offset -- the cast's is absolute), depth as
    need_per_lane defines it."""
    depth = bank_depth(v, lanes)
    if depth is None:
        raise NotImplementedError(
            "a lane's rows of %s are not a run at one step" % v.type)
    return bank_cast(v, bank_offset(v), [1, depth], [depth, 1], view_op)


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
    """Every bank view (BANK_VIEW_OPS) written back as the `memref.reinterpret_cast` it describes."""
    from mlir.ir import (InsertionPoint, IntegerAttr, MemRefType, ShapedType)
    from mlir.dialects import memref
    from .lib_ir import int_array, walk
    made = 0
    for op in list(walk(module.operation)):
        if name_of(op) not in BANK_VIEW_OPS:
            continue
        res = op.results[0]
        shape = list(ShapedType(res.type).shape)
        offset = IntegerAttr(op.attributes["offset"]).value
        strides = int_array(op.attributes["strides"])
        with InsertionPoint(op):
            got = memref.ReinterpretCastOp(
                MemRefType(res.type), op.operands[0], [], [], [],
                [offset], shape, strides).result
        res.replace_all_uses_with(got)
        op.erase()
        made += 1
    return made
