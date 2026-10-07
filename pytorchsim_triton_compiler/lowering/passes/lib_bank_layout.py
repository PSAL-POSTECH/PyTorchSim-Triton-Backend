"""lib_bank_layout: what a LANE's bank holds for a value. The layout is in the type: axis 0
rides the lanes, the rest are row-major in the bank, and a first axis of one is one bank."""

from .lib_ir import VIEW_OPS, int_array, name_of, op_of, strides_and_offset

MIN_VEC = 2

SPAD_SPACE = 1


def need_per_lane(numel, lane_extent, vectorlane):
    """Elements one lane actually HOLDS of a `numel`-element tile."""
    used = min(lane_extent, vectorlane) if lane_extent else vectorlane
    return -(-numel // max(used, 1))


def reserve_per_lane(numel, lane_extent, vectorlane):
    """Elements to RESERVE in each lane's bank: the need at MIN_VEC, the narrowest
    vector this machine moves in or out of a bank."""
    n = need_per_lane(numel, lane_extent, vectorlane)
    return -(-n // MIN_VEC) * MIN_VEC


def is_one_bank(shape):
    """Does a tile of `shape` sit in one bank? It does when it has no axis or its axis 0 holds
    one element; otherwise axis 0 rides the lanes."""
    return not shape or shape[0] == 1


def bank_layout(value):
    """`(strides, offset)` a lane's bank has for `value`: its type's, since bufferize lays every
    scratchpad buffer out lane-first. A cast that forgets them is read through to its source."""
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
    """Where `value`'s first element sits inside a lane's bank, `bank_layout`'s. A subview starting
    past row 0 of a banked source selects a BANK, which no address says: ValueError."""
    v = value
    while True:
        o = op_of(getattr(v, "owner", None))
        if name_of(o) == "memref.subview":
            src = o.operands[0]
            offs = int_array(o.attributes["static_offsets"])
            if (lanes_of(src) or 1) > 1 and offs[0] != 0:
                raise ValueError(
                    f"a subview starts at {offs[0]} along the lane axis "
                    f"0 of its source, which selects a BANK rather than an "
                    f"address inside one -- the lane is chosen by the hardware and "
                    f"there is no index for it: {value}")
        if name_of(o) not in VIEW_OPS:
            return bank_layout(value)[1]
        v = o.operands[0]


_BANK_CASTS = {}


def index_bank_casts(module):
    """Every cast already beside a buffer, by what it re-describes. ONE WALK at a
    pass's entry, so minting one never searches the IR -- that search read a block
    the same pass was growing, which is quadratic in the casts it mints."""
    _BANK_CASTS.clear()
    if module is None:
        return
    from mlir.ir import IntegerAttr
    from .lib_ir import funcs
    for func in funcs(module):
        ops = list(func.regions[0].blocks[0].operations)
        for k, o in enumerate(ops):
            owner = op_of(o)
            if owner is None or not len(owner.results):
                continue
            value = owner.results[0]
            for nxt in ops[k + 1:]:
                got = op_of(nxt)
                if (name_of(got) != BANK_VIEW_OP or len(got.operands) != 1
                        or got.operands[0] != value):
                    break
                _BANK_CASTS.setdefault(
                    (value, IntegerAttr(got.attributes["offset"]).value,
                     tuple(int_array(got.attributes["sizes"])),
                     tuple(int_array(got.attributes["strides"]))),
                    got.results[0])


def _at_function_top(op):
    """Is `op` in the function's own body, and not in a region below it? Then
    everything the function does is after it, which is what lets one cast serve
    every use of that buffer."""
    got = getattr(op, "operation", None)
    if got is None:
        return False
    return got.parent is not None and name_of(got.parent) == "func.func"


BANK_VIEW_OP = "torchsim.bank_view"


def bank_cast(value, offset, shape, strides):
    """`value` retyped to `(offset, shape, strides)`, made ONCE beside a buffer the
    whole function can see. A CAST KEPT FROM A LOOP BODY DOES NOT DOMINATE THE
    NEXT, so only what the entry block holds is hoisted and shared."""
    from mlir.ir import InsertionPoint, MemRefType, StridedLayoutAttr
    mt = MemRefType(value.type)
    shape, strides = list(shape), list(strides)
    ty = MemRefType.get(shape, mt.element_type,
                        StridedLayoutAttr.get(offset, strides), mt.memory_space)

    def mint():
        from mlir.ir import DenseI64ArrayAttr, IntegerAttr, IntegerType, Operation
        i64 = IntegerType.get_signless(64)
        return Operation.create(
            BANK_VIEW_OP, results=[ty], operands=[value],
            attributes={"offset": IntegerAttr.get(i64, offset),
                        "sizes": DenseI64ArrayAttr.get(shape),
                        "strides": DenseI64ArrayAttr.get(strides)}).results[0]

    owner = op_of(getattr(value, "owner", None))
    if owner is None or not _at_function_top(owner):
        return mint()
    key = (value, offset, tuple(shape), tuple(strides))
    got = _BANK_CASTS.get(key)
    if got is None:
        with InsertionPoint.after(owner):
            got = mint()
        _BANK_CASTS[key] = got
    return got


def row_major(shape):
    """Row-major strides for `shape`: the minor axis gets 1. NOT A BANKED
    LAYOUT -- this is the DRAM image and the one-lane scratchpad case, where
    every element of the tile sits in one address space."""
    out, acc = [], 1
    for d in reversed(shape):
        out.insert(0, acc)
        acc *= d
    return out


def build_a_scratchpad_buffer(module_top, entry, shape, elem, name):
    """A scratchpad buffer: memref.global in space 1 + get_global at entry.
    THE ONE PLACE that knows how one is made, because getting it wrong is not a
    compile error -- a plain alloc lands on the heap and every lane reads it."""
    from mlir.ir import InsertionPoint, Operation, StringAttr, Type, TypeAttr
    from mlir.dialects import memref
    dims = "".join(f"{d}x" for d in shape)
    ty = Type.parse(f"memref<{dims}{elem}, {SPAD_SPACE}>")
    with InsertionPoint.at_block_begin(module_top):
        Operation.create("memref.global", results=[], operands=[],
                         attributes={"sym_name": StringAttr.get(name),
                                     "type": TypeAttr.get(ty)})
    with InsertionPoint.at_block_begin(entry):
        return memref.GetGlobalOp(ty, name).result


def _lanes_of_a_type(ty):
    """How many lanes a scratchpad memref type spreads over: 1 in one bank, else its axis 0's
    extent. None for any other type."""
    from mlir.ir import IntegerAttr, MemRefType
    try:
        mt = MemRefType(ty)
    except (TypeError, ValueError):
        return None
    if mt.memory_space is None or IntegerAttr(mt.memory_space).value != SPAD_SPACE:
        return None
    shape = list(mt.shape)
    return 1 if is_one_bank(shape) else shape[0]


def lanes_of_a_buffer(op):
    """`lanes_of` for a scratchpad `memref.global`, off its declared type; None for any other op."""
    from mlir.ir import TypeAttr
    if op is None or op.name != "memref.global" or "type" not in op.attributes:
        return None
    try:
        return _lanes_of_a_type(TypeAttr(op.attributes["type"]).value)
    except (TypeError, ValueError):
        return None


def lane_split(shape, one_lane, lanes):
    """(trips over axis 0, lanes that carry data) for a tile of `shape`. The DMA banks axis 0
    one element per lane: a taller tile is dim/lanes trips, a shorter one leaves lanes idle."""
    if one_lane:
        return (shape[0] if shape else 1), 1
    dim = shape[0]
    if dim > lanes and dim % lanes:
        raise ValueError(
            f"tile dim0={dim} is neither <= {lanes} lanes nor a multiple "
            f"of them, and a scalar fallback would read the banked scratchpad "
            f"linearly")
    return (dim // lanes if dim > lanes else 1), min(dim, lanes)


def lanes_of(value):
    """How many lanes `value` spreads over, off its own type: 1 in one bank, else its axis 0's
    extent. None outside the scratchpad."""
    return _lanes_of_a_type(value.type)


def why_no_lane_axis(value):
    """Why `lanes_of` answered None, for a caller's own refusal."""
    return "%s is not a scratchpad buffer, so no lane rides it" % value.type


def bank_view(v, lanes):
    """`v` as the `[1, in-bank]` view a lane's bank really has, from the depth `v` starts at
    (a view's own offset -- the cast's is absolute), depth as need_per_lane defines it.
    A vector here spans the bank, so no tile axis bounds it."""
    depth = bank_depth(v, lanes)
    if depth is None:
        raise NotImplementedError(
            "a lane's rows of %s are not a run at one step" % v.type)
    return bank_cast(v, bank_offset(v), [1, depth], [depth, 1])


def bank_depth(v, lanes):
    """How many elements one lane's bank holds of `v`, as one run from where `v` starts. None
    for a subview that is not one: rank-reduced, cut along the lane axis, more rows than
    lanes, or skipping inside the bank."""
    from mlir.ir import MemRefType
    shape = list(MemRefType(v.type).shape)
    numel = 1
    for d in shape:
        numel *= d
    owner = op_of(getattr(v, "owner", None))
    if name_of(owner) == "memref.subview" and not _is_one_run_per_lane(owner, lanes):
        return None
    return need_per_lane(numel, lanes_of(v), lanes)


def _is_one_run_per_lane(subview, lanes):
    """Does this subview hold, in each lane's bank, one contiguous run? It keeps the lane
    axis from row 0 with no more rows than lanes, and its in-bank axes step row-major."""
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
    """Every `torchsim.bank_view` written back as the `memref.reinterpret_cast` it
    describes. THE NAME IS FOR THE PASSES, NOT THE MACHINE -- nothing below here
    lowers an unregistered op, or needs to know why a buffer was retyped."""
    from mlir.ir import (InsertionPoint, IntegerAttr, MemRefType, ShapedType)
    from mlir.dialects import memref
    from .lib_ir import int_array, walk
    made = 0
    for op in list(walk(module.operation)):
        if name_of(op) != BANK_VIEW_OP:
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
