"""Small helpers shared across the MLIR passes, and DEPENDENCY-LIGHT ON PURPOSE:
no top-level `import mlir.ir`, so a pass is importable without the bindings.
The builders import lazily and need the caller's active context."""

from ...dialects.torchsim import TransferOp, operand_number


def walk_ops(block):
    """Every op under `block` in program order, recursing into regions.
    SNAPSHOTS EACH BLOCK'S LIST, so a caller may erase while it iterates. The trace
    side holds a second copy; the two can come to enumerate a kernel differently."""
    for op in list(block.operations):
        yield op
        for region in op.operation.regions:
            for b in region.blocks:
                yield from walk_ops(b)


def walk(op):
    """Every Operation under `op`, recursing into regions. RETURNS A LIST, fully
    walked first, because callers erase while iterating and a lazy generator would
    descend into an op already gone. An Operation, never an OpView."""
    out = []

    def rec(o):
        for region in o.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    out.append(inner.operation)
                    rec(inner.operation)

    rec(op)
    return out


def reassoc_groups(op):
    """A reassociation's groups, as source axes per result axis, or None.
    ONE READER FOR THE ATTRIBUTE (rule 5): expand_shape and collapse_shape both
    state their whole correspondence here, and four passes each read it by hand."""
    from mlir.ir import ArrayAttr, IntegerAttr
    try:
        return [[IntegerAttr(x).value for x in ArrayAttr(g)]
                for g in ArrayAttr(op.attributes["reassociation"])]
    except (KeyError, TypeError, ValueError):
        return None


def CommonFactors(a, b):
    """Upstream's (util.cc): the `(i, j)` bounds where `a[:i]` and `b[:j]` have one product,
    first to last. A reshape relates the axes between two consecutive bounds and no others."""
    from math import prod
    if list(a) == list(b):
        return [(i, i) for i in range(len(a) + 1)]
    bounds, i, j, prior_i, prior_j = [], 0, 0, -1, -1
    while i < len(a) and j < len(b) and a[i] == b[j]:
        prior_i, prior_j = i, j
        bounds.append((i, j))
        i, j = i + 1, j + 1
    if prod(a[i:]) != prod(b[j:]):
        return [(0, 0), (len(a), len(b))]
    if prod(a[i:]) == 0:
        return bounds + [(i, j), (len(a), len(b))]
    pa = pb = 1
    while True:
        if pa == pb and (i > prior_i or j > prior_j):
            prior_i, prior_j = i, j
            bounds.append((i, j))
            continue
        in_a, in_b = i < len(a), j < len(b)
        if not (in_a or in_b):
            return bounds
        next_a = pa < pb or (in_a and (not in_b or (pa == pb and a[i] <= b[j])))
        next_b = pb < pa or (in_b and (not in_a or (pb == pa and b[j] <= a[i])))
        if next_a:
            pa, i = pa * a[i], i + 1
        if next_b:
            pb, j = pb * b[j], j + 1


def is_reshape_shape(v):
    """Is `v` only ever the new shape a reshape takes -- a list of sizes, not a tile?"""
    uses = list(v.uses)
    return bool(uses) and all(name_of(op_of(u.owner)) in ("tensor.reshape", "memref.reshape")
                              and u.operand_number == 1 for u in uses)


def attrs_of(op):
    """`{name: attr}` of every attribute `op` carries -- what Operation.create takes back."""
    return {op.attributes[i].name: op.attributes[i].attr for i in range(len(op.attributes))}


def int_attr(op, name):
    """`op`'s `name` attribute as a python int. Through the wrapper and not by
    hand: the wrong wrapper type reads a plausible number rather than raising."""
    from mlir.ir import IntegerAttr
    return IntegerAttr(op.attributes[name]).value


def elem_type(x):
    """The element type of a value OR a type -- itself when it is not a
    tile. Ask here rather than unwrapping a ShapedType: this is `elem_bits`' half."""
    from mlir.ir import ShapedType
    ty = getattr(x, "type", x)
    try:
        return ShapedType(ty).element_type
    except (TypeError, ValueError):
        return ty


def elem_bits(ty):
    """Width of one element in bits. `index` IS NAMED, and has to be: on
    rv64 it is 64, it is neither an integer nor a float type so neither constructor
    answers for it, and it is the type every address computation lands in."""
    from mlir.ir import FloatType, IndexType, IntegerType
    if isinstance(ty, IndexType):
        return 64
    return (IntegerType(ty).width if isinstance(ty, IntegerType)
            else FloatType(ty).width)


def elem_bytes(ty):
    """Bytes one element occupies IN MEMORY, which is not elem_bits / 8.
    A PREDICATE IS THE CASE THAT DIFFERS: no buffer holds i1, bufferize widens every
    one to i8, so an i8 tile over a `memref<*xi1>` argument is the same byte."""
    return max(elem_bits(ty), 8) // 8


def _shape_of(v):
    """The shape of a shaped value, or None if it is not one (a scalar)."""
    from mlir.ir import ShapedType
    try:
        return list(ShapedType(v.type).shape)
    except (TypeError, ValueError):
        return None


def rank_of(v):
    """Rank of a shaped value, or None if it is not one (a scalar)."""
    got = _shape_of(v)
    return None if got is None else len(got)


def _i64_signed(v):
    """An unsigned 64-bit packed value as a signed int64, which is what
    getI64IntegerAttr on the C++ side reads back."""
    v &= 0xFFFFFFFFFFFFFFFF
    return v - (1 << 64) if v >= (1 << 63) else v

def op_of(x):
    """An Operation from an OpView, an Operation, or a use's owner.
    THE ONE PLACE THAT MAY ASK: a use whose owner is a BLOCK comes back as the block,
    and `Operation == Block` raises TypeError here rather than answering False."""
    return getattr(x, "operation", x)


def int_array(a):
    """A DenseI64ArrayAttr / ArrayAttr read into a python list.
    PYTHON INTS EITHER WAY: a DenseI64ArrayAttr indexes to an int and an ArrayAttr to
    an IntegerAttr, and the second compares False against every int it equals."""
    return [int(a[i]) for i in range(len(a))]


def name_of(x):
    """`x`'s op name, or None when `x` does not have one. TWO THINGS ANSWER
    NONE: `op_of` hands back a BLOCK for a block argument, and a producer chain runs
    out at a function argument. Compare a name here and carry no guard."""
    return getattr(x, "name", None)


def producer(v):
    """The op that defines `v`, or None if a block argument does.
    `v.owner` is an Operation for a result and a BLOCK for a block argument, and the
    two are not duck-typed apart: `Operation == Block` raises TypeError."""
    from mlir.ir import BlockArgument
    if isinstance(v, BlockArgument):
        return None
    return op_of(v.owner)


def operand_segments(op):
    """`operandSegmentSizes` as a plain list of ints -- which
    operands are `ins` and which are `outs`, on every linalg op. NOT NAMED `seg`:
    callers have locals by that name, and shadowing one is a run-time NameError."""
    a = op.attributes["operandSegmentSizes"]
    return [a[i] for i in range(len(a))]


def unread_destinations(op):
    """The operand slots of a linalg op's `outs` its body never reads. Upstream's
    op has no such operand -- it defines a new value -- so neither the graph nor
    the preparation for it treats one as a tile the op takes in."""
    nm = name_of(op) or ""
    if not nm.startswith("linalg.") or not op.regions or not op.regions[0].blocks:
        return set()
    body = op.regions[0].blocks[0]
    if len(body.arguments) != len(op.operands):
        return set()
    n_in = len(op.operands) - len(op.results)
    return {k for k in range(n_in, len(op.operands))
            if not list(body.arguments[k].uses)}


def writes_through_this_operand(o, use):
    """Does `o` write the buffer it takes at this operand? UNKNOWN COUNTS AS A
    WRITE -- the set of ops that can store is open -- but a TERMINATOR FORWARDS
    a value and writes nothing, and that is known rather than unknown."""
    nm = o.name
    i = use.operand_number
    if nm in ("affine.vector_store", "memref.store", "memref.copy"):
        return i == 1
    if nm == TransferOp.OPERATION_NAME:
        into = "sram" if "MVIN" in o.operation.opview.dma_kind.value else "dram"
        return i == operand_number(o, into)
    if nm in ("linalg.transpose", "linalg.copy"):
        return i == 1
    if nm == "linalg.reduce":
        n = len(o.operands)
        return i >= n // 2 if n and not n % 2 else True
    if nm == DOT_OP:
        return i == 2
    if nm in ("linalg.generic", "linalg.broadcast", "linalg.fill", "linalg.map"):
        try:
            n_in = operand_segments(o)[0]
        except Exception:
            return True
        return i >= n_in
    if nm in ("affine.vector_load", "memref.load", "affine.load"):
        return False
    if nm in ("scf.yield", "scf.condition", "func.return"):
        return False
    return True


def reads_through_this_operand(o, use):
    """Does `o` read the buffer it takes at this operand? A store's or copy's target is not
    read; a linalg op's outs are when its payload reads them, and a dot or reduce always
    accumulates into them. UNKNOWN COUNTS AS A READ."""
    if not writes_through_this_operand(o, use):
        return True
    nm = o.name
    if nm in ("linalg.reduce", DOT_OP):
        return True
    if nm == "linalg.generic":
        return any(True for _ in o.regions[0].blocks[0].arguments[use.operand_number].uses)
    return nm not in ("affine.vector_store", "memref.store", "memref.copy", "linalg.transpose",
                      "linalg.copy", "linalg.fill", "linalg.broadcast", "linalg.map",
                      TransferOp.OPERATION_NAME)


FOR_INITS = 3
FOR_ARGS = 1
CONDITION_VALUES = 1


def get_condition_forwarded(while_op):
    """What `scf.condition` hands on, in result order: the working region's block
    arguments and the loop's results. ITS FIRST OPERAND IS THE BIT THAT DECIDES,
    which is why the values start at `CONDITION_VALUES`."""
    for region in while_op.regions:
        ops = region.blocks[0].operations
        term = ops[len(ops) - 1].operation
        if term.name == "scf.condition":
            return list(term.operands)[CONDITION_VALUES:]
    raise ValueError(f"scf.while with no scf.condition: {while_op}")


def is_memref(x):
    """Is this a memref? A value OR a type: callers ask it both ways."""
    from mlir.ir import MemRefType
    try:
        MemRefType(getattr(x, "type", x))
        return True
    except (TypeError, ValueError):
        return False


def strides_and_offset(x):
    """(strides, offset) a memref TYPE states, or None when it states no strided
    layout. A value OR a type. THE POLICY IS THE CALLER'S -- three passes wrote
    this call with three different `except` clauses, meaning three things."""
    from mlir.ir import MemRefType
    try:
        strides, offset = MemRefType(
            getattr(x, "type", x)).get_strides_and_offset()
        return list(strides), offset
    except (TypeError, ValueError, NotImplementedError):
        return None


def is_shaped(v):
    """Is this value a tile rather than one element?"""
    from mlir.ir import ShapedType
    try:
        ShapedType(v.type)
        return True
    except (TypeError, ValueError):
        return False


def zero_of(elem, loc=None):
    """A zero constant of `elem`, integer or float. `loc` may be implicit.
    THE TYPE TEST IS THE CONSTRUCTOR RAISING: `IntegerType.isinstance` does not exist
    in these bindings, so there is no cheaper way to ask."""
    from mlir.ir import FloatAttr, IntegerAttr, IntegerType
    from mlir.dialects import arith
    try:
        attr = IntegerAttr.get(IntegerType(elem), 0)
    except (TypeError, ValueError):
        attr = FloatAttr.get(elem, 0.0)
    kw = {"loc": loc} if loc is not None else {}
    return arith.ConstantOp(elem, attr, **kw).result


def splat(value, like, loc):
    """`value` (a scalar) broadcast to `like`'s shape, or itself if `like` is
    one. THE REGION IS BUILT BY HAND: linalg.FillOp's constructor makes the op and
    not its body, and an empty region fails the verifier."""
    from mlir.ir import (DenseI32ArrayAttr, InsertionPoint, Operation,
                         RankedTensorType, ShapedType)
    from mlir.dialects import linalg, tensor
    if not is_shaped(like):
        return value
    st = ShapedType(like.type)
    shape, elem = list(st.shape), st.element_type
    ty = RankedTensorType.get(shape, elem, RankedTensorType(like.type).encoding)
    empty = Operation.create("tensor.empty", results=[ty]).results[0]
    fill = Operation.create(
        "linalg.fill", results=[ty],
        operands=[value, empty],
        attributes={"operandSegmentSizes": DenseI32ArrayAttr.get([1, 1])},
        regions=1, loc=loc)
    blk = fill.regions[0].blocks.append(elem, elem)
    with InsertionPoint(blk):
        linalg.YieldOp([blk.arguments[0]])
    return fill.results[0]


def const_index(n):
    """An `index` constant. Must be built under an InsertionPoint."""
    from mlir.dialects import arith
    from mlir.ir import IndexType, IntegerAttr
    ty = IndexType.get()
    return arith.ConstantOp(ty, IntegerAttr.get(ty, n)).result


def widen(v, want):
    """`v` as integer type `want`, sign-extending or truncating only when it
    must. SIGNED: an index into a pointer is signed, and every caller here is
    one -- a zero extension puts a negative offset 4 GB away."""
    from mlir.dialects import arith
    from mlir.ir import IntegerType
    w, ww = IntegerType(v.type).width, IntegerType(want).width
    if w == ww:
        return v
    return (arith.ExtSIOp(want, v).result if w < ww
            else arith.TruncIOp(want, v).result)


def funcs(module, name="func.func"):
    """Every top-level function WITH A BODY. `tt.func` above triton-shared,
    `func.func` below it. The body test is not decoration: a declaration has an
    empty region, and `regions[0].blocks[0]` on one raises."""
    return [op for op in module.body.operations
            if op.operation.name == name
            and len(op.operation.regions[0].blocks) > 0]


def parallel_iterator():
    """`#linalg.iterator_type<parallel>`. A FUNCTION, not a constant: parsing an
    attribute needs a live Context, and this module is imported without one."""
    from mlir.ir import Attribute
    return Attribute.parse("#linalg.iterator_type<parallel>")


def iterator_types(op):
    """`op`'s iterator kinds as strings, or None when it has none."""
    from mlir.ir import ArrayAttr
    if "iterator_types" not in op.attributes:
        return None
    return [str(a) for a in ArrayAttr(op.attributes["iterator_types"])]


DOT_OP = "torchsim.dot"


def array_kind(op):
    """"matmul" or "batch_matmul" for the op the systolic array takes, torchsim.dot, by its rank;
    else None."""
    if name_of(op) != DOT_OP:
        return None
    return "matmul" if rank_of(op.operands[2]) == 2 else "batch_matmul"


def indexing_maps(op):
    """`op`'s indexing maps as AffineMaps, or None when it has none."""
    from mlir.ir import AffineMapAttr, ArrayAttr
    if "indexing_maps" not in op.attributes:
        return None
    return [AffineMapAttr(m).value for m in ArrayAttr(op.attributes["indexing_maps"])]


def set_indexing_maps(op, maps):
    """Write `maps` (AffineMaps) back as `op`'s indexing_maps."""
    from mlir.ir import AffineMapAttr, ArrayAttr
    op.attributes["indexing_maps"] = ArrayAttr.get(
        [AffineMapAttr.get(m) for m in maps])



def build_vector_body(ops, width, env, provide=None, index_value=None, loc=None):
    """A linalg or reduction body re-built on vectors of `width`, returning what it yields.
    `env` maps a value of the body to its vector; `provide` supplies one it lacks (a load),
    `index_value` a linalg.index, and anything else is a capture from outside, broadcast."""
    from mlir.ir import IntegerAttr, Operation, VectorType
    from mlir.dialects import vector

    def value_of(v):
        if v not in env:
            got = provide(v) if provide is not None else None
            env[v] = got if got is not None else vector.BroadcastOp(
                VectorType.get([width], v.type), v).result
        return env[v]

    for inner in ops:
        o = op_of(inner)
        if o.name in ("linalg.yield", "torchsim.yield"):
            return [value_of(v) for v in o.operands]
        if o.name == "linalg.index":
            env[o.results[0]] = index_value(IntegerAttr(o.attributes["dim"]).value)
            continue
        new = Operation.create(
            o.name, results=[VectorType.get([width], r.type) for r in o.results],
            operands=[value_of(v) for v in o.operands], attributes=attrs_of(o), loc=loc)
        for old_r, new_r in zip(o.results, new.results):
            env[old_r] = new_r

def elementwise_generic(ins, out_elem, build, loc, shape=None, like=None,
                        maps=None):
    """An elementwise linalg.generic over `ins`, its body
    `build(*args)`. Axes: `like` COPIES an op's, `maps` STATES them (one per operand,
    out last). A SEQUENCE of `out_elem` makes N results, and `build` returns N."""
    from mlir.dialects import linalg, tensor
    from mlir.ir import (AffineMap, AffineMapAttr, ArrayAttr,
                         DenseI32ArrayAttr, InsertionPoint, Operation,
                         RankedTensorType, ShapedType)
    ins = list(ins)
    multi = isinstance(out_elem, (list, tuple))
    outs = list(out_elem) if multi else [out_elem]
    shape = list(ShapedType(ins[0].type).shape) if shape is None else list(shape)
    ident = AffineMapAttr.get(AffineMap.get_identity(len(shape)))
    par = parallel_iterator()
    if like is not None:
        axes = {"indexing_maps": like.attributes["indexing_maps"],
                "iterator_types": like.attributes["iterator_types"]}
    else:
        axes = {"indexing_maps": ArrayAttr.get(
                    [ident] * (len(ins) + len(outs)) if maps is None
                    else list(maps)),
                "iterator_types": ArrayAttr.get([par] * len(shape))}
    # The layout: the op this one is rebuilt from (`like`) already has its answer;
    # otherwise the first input of the result's shape that carries one.
    enc = next((RankedTensorType(r.type).encoding for r in (like.results if like is not None else [])
                if isinstance(r.type, RankedTensorType) and list(RankedTensorType(r.type).shape) == shape
                and RankedTensorType(r.type).encoding is not None), None)
    if enc is None:
        enc = next((RankedTensorType(v.type).encoding for v in ins
                    if isinstance(v.type, RankedTensorType) and list(RankedTensorType(v.type).shape) == shape
                    and RankedTensorType(v.type).encoding is not None), None)
    types = [RankedTensorType.get(shape, e, enc) for e in outs]
    empties = [Operation.create("tensor.empty", results=[ty]).results[0] for ty in types]
    gen = Operation.create(
        "linalg.generic",
        results=types,
        operands=ins + empties,
        attributes=dict(axes, operandSegmentSizes=DenseI32ArrayAttr.get(
            [len(ins), len(outs)])),
        regions=1, loc=loc)
    blk = gen.regions[0].blocks.append(
        *[ShapedType(v.type).element_type for v in ins], *outs)
    with InsertionPoint(blk):
        got = build(*list(blk.arguments)[:len(ins)])
        linalg.YieldOp(list(got) if multi else [got])
    return list(gen.results) if multi else gen.results[0]


def gather(ptr, off, result_ty, loc, mask=None, other=None):
    """tts.gather over (ptr, offsets), with a mask and fill when given; segments
    are [ptr, offset, mask, other]. `other` IS A SCALAR -- a caller holding a tile
    of fill reduces it to a scalar first and refuses when it cannot."""
    from mlir.ir import DenseI32ArrayAttr, Operation
    ops = [ptr, off] + ([] if mask is None else [mask]) \
                     + ([] if other is None else [other])
    return Operation.create(
        "tts.gather", results=[result_ty], operands=ops,
        attributes={"operandSegmentSizes":
                    DenseI32ArrayAttr.get([1, 1, 0 if mask is None else 1,
                                           0 if other is None else 1])},
        loc=loc).results[0]


GATHER_PTR, GATHER_OFF, GATHER_MASK, GATHER_OTHER = 0, 1, 2, 3
GS_STRIDE_SEG, GS_OFFSET_SEG, GS_SHAPE_SEG = 2, 3, 4
GS_MASK_SEG, GS_BASE_SEG = 5, 6


def transfer_read(vec_ty, source, indices, padding):
    """A `vector.transfer_read` of `vec_ty` at `indices`, padded with `padding`."""
    import mlir.ir as ir
    from mlir.dialects import vector
    src_rank = len(ir.MemRefType(source.type).shape)
    vec_rank = len(ir.VectorType(vec_ty).shape)
    perm = ir.AffineMap.get_minor_identity(src_rank, vec_rank)
    return vector.TransferReadOp(vec_ty, source, list(indices), perm, padding,
                                 [False] * vec_rank).result


def transfer_write(value, dest, indices):
    """A `vector.transfer_write` of `value` at `indices`."""
    import mlir.ir as ir
    from mlir.dialects import vector
    dst_rank = len(ir.MemRefType(dest.type).shape)
    vec_rank = len(ir.VectorType(value.type).shape)
    perm = ir.AffineMap.get_minor_identity(dst_rank, vec_rank)
    vector.TransferWriteOp(None, value, dest, list(indices), perm, [False] * vec_rank)


def build_trip_loop(body_ip, trips):
    """One `affine.for 0 to trips` with its terminator in place, and the insertion
    point moved INSIDE it: (induction variable, that point) -- gemm's three nests
    and the gathering trip's passes."""
    import mlir.ir as ir
    from mlir.dialects import affine
    with body_ip:
        lp = affine.AffineForOp(0, trips, 1)
    with ir.InsertionPoint(lp.body):
        affine.AffineYieldOp([])
    return lp.induction_variable, ir.InsertionPoint.at_block_terminator(lp.body)
