"""Small helpers shared across the MLIR passes, and DEPENDENCY-LIGHT ON PURPOSE: no top-level
`import mlir.ir`, so a pass is importable without the bindings."""

from ...dialects.torchsim import TransferOp, effects_on, is_load, operand_number


VIEW_OPS = ("memref.subview", "memref.cast", "memref.reinterpret_cast", "memref.expand_shape",
            "memref.collapse_shape", "memref.reshape")


def walk_ops(block):
    """Every op under `block` in program order, recursing into regions."""
    for op in list(block.operations):
        yield op
        for region in op.operation.regions:
            for b in region.blocks:
                yield from walk_ops(b)


def walk(op):
    """Every Operation under `op`, recursing into regions."""
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
    """A reassociation's groups, as source axes per result axis, or None."""
    from mlir.ir import ArrayAttr, IntegerAttr
    try:
        return [[IntegerAttr(x).value for x in ArrayAttr(g)]
                for g in ArrayAttr(op.attributes["reassociation"])]
    except (KeyError, TypeError, ValueError):
        return None


def CommonFactors(a, b):
    """Upstream's (util.cc): the `(i, j)` bounds where `a[:i]` and `b[:j]` have one product, first
    to last."""
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
    """`op`'s `name` attribute as a python int."""
    from mlir.ir import IntegerAttr
    return IntegerAttr(op.attributes[name]).value


def elem_type(x):
    """The element type of a value OR a type -- itself when it is not a tile."""
    from mlir.ir import ShapedType
    ty = getattr(x, "type", x)
    try:
        return ShapedType(ty).element_type
    except (TypeError, ValueError):
        return ty


def elem_bits(ty):
    """Width of one element in bits."""
    from mlir.ir import FloatType, IndexType, IntegerType
    if isinstance(ty, IndexType):
        return 64
    return (IntegerType(ty).width if isinstance(ty, IntegerType)
            else FloatType(ty).width)


def elem_bytes(ty):
    """Bytes one element occupies IN MEMORY, which is not elem_bits / 8."""
    return max(elem_bits(ty), 8) // 8


def _shape_of(v):
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
    v &= 0xFFFFFFFFFFFFFFFF
    return v - (1 << 64) if v >= (1 << 63) else v

def op_of(x):
    """An Operation from an OpView, an Operation, or a use's owner."""
    return getattr(x, "operation", x)


def int_array(a):
    """A DenseI64ArrayAttr / ArrayAttr read into a python list."""
    return [int(a[i]) for i in range(len(a))]


def name_of(x):
    """`x`'s op name, or None when `x` does not have one."""
    return getattr(x, "name", None)


def values(op):
    """Every Value under `op`: each block's arguments and each op's results."""
    out = []

    def rec(o):
        for region in o.regions:
            for block in region.blocks:
                out.extend(block.arguments)
                for inner in list(block.operations):
                    out.extend(inner.operation.results)
                    rec(inner.operation)

    rec(op)
    return out


def op_counts(op):
    """Op name -> how many ops of that name are under `op`."""
    from collections import Counter
    return Counter(o.name for o in walk(op))


def attr_counts(op):
    """Attribute name -> how many ops under `op` carry an attribute of that name."""
    from collections import Counter
    return Counter(o.attributes[i].name for o in walk(op) for i in range(len(o.attributes)))


def opaque_parts(x):
    """(dialect, mnemonic, body) of a type or attribute whose dialect has no bindings here, read
    from its opaque form `!d.m<body>` / `#d.m<body>`; None for one that is not opaque."""
    from mlir.ir import OpaqueAttr, OpaqueType
    if not isinstance(x, (OpaqueAttr, OpaqueType)):
        return None
    data = x.data
    data = data.decode() if isinstance(data, bytes) else data
    mnemonic, _, body = data.partition("<")
    return x.dialect_namespace, mnemonic, body[:-1] if body.endswith(">") else body


_SAMPLE_TYPE = {("ptr", "ptr"): "!ptr.ptr<#ptr.generic_space>"}
_TYPEID = {}


def is_type(ty, dialect, mnemonic):
    """Whether `ty` is `!dialect.mnemonic<..>`: by TypeID for a dialect registered here, by the
    opaque form's dialect and mnemonic for one that is not."""
    parts = opaque_parts(ty)
    if parts is not None:
        return parts[:2] == (dialect, mnemonic)
    key = (dialect, mnemonic)
    if key not in _SAMPLE_TYPE:
        return False
    if key not in _TYPEID:
        from mlir.ir import Type
        _TYPEID[key] = Type.parse(_SAMPLE_TYPE[key], context=ty.context).typeid
    return ty.typeid == _TYPEID[key]


def holds_type(x, dialect, mnemonic):
    """Whether a value or type is `!dialect.mnemonic` itself or a tile of it."""
    return is_type(elem_type(x), dialect, mnemonic)


def attr_is(attr, text):
    """Whether `attr` is the attribute `text` spells -- an enum with no Python class here."""
    from mlir.ir import Attribute
    return attr == Attribute.parse(text, context=attr.context)


def enum_of(attr, form, names):
    """The name in `names` whose `form % name` is `attr`, else None."""
    return next((n for n in names if attr_is(attr, form % n)), None)


def opaque_body_attr(x, dialect, mnemonic):
    """The `<...>` body of the opaque attribute `#dialect.mnemonic<...>`, parsed as an attribute of
    its own; None when `x` is not that attribute."""
    parts = opaque_parts(x)
    if parts is None or parts[:2] != (dialect, mnemonic):
        return None
    from mlir.ir import Attribute
    return Attribute.parse(parts[2], context=x.context)


def producer(v):
    """The op that defines `v`, or None if a block argument does."""
    from mlir.ir import BlockArgument
    if isinstance(v, BlockArgument):
        return None
    return op_of(v.owner)


def operand_segments(op):
    """`operandSegmentSizes` as a plain list of ints -- which operands are `ins` and which are
    `outs`, on every linalg op."""
    a = op.attributes["operandSegmentSizes"]
    return [a[i] for i in range(len(a))]


def unread_destinations(op):
    """The operand slots of a linalg op's `outs` its body never reads."""
    nm = name_of(op) or ""
    if not nm.startswith("linalg.") or not op.regions or not op.regions[0].blocks:
        return set()
    body = op.regions[0].blocks[0]
    if len(body.arguments) != len(op.operands):
        return set()
    n_in = len(op.operands) - len(op.results)
    return {k for k in range(n_in, len(op.operands))
            if not list(body.arguments[k].uses)}


def ends_lifetime(op, value):
    """Does `op` only free `value` -- its buffer's life ends there, and no element is read or
    written?"""
    return effects_on(op, value) == frozenset({"free"})


def bounds_lives_only(op):
    """Does `op` only begin or end buffers' lives: no region, and all it states of each memref it
    takes or makes is an allocate, or all of it a free?"""
    vals = [v for v in list(op.operands) + list(op.results) if is_memref(v)]
    return not len(op.regions) and bool(vals) and all(
        effects_on(op, v) in (frozenset({"allocate"}), frozenset({"free"})) for v in vals)


def touches_data(o, use):
    """Is the buffer `o` takes at this operand something `o` may read or write? Not when all `o`
    states of it is a free (ends_lifetime) or its transfers in flight (a wait's landing)."""
    return effects_on(o, o.operands[use.operand_number]) not in (frozenset({"free"}),
                                                                 frozenset({"sync"}))


def writes_through_this_operand(o, use):
    """Does `o` write the buffer it takes at this operand?"""
    if not touches_data(o, use):
        return False
    nm = o.name
    i = use.operand_number
    if nm in ("affine.vector_store", "memref.store", "memref.copy"):
        return i == 1
    if nm == TransferOp.OPERATION_NAME:
        into = "sram" if is_load(o) else "dram"
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
    """Does `o` read the buffer it takes at this operand?"""
    if not touches_data(o, use):
        return False
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
    """What `scf.condition` hands on, in result order: the working region's block arguments and the
    loop's results."""
    for region in while_op.regions:
        ops = region.blocks[0].operations
        term = ops[len(ops) - 1].operation
        if term.name == "scf.condition":
            return list(term.operands)[CONDITION_VALUES:]
    raise ValueError(f"scf.while with no scf.condition: {while_op}")


def is_memref(x):
    """Is this a memref?"""
    from mlir.ir import MemRefType
    try:
        MemRefType(getattr(x, "type", x))
        return True
    except (TypeError, ValueError):
        return False


def strides_and_offset(x):
    """(strides, offset) a memref TYPE states, or None when it states no strided layout."""
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
    """A zero constant of `elem`, integer or float."""
    from mlir.ir import FloatAttr, IntegerAttr, IntegerType
    from mlir.dialects import arith
    try:
        attr = IntegerAttr.get(IntegerType(elem), 0)
    except (TypeError, ValueError):
        attr = FloatAttr.get(elem, 0.0)
    kw = {"loc": loc} if loc is not None else {}
    return arith.ConstantOp(elem, attr, **kw).result


def splat(value, like, loc):
    """`value` (a scalar) broadcast to `like`'s shape, or itself if `like` is one."""
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
    """An `index` constant."""
    from mlir.dialects import arith
    from mlir.ir import IndexType, IntegerAttr
    ty = IndexType.get()
    return arith.ConstantOp(ty, IntegerAttr.get(ty, n)).result


def widen(v, want):
    """`v` as integer type `want`, sign-extending or truncating only when it must."""
    from mlir.dialects import arith
    from mlir.ir import IntegerType
    w, ww = IntegerType(v.type).width, IntegerType(want).width
    if w == ww:
        return v
    return (arith.ExtSIOp(want, v).result if w < ww
            else arith.TruncIOp(want, v).result)


LANES_ATTR = "ttg.threads-per-warp"


def lanes_of_module(module):
    """The lane count `module` was made for: its LANES_ATTR, as Triton wrote it. A module that does
    not say is refused by name -- Triton's own default (32) is no machine's answer here."""
    from mlir.ir import IntegerAttr
    attrs = module.operation.attributes
    if LANES_ATTR not in attrs:
        raise ValueError(f"the module does not say the lane count it was made for ({LANES_ATTR})")
    if not isinstance(attrs[LANES_ATTR], IntegerAttr):
        raise ValueError(f"the module's {LANES_ATTR} is not an integer: {attrs[LANES_ATTR]}")
    return IntegerAttr(attrs[LANES_ATTR]).value


def funcs(module, name="func.func"):
    """Every top-level function WITH A BODY."""
    return [op for op in module.body.operations
            if op.operation.name == name
            and len(op.operation.regions[0].blocks) > 0]


def parallel_iterator():
    """`#linalg.iterator_type<parallel>`."""
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
    """"matmul" or "batch_matmul" for the op the matrix unit takes, torchsim.dot, by its rank;
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
    """A linalg or reduction body re-built on vectors of `width`, returning what it yields."""
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
    """An elementwise linalg.generic over `ins`, its body `build(*args)`."""
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


def gather(ptr, off, result_ty, loc):
    """tts.gather over (ptr, offsets), with no mask and no fill."""
    from mlir.ir import DenseI32ArrayAttr, Operation
    return Operation.create(
        "tts.gather", results=[result_ty], operands=[ptr, off],
        attributes={"operandSegmentSizes": DenseI32ArrayAttr.get([1, 1, 0, 0])},
        loc=loc).results[0]


GATHER_PTR, GATHER_OFF, GATHER_MASK, GATHER_OTHER = 0, 1, 2, 3
GS_STRIDE_SEG, GS_OFFSET_SEG, GS_SHAPE_SEG = 2, 3, 4
GS_MASK_SEG, GS_BASE_SEG = 5, 6


def transfer_read(vec_ty, source, indices, padding, in_bounds=False):
    """A `vector.transfer_read` of `vec_ty` at `indices`, padded with `padding`; `in_bounds` when the
    caller has shown the read never leaves `source`, so the lowering needs no bounds check."""
    import mlir.ir as ir
    from mlir.dialects import vector
    src_rank = len(ir.MemRefType(source.type).shape)
    vec_rank = len(ir.VectorType(vec_ty).shape)
    perm = ir.AffineMap.get_minor_identity(src_rank, vec_rank)
    return vector.TransferReadOp(vec_ty, source, list(indices), perm, padding,
                                 [in_bounds] * vec_rank).result


def transfer_write(value, dest, indices, in_bounds=False, mask=None):
    """A `vector.transfer_write` of `value` at `indices`; `in_bounds` as transfer_read's, and only
    the elements `mask` holds true are written when it is given."""
    import mlir.ir as ir
    from mlir.dialects import vector
    dst_rank = len(ir.MemRefType(dest.type).shape)
    vec_rank = len(ir.VectorType(value.type).shape)
    perm = ir.AffineMap.get_minor_identity(dst_rank, vec_rank)
    return vector.TransferWriteOp(None, value, dest, list(indices), perm, [in_bounds] * vec_rank,
                                  mask=mask)


def build_trip_loop(body_ip, trips):
    """One `affine.for 0 to trips` with its terminator in place, and the insertion point moved
    INSIDE it: (induction variable, that point) -- gemm's three nests and the gathering trip's
    passes."""
    import mlir.ir as ir
    from mlir.dialects import affine
    with body_ip:
        lp = affine.AffineForOp(0, trips, 1)
    with ir.InsertionPoint(lp.body):
        affine.AffineYieldOp([])
    return lp.induction_variable, ir.InsertionPoint.at_block_terminator(lp.body)


def clone_ops(ops, mapping, ip):
    """Each of `ops` cloned at `ip`, in order, reading `mapping`'s values for the ones it names
    (its regions too); each clone's results join `mapping` for the ops after it."""
    for o in ops:
        new = o.clone(ip).operation
        for x in [new] + walk(new):
            for k, v in enumerate(list(x.operands)):
                if v in mapping:
                    x.operands[k] = mapping[v]
        mapping.update(zip(o.results, new.results))


def symbols(module):
    """The sym_name of every top-level op of `module`."""
    from mlir.ir import StringAttr
    return {StringAttr(o.operation.attributes["sym_name"]).value
            for o in module.operation.regions[0].blocks[0].operations
            if "sym_name" in o.operation.attributes}


def const_int(v):
    """`v`'s value when an `arith.constant` integer defines it, else None."""
    from mlir.ir import IntegerAttr
    p = producer(v)
    if p is None or p.name != "arith.constant":
        return None
    try:
        return IntegerAttr(p.attributes["value"]).value
    except (TypeError, ValueError):
        return None


def eval_affine(e, dims, syms):
    """The value of affine expression `e` at these dim and symbol values."""
    import mlir.ir as ir
    if isinstance(e, ir.AffineConstantExpr):
        return e.value
    if isinstance(e, ir.AffineDimExpr):
        return dims[e.position]
    if isinstance(e, ir.AffineSymbolExpr):
        return syms[e.position]
    for kind, f in ((ir.AffineAddExpr, lambda x, y: x + y), (ir.AffineMulExpr, lambda x, y: x * y),
                    (ir.AffineModExpr, lambda x, y: x % y),
                    (ir.AffineFloorDivExpr, lambda x, y: x // y),
                    (ir.AffineCeilDivExpr, lambda x, y: -(-x // y))):
        if isinstance(e, kind):
            return f(eval_affine(e.lhs, dims, syms), eval_affine(e.rhs, dims, syms))
    raise ValueError(f"an affine expression of no known kind: {e}")


def _affine_bound(m, operands):
    """The value a one-result bound map names, when that result is a constant or a constant
    dim/symbol operand; None for any other map, so no map is read as its first operand."""
    import mlir.ir as ir
    if len(m.results) != 1:
        return None
    e = m.results[0]
    if isinstance(e, ir.AffineConstantExpr):
        return e.value
    if isinstance(e, ir.AffineDimExpr):
        return const_int(operands[e.position])
    if isinstance(e, ir.AffineSymbolExpr):
        return const_int(operands[m.n_dims + e.position])
    return None


def loop_bounds(loop):
    """(start, end, step) of an `scf.for` / `affine.for`, each an int or None where it is not a
    constant; None for any other op. The one reading of a loop's bounds."""
    import mlir.ir as ir
    o = op_of(loop)
    if o.name == "scf.for":
        ops = list(o.operands)
        return tuple(const_int(v) for v in ops[:3]) if len(ops) >= 3 else None
    if o.name != "affine.for":
        return None
    ops, n_lb = list(o.operands), int(o.attributes["operandSegmentSizes"][0])
    n_ub = int(o.attributes["operandSegmentSizes"][1])
    lb, ub = (ir.AffineMapAttr(o.attributes[n]).value for n in ("lowerBoundMap", "upperBoundMap"))
    return (_affine_bound(lb, ops[:n_lb]), _affine_bound(ub, ops[n_lb:n_lb + n_ub]),
            ir.IntegerAttr(o.attributes["step"]).value)


def trip_count(loop):
    """How many trips `loop` makes, when its bounds are constants; else None."""
    b = loop_bounds(loop)
    if b is None or any(x is None for x in b) or b[2] <= 0:
        return None
    return max(0, -(-(b[1] - b[0]) // b[2]))
