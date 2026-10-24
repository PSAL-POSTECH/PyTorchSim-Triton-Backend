"""pytorchsim-fuse: collapse the elementwise linalg chain into one op after the layouts are decided, a
broadcast included; a cast is re-made at each reader, and an op only a reduce reads goes into the reduce's
body, so no made tile is written to the scratchpad and read back."""

from .lib_bank_layout import is_one_bank
from .lib_ir import (attrs_of, funcs, int_array, name_of, op_of, parallel_iterator, rank_of,
                     walk)

FENCE = "pytorchsim.fusion_fence"

FORBIDS = (FENCE,)

_ARRAY_OPS = ("pytorchsim.dot",)

_RESHAPES = ("tensor.collapse_shape", "tensor.expand_shape", "tensor.reshape")


CASTS = ("arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp", "arith.fptosi",
          "arith.fptoui", "arith.extsi", "arith.extui", "arith.trunci")

def run(module, **_opts):
    """The table of contents: fence, let upstream fuse, unfence, lower what fusion left."""
    from mlir.ir import Location
    from mlir.passmanager import PassManager

    with Location.unknown():
        for f in funcs(module):
            _rewrite_broadcasts_to_generics(f.operation)
            _rewrite_array_fences(f.operation)
            _rewrite_reshape_fences(f.operation)
            _rewrite_casts_into_readers(f.operation)
            PassManager.parse("func.func(linalg-fuse-elementwise-ops)").run(f.operation)
            _erase_fences(f.operation)
            _rewrite_loop_partials_into_rows(f.operation)
            _rewrite_producers_into_reduces(f.operation)
            _rewrite_readers_into_producers(f.operation)
            _rewrite_index_makers_into_readers(f.operation)
    _clean_up(module)
    return module


def _rewrite_broadcasts_to_generics(func):
    from mlir.dialects import linalg
    from mlir.ir import (AffineDimExpr, AffineMap, AffineMapAttr, ArrayAttr, DenseI32ArrayAttr,
                         InsertionPoint, Operation, RankedTensorType)
    for op in [o for o in walk(func) if name_of(o) == "linalg.broadcast"]:
        src, init = op.operands[0], op.operands[1]
        rank = rank_of(init)
        added = set(int_array(op.attributes["dimensions"]))
        kept = [AffineDimExpr.get(d) for d in range(rank) if d not in added]
        with InsertionPoint(op_of(op)):
            gen = Operation.create(
                "linalg.generic", results=[init.type], operands=[src, init],
                attributes={"indexing_maps": ArrayAttr.get([
                                AffineMapAttr.get(AffineMap.get(rank, 0, kept)),
                                AffineMapAttr.get(AffineMap.get_identity(rank))]),
                            "iterator_types": ArrayAttr.get([parallel_iterator()] * rank),
                            "operandSegmentSizes": DenseI32ArrayAttr.get([1, 1])},
                regions=1)
        blk = gen.regions[0].blocks.append(RankedTensorType(src.type).element_type,
                                           RankedTensorType(init.type).element_type)
        with InsertionPoint(blk):
            linalg.YieldOp([blk.arguments[0]])
        op.results[0].replace_all_uses_with(gen.results[0])
        op_of(op).erase()


_COMBINERS = {"arith.addf": 0.0, "arith.maxnumf": float("-inf"), "arith.minnumf": float("inf"),
              "arith.maximumf": float("-inf"), "arith.minimumf": float("inf")}


def _rewrite_loop_partials_into_rows(func):
    """Inductor's looped reduction keeps a [rows, R0] tile of partials across the loop
    (`acc = combine(acc, v)` per element each trip) and reduces it over R0 once after the loop.
    When the combine and the reduce are the same op and both start from its identity, that is the
    loop carrying one value a row, each trip reducing its chunk into it: the carried tile becomes a
    [rows] tensor and the per-element combine a generic that reduces its last dim into the carry."""
    for loop in [o for o in walk(func) if name_of(o) == "scf.for"]:
        _loop_partials_into_rows(loop)


def _loop_partials_into_rows(loop):
    from mlir.ir import (AffineDimExpr, AffineMap, AffineMapAttr, ArrayAttr, Attribute,
                         DenseI32ArrayAttr, InsertionPoint, Operation, RankedTensorType)
    body = loop.regions[0].blocks[0]
    n_iv = 1
    inits = list(loop.operands[3:])
    yields = [x.operation for x in body.operations][-1]
    for i, (init, res) in enumerate(zip(inits, loop.results)):
        carry = body.arguments[n_iv + i]
        uses = list(carry.uses)
        if len(uses) != 1 or len(list(res.uses)) != 1:
            continue
        gen = op_of(uses[0].owner)
        red = op_of(next(iter(res.uses)).owner)
        fill = op_of(getattr(init, "owner", None))
        if name_of(gen) != "linalg.generic" or name_of(red) != "linalg.reduce" \
                or name_of(fill) != "linalg.fill" or yields.operands[i] != gen.results[0]:
            continue
        if len(gen.results) != 1 or len(red.operands) != 2 or len(list(gen.results[0].uses)) != 1:
            continue
        rank = rank_of(carry)
        if list(int_array(red.attributes["dimensions"])) != [rank - 1]:
            continue
        maps = [AffineMapAttr(m).value for m in ArrayAttr(gen.attributes["indexing_maps"])]
        if any(m != AffineMap.get_identity(rank) for m in maps) or any(
                str(t) != str(parallel_iterator()) for t in ArrayAttr(gen.attributes["iterator_types"])):
            continue
        n_in = len(gen.operands) - 1
        k = uses[0].operand_number
        if k >= n_in:
            continue
        gbody, rbody = gen.regions[0].blocks[0], red.regions[0].blocks[0]
        gsteps = [x.operation for x in gbody.operations]
        rsteps = [x.operation for x in rbody.operations]
        comb = producer_op(gsteps[-1].operands[0])
        if comb is None or len(rsteps) != 2 or rsteps[0].name != getattr(comb, "name", None) \
                or comb.name not in _COMBINERS or gbody.arguments[k] not in list(comb.operands):
            continue
        if list(gbody.arguments[n_in].uses) or len(list(gbody.arguments[k].uses)) != 1:
            continue
        ident = _COMBINERS[comb.name]
        if _const_float(fill.operands[0]) != ident or _const_float(
                op_of(getattr(red.operands[1], "owner", None)).operands[0]
                if name_of(op_of(getattr(red.operands[1], "owner", None))) == "linalg.fill"
                else None) != ident:
            continue
        row_ty = RankedTensorType(red.results[0].type)
        # the carry starts from the reduce's own init, made ahead of the loop
        rinit = red.operands[1]
        rfill = op_of(rinit.owner)
        rempty = op_of(getattr(rfill.operands[1], "owner", None))
        with InsertionPoint(loop):
            e = rempty.clone() if rempty is not None else None
            f = rfill.clone()
        if e is not None:
            f.operands[1] = e.results[0]
        loop.operands[3 + i] = f.results[0]
        carry.set_type(row_ty)
        res.set_type(row_ty)
        ins = [v for j, v in enumerate(gen.operands[:n_in]) if j != k]
        kept = AffineMap.get(rank, 0, [AffineDimExpr.get(d) for d in range(rank - 1)])
        with InsertionPoint(gen):
            new = Operation.create(
                "linalg.generic", results=[row_ty], operands=ins + [carry],
                attributes={
                    "indexing_maps": ArrayAttr.get(
                        [AffineMapAttr.get(AffineMap.get_identity(rank))] * len(ins)
                        + [AffineMapAttr.get(kept)]),
                    "iterator_types": ArrayAttr.get(
                        [parallel_iterator()] * (rank - 1)
                        + [Attribute.parse("#linalg.iterator_type<reduction>")]),
                    "operandSegmentSizes": DenseI32ArrayAttr.get([len(ins), 1])},
                regions=1, loc=gen.location)
        blk = new.regions[0].blocks.append(
            *[gbody.arguments[j].type for j in range(n_in) if j != k], row_ty.element_type)
        env = {}
        it = iter(blk.arguments[:len(ins)])
        for j in range(n_in):
            if j != k:
                env[gbody.arguments[j]] = next(it)
        env[gbody.arguments[k]] = blk.arguments[len(ins)]
        with InsertionPoint(blk):
            for inner in gbody.operations:
                o = inner.operation
                got = Operation.create(o.name, results=[r.type for r in o.results],
                                       operands=[env.get(v, v) for v in o.operands],
                                       attributes=attrs_of(o), loc=o.location)
                env.update(zip(o.results, got.results))
        yields.operands[i] = new.results[0]
        gen.erase()
        red.results[0].replace_all_uses_with(res)
        red.erase()
        for dead in (rfill, rempty, fill):
            if dead is not None and not any(list(r.uses) for r in dead.results):
                dead.erase()


def producer_op(v):
    o = op_of(getattr(v, "owner", None))
    return o


def _const_float(v):
    from mlir.ir import FloatAttr
    if v is None:
        return None
    o = op_of(getattr(v, "owner", None))
    if name_of(o) != "arith.constant":
        return None
    try:
        return FloatAttr(o.attributes["value"]).value
    except ValueError:
        return None


def _rewrite_producers_into_reduces(func):
    """A `linalg.reduce` over the last dim of one input that an all-parallel generic makes and no
    one else reads becomes one generic: the producer's inputs and maps, the reduce's init, and a
    body that runs the producer's then combines into the init."""
    for red in [o for o in walk(func) if name_of(o) == "linalg.reduce"]:
        fused = _producer_into_reduce(red)
        if fused is not None:
            prod = op_of(red.operands[0].owner)
            for a, b in zip(red.results, fused.results):
                a.replace_all_uses_with(b)
            red.erase()
            prod.erase()


def _producer_into_reduce(red):
    from mlir.ir import (AffineDimExpr, AffineMap, AffineMapAttr, ArrayAttr, Attribute,
                         DenseI32ArrayAttr, InsertionPoint, Operation, RankedTensorType)
    if len(red.operands) != 2 or len(red.results) != 1:
        return None
    src, init = red.operands
    prod = op_of(getattr(src, "owner", None))
    if name_of(prod) != "linalg.generic" or len(prod.results) != 1 \
            or len(list(src.uses)) != 1:
        return None
    rank = rank_of(src)
    dims = list(int_array(red.attributes["dimensions"]))
    if dims != [rank - 1] or folds_lanes(red):
        return None
    if any(str(t) != str(parallel_iterator()) for t in ArrayAttr(prod.attributes["iterator_types"])):
        return None
    maps = [AffineMapAttr(m).value for m in ArrayAttr(prod.attributes["indexing_maps"])]
    if maps[-1] != AffineMap.get_identity(rank):
        return None
    n_in = len(prod.operands) - 1
    pbody, rbody = prod.regions[0].blocks[0], red.regions[0].blocks[0]
    if list(pbody.arguments[n_in].uses):
        return None                     # the producer reads its own init
    kept = AffineMap.get(rank, 0, [AffineDimExpr.get(d) for d in range(rank - 1)])
    out_ty = RankedTensorType(red.results[0].type)
    with InsertionPoint(red):
        gen = Operation.create(
            "linalg.generic", results=[out_ty], operands=list(prod.operands[:n_in]) + [init],
            attributes={
                "indexing_maps": ArrayAttr.get([AffineMapAttr.get(m) for m in maps[:n_in]]
                                               + [AffineMapAttr.get(kept)]),
                "iterator_types": ArrayAttr.get(
                    [parallel_iterator()] * (rank - 1)
                    + [Attribute.parse("#linalg.iterator_type<reduction>")]),
                "operandSegmentSizes": DenseI32ArrayAttr.get([n_in, 1])},
            regions=1, loc=red.location)
    blk = gen.regions[0].blocks.append(*[a.type for a in pbody.arguments[:n_in]],
                                       out_ty.element_type)
    env = dict(zip(list(pbody.arguments[:n_in]), list(blk.arguments[:n_in])))
    with InsertionPoint(blk):
        for inner in pbody.operations:
            o = inner.operation
            if o.name == "linalg.yield":
                env[rbody.arguments[0]] = env.get(o.operands[0], o.operands[0])
                env[rbody.arguments[1]] = blk.arguments[n_in]
                break
            new = Operation.create(o.name, results=[r.type for r in o.results],
                                   operands=[env.get(v, v) for v in o.operands],
                                   attributes=attrs_of(o), loc=o.location)
            env.update(zip(o.results, new.results))
        for inner in rbody.operations:
            o = inner.operation
            new = Operation.create(o.name, results=[r.type for r in o.results],
                                   operands=[env.get(v, v) for v in o.operands],
                                   attributes=attrs_of(o), loc=o.location)
            env.update(zip(o.results, new.results))
    return gen


def folds_lanes(red):
    """A reduce over the lane axis (axis 0 of a tile not in one bank) folds the lanes: it is lowered
    as a move across them, so no walk takes it into its body."""
    from mlir.ir import ShapedType
    shape = list(ShapedType(red.operands[0].type).shape)
    return 0 in list(int_array(red.attributes["dimensions"])) and not is_one_bank(shape)


def _rewrite_readers_into_producers(func):
    """A tile several ops read is made once, by one walk that also does what its readers can do
    there: an elementwise reader over the same space joins it as another output, and a reduce over
    its last dim joins it as a reduced output. The tile stays an output only while something
    else still reads it, so softmax's p is never written in f32 for a sum and a cast to read."""
    while True:
        for g in [o for o in walk(func) if _is_parallel_maker(o)]:
            if _elementwise_reader_into(g) or _reduce_reader_into(g):
                break
        else:
            break
    for g in [o for o in walk(func) if name_of(o) == "linalg.generic" and len(o.results) > 1]:
        _drop_unread_outputs(g)


def _is_parallel_maker(op):
    """An all-parallel generic whose outputs are written element for element and never read."""
    from mlir.ir import AffineMap, AffineMapAttr, ArrayAttr
    if name_of(op) != "linalg.generic" or not op.results:
        return False
    if any(str(t) != str(parallel_iterator()) for t in ArrayAttr(op.attributes["iterator_types"])):
        return False
    n_out = len(op.results)
    rank = rank_of(op.results[0])
    maps = [AffineMapAttr(m).value for m in ArrayAttr(op.attributes["indexing_maps"])]
    blk = op.regions[0].blocks[0]
    n_in = len(op.operands) - n_out
    return all(m == AffineMap.get_identity(rank) for m in maps[n_in:]) and not any(
        list(a.uses) for a in list(blk.arguments)[n_in:])


def _dominates(v, op):
    """`v` is available where `op` is: a value of an earlier op of its block, or an argument of a
    block that holds it."""
    owner = getattr(v, "owner", None)
    blk = op.operation.block
    if hasattr(owner, "name") or hasattr(owner, "operation"):
        o = op_of(owner)
        return o.operation.block == blk and o.operation.is_before_in_block(op.operation)
    b = blk
    while b is not None:
        if b == owner:
            return True
        parent = b.owner if hasattr(b, "owner") else None
        b = parent.block if parent is not None and hasattr(parent, "block") else None
    return False


def _after_or_at(user, anchor):
    """`user`, or the op of `anchor`'s block that holds it, is `anchor` or comes after it."""
    o = user.operation
    while o is not None and o.block != anchor.operation.block:
        o = o.parent
    return o is not None and (o == anchor.operation or anchor.operation.is_before_in_block(o))


def _is_empty(v):
    return name_of(op_of(getattr(v, "owner", None))) == "tensor.empty"


def _captured(op):
    """The values `op`'s body reads that are made outside it."""
    ops = walk(op)
    inside = set()
    for o in [op.operation] + ops:
        for r in o.regions:
            for b in r.blocks:
                inside.update(b.arguments)
        if o != op.operation:
            inside.update(o.results)
    return [v for o in ops for v in o.operands if v not in inside]


def _elem_type(v):
    from mlir.ir import RankedTensorType
    try:
        return RankedTensorType(v.type).element_type
    except ValueError:
        return v.type


def _inline_body(blk, env):
    """`blk`'s ops cloned at the insertion point with its arguments as `env` says; what it
    yields, in the clones' values."""
    from mlir.ir import Operation
    for inner in blk.operations:
        o = inner.operation
        if o.name == "linalg.yield":
            return [env.get(v, v) for v in o.operands]
        got = Operation.create(o.name, results=[r.type for r in o.results],
                               operands=[env.get(v, v) for v in o.operands],
                               attributes=attrs_of(o), loc=o.location)
        env.update(zip(o.results, got.results))
    return []


def _generic(ins, outs, maps, iters, loc):
    from mlir.ir import (AffineMapAttr, ArrayAttr, DenseI32ArrayAttr, Operation)
    gen = Operation.create(
        "linalg.generic", results=[v.type for v in outs], operands=list(ins) + list(outs),
        attributes={"indexing_maps": ArrayAttr.get([AffineMapAttr.get(m) for m in maps]),
                    "iterator_types": ArrayAttr.get(iters),
                    "operandSegmentSizes": DenseI32ArrayAttr.get([len(ins), len(outs)])},
        regions=1, loc=loc)
    blk = gen.regions[0].blocks.append(*[_elem_type(v) for v in list(ins) + list(outs)])
    return gen, blk


def _elementwise_reader_into(g):
    """An all-parallel reader `e` of one of `g`'s tiles, over the same space, every other operand
    of it already there where `g` is: `g` and `e` become one generic at `g`, `g`'s outputs first."""
    from mlir.dialects import linalg
    from mlir.ir import AffineMap, AffineMapAttr, ArrayAttr, InsertionPoint
    rank = rank_of(g.results[0])
    for j, r in enumerate(g.results):
        for e in dict.fromkeys(op_of(u.owner) for u in r.uses):
            if not _is_parallel_maker(e) or rank_of(e.results[0]) != rank:
                continue
            e_in = len(e.operands) - len(e.results)
            emaps = [AffineMapAttr(m).value for m in ArrayAttr(e.attributes["indexing_maps"])]
            reads = [k for k in range(e_in) if e.operands[k] in list(g.results)]
            if any(emaps[k] != AffineMap.get_identity(rank) for k in reads):
                continue
            if any(e.operands[k] not in list(g.results) and not _dominates(e.operands[k], g)
                   for k in range(e_in)):
                continue
            if not all(_dominates(v, g) for v in _captured(e)):
                continue
            # Every output needs a buffer of its own: two on one init (an empty tensor of one type
            # is one value after cse) make bufferization run the op once for each.
            g_outs = list(g.operands)[len(g.operands) - len(g.results):]
            inits = [str(o.type) if _is_empty(o) else o for o in g_outs]
            outs = []
            for o in list(e.operands)[e_in:]:
                key = str(o.type) if _is_empty(o) else o
                if key in inits:
                    break
                inits.append(key)
                if _dominates(o, g):
                    outs.append(o)
                elif _is_empty(o):
                    outs.append(op_of(o.owner).clone(ip=InsertionPoint(g)).results[0])
                else:
                    break
            else:
                g_in = len(g.operands) - len(g.results)
                gmaps = [AffineMapAttr(m).value for m in ArrayAttr(g.attributes["indexing_maps"])]
                others = [k for k in range(e_in) if k not in reads]
                with InsertionPoint(g):
                    new, blk = _generic(
                        list(g.operands)[:g_in] + [e.operands[k] for k in others],
                        list(g.operands)[g_in:] + outs,
                        gmaps[:g_in] + [emaps[k] for k in others] + gmaps[g_in:] + emaps[e_in:],
                        list(ArrayAttr(g.attributes["iterator_types"])), g.location)
                    gb, eb = g.regions[0].blocks[0], e.regions[0].blocks[0]
                    args = list(blk.arguments)
                    with InsertionPoint(blk):
                        made = _inline_body(gb, dict(zip(list(gb.arguments)[:g_in], args[:g_in])))
                        env = {eb.arguments[k]: made[list(g.results).index(e.operands[k])]
                               for k in reads}
                        env.update(zip([eb.arguments[k] for k in others],
                                       args[g_in:g_in + len(others)]))
                        linalg.YieldOp(made + _inline_body(eb, env))
                for a, b in zip(list(g.results) + list(e.results), new.results):
                    a.replace_all_uses_with(b)
                e.erase()
                g.erase()
                return True
    return False


def _reduce_reader_into(g):
    """A `linalg.reduce` over the last dim of one of `g`'s tiles, everything else that reads
    `g` after it: one generic at the reduce, its last dim a reduction, `g`'s outputs written as
    they were and the reduce's init combined into as a row."""
    from mlir.dialects import linalg
    from mlir.ir import (AffineDimExpr, AffineMap, AffineMapAttr, ArrayAttr, Attribute,
                         InsertionPoint)
    rank = rank_of(g.results[0])
    for j, r in enumerate(g.results):
        for red in dict.fromkeys(op_of(u.owner) for u in r.uses):
            if name_of(red) != "linalg.reduce" or len(red.operands) != 2 or red.operands[0] != r \
                    or list(int_array(red.attributes["dimensions"])) != [rank - 1] \
                    or folds_lanes(red):
                continue
            if not all(_after_or_at(op_of(u.owner), red) for x in g.results for u in x.uses):
                continue
            g_in = len(g.operands) - len(g.results)
            if red.operands[1] in list(g.operands)[g_in:]:
                continue
            gmaps = [AffineMapAttr(m).value for m in ArrayAttr(g.attributes["indexing_maps"])]
            kept = AffineMap.get(rank, 0, [AffineDimExpr.get(d) for d in range(rank - 1)])
            with InsertionPoint(red):
                new, blk = _generic(
                    list(g.operands)[:g_in], list(g.operands)[g_in:] + [red.operands[1]],
                    gmaps + [kept],
                    [parallel_iterator()] * (rank - 1)
                    + [Attribute.parse("#linalg.iterator_type<reduction>")], red.location)
            gb, rb = g.regions[0].blocks[0], red.regions[0].blocks[0]
            args = list(blk.arguments)
            with InsertionPoint(blk):
                made = _inline_body(gb, dict(zip(list(gb.arguments)[:g_in], args[:g_in])))
                comb = _inline_body(rb, {rb.arguments[0]: made[j], rb.arguments[1]: args[-1]})
                linalg.YieldOp(made + comb)
            for a, b in zip(list(g.results) + list(red.results), new.results):
                a.replace_all_uses_with(b)
            red.erase()
            g.erase()
            return True
    return False


def _drop_unread_outputs(g):
    """`g` again without the outputs nothing reads, when made into an empty tensor."""
    from mlir.dialects import linalg
    from mlir.ir import AffineMapAttr, ArrayAttr, InsertionPoint
    n_out = len(g.results)
    g_in = len(g.operands) - n_out
    blk = g.regions[0].blocks[0]
    gone = [i for i, r in enumerate(g.results)
            if not list(r.uses) and not list(blk.arguments[g_in + i].uses)
            and name_of(op_of(getattr(g.operands[g_in + i], "owner", None))) == "tensor.empty"]
    if not gone or len(gone) == n_out:
        return
    keep = [i for i in range(n_out) if i not in gone]
    maps = [AffineMapAttr(m).value for m in ArrayAttr(g.attributes["indexing_maps"])]
    with InsertionPoint(g):
        new, nb = _generic(list(g.operands)[:g_in], [g.operands[g_in + i] for i in keep],
                           maps[:g_in] + [maps[g_in + i] for i in keep],
                           list(ArrayAttr(g.attributes["iterator_types"])), g.location)
    args = list(nb.arguments)
    with InsertionPoint(nb):
        env = dict(zip(list(blk.arguments)[:g_in], args[:g_in]))
        env.update({blk.arguments[g_in + i]: args[g_in + n] for n, i in enumerate(keep)})
        made = _inline_body(blk, env)
        linalg.YieldOp([made[i] for i in keep])
    for n, i in enumerate(keep):
        g.results[i].replace_all_uses_with(new.results[n])
    g.erase()


def _rewrite_casts_into_readers(func):
    """A cast-only generic leaves no tile of its own: a reduce reading it takes the cast into its
    body, and every elementwise reader gets a copy of its own, which upstream then fuses."""
    from mlir.ir import InsertionPoint
    for cast in [o for o in walk(func) if _is_cast(o)]:
        for owner, n in [(op_of(u.owner), u.operand_number) for u in cast.results[0].uses]:
            if name_of(owner) == "linalg.reduce" and len(owner.operands) == 2 and n == 0:
                if not folds_lanes(owner):
                    _rewrite_reduce_over_cast(owner, cast)
            elif name_of(owner) == "linalg.generic" and n < len(owner.operands) - len(owner.results):
                owner.operands[n] = cast.clone(ip=InsertionPoint(owner)).results[0]
        if not any(True for _ in cast.results[0].uses):
            cast.erase()


def _rewrite_index_makers_into_readers(func):
    """A generic that reads no tile -- its element made of its indices and constants, as a mask
    is -- leaves no tile of its own: each generic reading it makes the element itself, from its
    own indices through its map of that operand. A mask the reduces of a softmax share is then a
    compare on the index in each reduce, where fit-to-numel sees how far the reduce needs to
    read."""
    for mk in [o for o in walk(func) if _is_index_maker(o)]:
        done = []
        while True:
            uses = [(op_of(u.owner), u.operand_number) for u in mk.results[0].uses]
            todo = [(o, k) for o, k in uses if name_of(o) == "linalg.generic"
                    and k < len(o.operands) - len(o.results) and (o, k) not in done]
            if not todo:
                break
            owner, n = todo[0]
            if not _index_maker_into(owner, n, mk):
                done.append((owner, n))
        if not any(True for _ in mk.results[0].uses):
            mk.erase()


def _is_index_maker(op):
    """An all-parallel generic of one output and no input, written through a permutation of its
    dims, whose body reads only its indices and values from outside."""
    from mlir.ir import AffineMap, AffineMapAttr, ArrayAttr
    if name_of(op) != "linalg.generic" or len(op.operands) != 1 or len(op.results) != 1:
        return False
    if any(str(t) != str(parallel_iterator()) for t in ArrayAttr(op.attributes["iterator_types"])):
        return False
    maps = [AffineMapAttr(m).value for m in ArrayAttr(op.attributes["indexing_maps"])]
    if len(maps) != 1 or not maps[0].is_permutation:
        return False
    blk = op.regions[0].blocks[0]
    if list(blk.arguments[0].uses):
        return False
    return all(i.operation.name in ("linalg.index", "linalg.yield")
               or i.operation.name.startswith("arith.") for i in blk.operations)


def _index_maker_into(reader, n, mk):
    """`reader` again without its operand `n`: its body makes that element with `mk`'s body, each
    `linalg.index d` of it the reader's dim its map gives for d. Left as it is when that map is
    not one dim of the reader per dim of the tile, or `mk` reads a value not there at `reader`."""
    from mlir.ir import (AffineDimExpr, AffineMapAttr, ArrayAttr, InsertionPoint, IntegerAttr,
                         IntegerType, Operation)
    maps = [AffineMapAttr(m).value for m in ArrayAttr(reader.attributes["indexing_maps"])]
    made_by = AffineMapAttr(ArrayAttr(mk.attributes["indexing_maps"])[0]).value
    try:
        at = [AffineDimExpr(e).position for e in maps[n].results]
        # the maker's dim d is the tile's dim i where its (permutation) map puts d
        own = [AffineDimExpr(e).position for e in made_by.results]
        dims = [at[own.index(d)] for d in range(len(own))]
    except ValueError:
        return False
    if not all(_dominates(v, reader) for v in _captured(mk)):
        return False
    n_in = len(reader.operands) - len(reader.results)
    keep = [k for k in range(len(reader.operands)) if k != n]
    with InsertionPoint(reader):
        new, blk = _generic([reader.operands[k] for k in keep if k < n_in],
                            [reader.operands[k] for k in keep if k >= n_in],
                            [maps[k] for k in keep],
                            list(ArrayAttr(reader.attributes["iterator_types"])), reader.location)
    rb, mb = reader.regions[0].blocks[0], mk.regions[0].blocks[0]
    with InsertionPoint(blk):
        env = {}
        made = None
        for inner in mb.operations:
            o = inner.operation
            if o.name == "linalg.yield":
                made = env.get(o.operands[0], o.operands[0])
                break
            attrs = attrs_of(o)
            if o.name == "linalg.index":
                attrs["dim"] = IntegerAttr.get(IntegerType.get_signless(64),
                                               dims[IntegerAttr(o.attributes["dim"]).value])
            got = Operation.create(o.name, results=[r.type for r in o.results],
                                   operands=[env.get(v, v) for v in o.operands],
                                   attributes=attrs, loc=o.location)
            env.update(zip(o.results, got.results))
        env = {rb.arguments[k]: blk.arguments[i] for i, k in enumerate(keep)}
        env[rb.arguments[n]] = made
        from mlir.dialects import linalg
        linalg.YieldOp(_inline_body(rb, env))
    for a, b in zip(reader.results, new.results):
        a.replace_all_uses_with(b)
    reader.erase()
    return True


def _is_cast(op):
    """An elementwise generic of one tile whose body is casts of its element and nothing else."""
    from mlir.ir import AffineMap, AffineMapAttr, ArrayAttr
    if name_of(op) != "linalg.generic" or len(op.operands) != 2 or len(op.results) != 1:
        return False
    maps = [AffineMapAttr(m).value for m in ArrayAttr(op.attributes["indexing_maps"])]
    if any(m != maps[0] or not m.is_permutation for m in maps):
        return False
    blk = op.regions[0].blocks[0]
    cur, steps = blk.arguments[0], [i.operation for i in blk.operations]
    for st in steps[:-1]:
        if st.name not in CASTS or st.operands[0] != cur:
            return False
        cur = st.results[0]
    return len(steps) > 1 and steps[-1].name == "linalg.yield" and steps[-1].operands[0] == cur


def _rewrite_reduce_over_cast(red, cast):
    """`red` again, reading the cast's own input and casting each element before it combines."""
    from mlir.ir import InsertionPoint, Operation
    src, init = cast.operands[0], red.operands[1]
    old = red.regions[0].blocks[0]
    with InsertionPoint(red):
        new = Operation.create("linalg.reduce", results=[r.type for r in red.results],
                               operands=[src, init], attributes=attrs_of(red), regions=1,
                               loc=red.location)
    blk = new.regions[0].blocks.append(cast.regions[0].blocks[0].arguments[0].type,
                                       old.arguments[1].type)
    env = {cast.regions[0].blocks[0].arguments[0]: blk.arguments[0]}
    with InsertionPoint(blk):
        for body, last in ((cast.regions[0].blocks[0], False), (old, True)):
            for inner in body.operations:
                o = inner.operation
                if o.name == "linalg.yield":
                    if last:
                        Operation.create("linalg.yield", operands=[env[v] for v in o.operands])
                    else:
                        env[old.arguments[0]], env[old.arguments[1]] = env[o.operands[0]], blk.arguments[1]
                    break
                got = Operation.create(o.name, results=[r.type for r in o.results],
                                       operands=[env.get(v, v) for v in o.operands],
                                       attributes=attrs_of(o), loc=o.location)
                env.update(zip(o.results, got.results))
    for a, b in zip(red.results, new.results):
        a.replace_all_uses_with(b)
    red.erase()


def _rewrite_array_fences(func):
    for op in [o for o in walk(func) if name_of(o) in _ARRAY_OPS]:
        for k in range(len(op.operands) - len(op.results)):
            if rank_of(op.operands[k]) is not None:
                _rewrite_operand_fenced(op, k)
        for r in op.results:
            if rank_of(r) is not None:
                _rewrite_result_fenced(r)


def _rewrite_reshape_fences(func):
    for op in [o for o in walk(func) if name_of(o) in _RESHAPES]:
        _rewrite_operand_fenced(op, 0)
        _rewrite_result_fenced(op.results[0])


def _rewrite_operand_fenced(op, k):
    from mlir.ir import InsertionPoint, Operation
    v = op.operands[k]
    with InsertionPoint(op_of(op)):
        f = Operation.create(FENCE, results=[v.type], operands=[v])
    op.operands[k] = f.results[0]


def _rewrite_result_fenced(r):
    from mlir.ir import InsertionPoint, Operation
    uses = [(u.owner, u.operand_number) for u in r.uses]
    if not uses:
        return
    with InsertionPoint.after(op_of(r.owner)):
        f = Operation.create(FENCE, results=[r.type], operands=[r])
    for owner, n in uses:
        owner.operands[n] = f.results[0]


def _erase_fences(func):
    for f in [o for o in walk(func) if name_of(o) == FENCE]:
        f.results[0].replace_all_uses_with(f.operands[0])
        op_of(f).erase()


def _clean_up(module):
    from mlir.passmanager import PassManager
    PassManager.parse("builtin.module(func.func(canonicalize,cse))").run(module.operation)
