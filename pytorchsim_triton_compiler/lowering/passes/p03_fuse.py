"""torchsim-fuse: collapse the elementwise linalg chain into one op, AFTER the layouts are decided,
as upstream fuses after layout assignment -- a broadcast included. A cast that several ops read is
re-made at each reader, and one a reduce reads goes into the reduce's body, so a widened copy of a
tile is never a walk of the scratchpad on its own. An elementwise op that only a reduce reads then
goes into the reduce: one generic with a reduction dim, whose body makes each element and combines
it, so the made tile is never written to the scratchpad and read back."""

from .lib_ir import (attrs_of, funcs, int_array, name_of, op_of, parallel_iterator, rank_of,
                     walk)

FENCE = "torchsim.fusion_fence"

FORBIDS = (FENCE,)

_ARRAY_OPS = ("torchsim.dot",)

_RESHAPES = ("tensor.collapse_shape", "tensor.expand_shape", "tensor.reshape")


_CASTS = ("arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp", "arith.fptosi",
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
    if dims != [rank - 1]:
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


def _rewrite_casts_into_readers(func):
    """A cast-only generic leaves no tile of its own: a reduce reading it takes the cast into its
    body, and every elementwise reader gets a copy of its own, which upstream then fuses."""
    from mlir.ir import InsertionPoint
    for cast in [o for o in walk(func) if _is_cast(o)]:
        for owner, n in [(op_of(u.owner), u.operand_number) for u in cast.results[0].uses]:
            if name_of(owner) == "linalg.reduce" and len(owner.operands) == 2 and n == 0:
                _rewrite_reduce_over_cast(owner, cast)
            elif name_of(owner) == "linalg.generic" and n < len(owner.operands) - len(owner.results):
                owner.operands[n] = cast.clone(ip=InsertionPoint(owner)).results[0]
        if not any(True for _ in cast.results[0].uses):
            cast.erase()


def _is_cast(op):
    """An elementwise generic of one tile whose body is casts of its element and nothing else."""
    from mlir.ir import AffineMap, AffineMapAttr, ArrayAttr
    if name_of(op) != "linalg.generic" or len(op.operands) != 2 or len(op.results) != 1:
        return False
    same = AffineMap.get_identity(rank_of(op.results[0]))
    if any(AffineMapAttr(m).value != same for m in ArrayAttr(op.attributes["indexing_maps"])):
        return False
    blk = op.regions[0].blocks[0]
    cur, steps = blk.arguments[0], [i.operation for i in blk.operations]
    for st in steps[:-1]:
        if st.name not in _CASTS or st.operands[0] != cur:
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
