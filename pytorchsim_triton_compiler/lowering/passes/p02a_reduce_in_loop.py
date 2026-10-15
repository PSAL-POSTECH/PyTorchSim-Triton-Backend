"""tritonpytorchsim-reduce-in-loop: a loop that combines an operand into its carried tensor elementwise,
whose result only a reduce with the same combiner reads, carries the reduced tensor and reduces inside.
Reassociates; both seeds must be one constant c with c combined with c folding back to c."""
from mlir.dialects import func, scf
from mlir.ir import (ArrayAttr, BlockArgument, DenseI32ArrayAttr, InsertionPoint, Module, Operation,
                     ShapedType)
from mlir.passmanager import PassManager

from .lib_ir import (elementwise_generic, funcs, operand_segments, parallel_iterator, producer, splat,
                     walk)

NAME = "tritonpytorchsim-reduce-in-loop"


def _only_use(v):
    uses = list(v.uses)
    return uses[0] if len(uses) == 1 else None


def _combiner(block, a, b):
    """The one op of `block` that combines block arguments `a` and `b` into what it yields."""
    c = _combiner_with(block, a)
    return c if c is not None and any(_key(x) == _key(b) for x in c.operands) else None


def _key(v):
    return (v.owner, v.arg_number) if isinstance(v, BlockArgument) else v


def _seed(v):
    """The constant every element of `v` holds, as (scalar value, attribute), or None."""
    fill = producer(v)
    if fill is None or fill.name != "linalg.fill":
        return None
    c = producer(fill.operands[0])
    if c is None or c.name != "arith.constant":
        return None
    return fill.operands[0], c.attributes["value"]


def _idempotent(comb, scalar, attr):
    """Whether combining `scalar` with itself folds to `attr`, asked of MLIR's own folders."""
    m = Module.create()
    with InsertionPoint(m.body):
        f = func.FuncOp("probe", ([], [scalar.type]))
        with InsertionPoint(f.add_entry_block()):
            k = producer(scalar).clone()
            c = comb.clone()
            c.operands[0] = k.result
            c.operands[1] = k.result
            func.ReturnOp([c.results[0]])
    PassManager.parse("builtin.module(func.func(canonicalize))").run(m.operation)
    ret = list(f.entry_block.operations)[-1]
    got = producer(ret.operands[0])
    return got is not None and got.name == "arith.constant" and got.attributes["value"] == attr


def _all_parallel(op):
    return all(t == parallel_iterator() for t in ArrayAttr(op.attributes["iterator_types"]))


def _match(red):
    """(loop, slot, the generic, its acc input, its combiner, seed) for `red`, or None."""
    if len(red.operands) != 2:
        return None
    loop = producer(red.operands[0])
    if loop is None or loop.name != "scf.for":
        return None
    slot = red.operands[0].result_number
    if _only_use(loop.results[slot]) is None:
        return None
    body = loop.regions[0].blocks[0]
    acc = body.arguments[1 + slot]
    gen = producer(list(body.operations)[-1].operands[slot])
    use = _only_use(acc)
    if gen is None or gen.name != "linalg.generic" or use is None or use.owner != gen:
        return None
    n_in, n_out = operand_segments(gen)
    j = use.operand_number
    maps = ArrayAttr(gen.attributes["indexing_maps"])
    if n_out != 1 or j >= n_in or not _all_parallel(gen) or maps[j] != maps[n_in]:
        return None
    if _only_use(gen.results[0]) is None:
        return None
    gb = gen.regions[0].blocks[0]
    comb = _combiner_with(gb, gb.arguments[j])
    rb = red.regions[0].blocks[0]
    rcomb = _combiner(rb, rb.arguments[0], rb.arguments[1])
    if comb is None or rcomb is None or comb.name != rcomb.name or len(list(gb.arguments[j].uses)) != 1:
        return None
    if {k: comb.attributes[k] for k in comb.attributes} != {k: rcomb.attributes[k] for k in rcomb.attributes}:
        return None
    s0, s1 = _seed(loop.operands[3 + slot]), _seed(red.operands[1])
    if s0 is None or s1 is None or s0[1] != s1[1] or not _idempotent(comb, *s0):
        return None
    return loop, slot, gen, j, comb, s0[0]


def _combiner_with(block, a):
    """The op combining block argument `a` with one other value into what `block` yields."""
    y = list(block.operations)[-1]
    c = producer(y.operands[0]) if len(y.operands) == 1 else None
    if c is None or len(c.operands) != 2 or len(c.results) != 1:
        return None
    return c if sum(_key(x) == _key(a) for x in c.operands) == 1 else None


def _elementwise_part(gen, j, comb):
    """`gen` without input `j`, yielding what its combiner combined that input with."""
    n_in, _ = operand_segments(gen)
    gb = gen.regions[0].blocks[0]
    keep = [k for k in range(len(gen.operands)) if k != j]
    maps = ArrayAttr(gen.attributes["indexing_maps"])
    ins = [gen.operands[k] for k in keep[:n_in - 1]]
    empty = Operation.create("tensor.empty", results=[gen.results[0].type]).results[0]
    new = Operation.create(
        "linalg.generic", results=[gen.results[0].type], operands=ins + [empty],
        attributes={"indexing_maps": ArrayAttr.get([maps[k] for k in keep]),
                    "iterator_types": gen.attributes["iterator_types"],
                    "operandSegmentSizes": DenseI32ArrayAttr.get([n_in - 1, 1])},
        regions=1, loc=gen.location)
    nb = new.regions[0].blocks.append(*[gb.arguments[k].type for k in keep])
    mapping = {_key(gb.arguments[k]): nb.arguments[i] for i, k in enumerate(keep)}
    other = next(x for x in comb.operands if _key(x) != _key(gb.arguments[j]))
    with InsertionPoint(nb):
        for o in list(gb.operations)[:-1]:
            if o == comb:
                continue
            c = o.clone()
            for k, v in enumerate(list(c.operands)):
                if _key(v) in mapping:
                    c.operands[k] = mapping[_key(v)]
            for r_old, r_new in zip(o.results, c.results):
                mapping[_key(r_old)] = r_new
        Operation.create("linalg.yield", operands=[mapping.get(_key(other), other)])
    return new


def _combine(comb, a, b):
    """`comb` cloned to combine `a` with `b`."""
    c = comb.clone()
    c.operands[0], c.operands[1] = a, b
    return c.results[0]


def _erase_unused(v):
    """The fill that made `v`, and the empty it filled, once nothing reads them."""
    fill = producer(v)
    if fill is None or list(v.uses):
        return
    dest = fill.operands[1]
    fill.erase()
    if producer(dest) is not None and not list(dest.uses):
        producer(dest).erase()


def _rewrite(red, loop, slot, gen, j, comb, scalar):
    seeds = [loop.operands[3 + slot], red.operands[1]]
    with InsertionPoint(loop), loop.location:
        seed = splat(scalar, red.results[0], loop.location)
        inits = list(loop.operands[3:])
        inits[slot] = seed
        new = scf.ForOp(loop.operands[0], loop.operands[1], loop.operands[2], inits)
    old_body, nb = loop.regions[0].blocks[0], new.body
    old_yield = list(old_body.operations)[-1]
    for o in list(old_body.operations)[:-1]:
        nb.append(o)
    for a, b in zip(old_body.arguments, nb.arguments):
        a.replace_all_uses_with(b)
    with InsertionPoint(nb), loop.location:
        part = _elementwise_part(gen, j, comb)
        init = splat(scalar, red.results[0], loop.location)
        r = red.clone()
        r.operands[0] = part.results[0]
        r.operands[1] = init
        acc = elementwise_generic([nb.arguments[1 + slot], r.results[0]], ShapedType(r.results[0].type)
                                  .element_type, lambda a, b: _combine(comb, a, b), loop.location)
        ys = list(old_yield.operands)
        ys[slot] = acc
        scf.YieldOp(ys)
    old_yield.erase()
    gen.erase()
    red.results[0].replace_all_uses_with(new.results[slot])
    for k, res in enumerate(loop.results):
        if k != slot:
            res.replace_all_uses_with(new.results[k])
    red.erase()
    loop.erase()
    for v in seeds:
        _erase_unused(v)


def run(module, **_opts):
    """Every reduce is read for the shape above; anything else is left as it was."""
    for f in funcs(module):
        for red in [o for o in walk(f.operation) if o.name == "linalg.reduce"]:
            found = _match(red)
            if found:
                _rewrite(red, *found)
    return module
