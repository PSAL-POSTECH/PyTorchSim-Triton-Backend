"""pytorchsim-pipeline-reductions: a reduce over a whole-tile load runs in TORCHSIM_PIPE_STAGES chunks (1, the
default, leaves it whole), each trip folding its chunk into a vector of partials a row so the next load
overlaps this arithmetic; the partials fold across the vector once, after the loop."""

import os

from .lib_bank_layout import is_a_scratchpad_buffer
from .lib_ir import ends_lifetime, funcs, int_array, op_of, producer, walk
from . import pass_name

NAME = pass_name(__name__)

_STAGES = int(os.environ.get("TORCHSIM_PIPE_STAGES", 1))   # 1: off unless a build asks
_GRAIN = 64
_TRANSFER = "pytorchsim.transfer"


def run(module, **_opts):
    if _STAGES <= 1:
        return module                   # asked to reduce every tile whole
    for func in funcs(module):
        for gen in [o for o in walk(func.operation) if _reduces_last_dim(o)]:
            _pipeline(module, gen)
    return module


def _reduces_last_dim(o):
    from mlir.ir import ArrayAttr
    if o.name != "linalg.generic":
        return False
    its = [str(t) for t in ArrayAttr(o.attributes["iterator_types"])]
    return its.count("#linalg.iterator_type<reduction>") == 1 and "reduction" in its[-1]


def _pipeline(module, gen):
    from mlir.ir import (AffineMap, AffineMapAttr, ArrayAttr, FlatSymbolRefAttr, IndexType,
                         InsertionPoint, IntegerAttr, MemRefType, StringAttr, TypeAttr)
    from mlir.dialects import arith, scf
    n_in = gen.attributes["operandSegmentSizes"][0]
    ins = list(gen.operands[:n_in])
    maps = [AffineMapAttr(m).value for m in ArrayAttr(gen.attributes["indexing_maps"])]
    rank = len(maps[0].results) if maps else 0
    if not ins or any(m != AffineMap.get_identity(rank) for m in maps[:n_in]):
        return
    block = gen.operation.block
    loads = []
    for v in ins:
        if not is_a_scratchpad_buffer(v):
            return
        g = producer(v)
        users = [op_of(u.owner) for u in v.uses if not ends_lifetime(op_of(u.owner), v)]
        tx = [u for u in users if u.name == _TRANSFER]
        if len(users) != 2 or len(tx) != 1 or not _whole_load(tx[0], v):
            return
        if not tx[0].operation.is_before_in_block(gen.operation):
            return
        loads.append((v, g, tx[0]))
    shape = list(MemRefType(ins[0].type).shape)
    k = shape[-1]
    trips = _STAGES
    if k % trips or (k // trips) % _GRAIN:
        return
    chunk = k // trips
    idx_ty = IndexType.get()
    globals_ = {str(o.attributes["sym_name"]).strip('"'): o for o in walk(module.operation)
                if o.name == "memref.global"}
    new_bufs = {}
    resident = {v for v, g, tx in loads if _rows_share_dram(tx)}
    steps = {v: _chunk_step(tx, chunk, trips) for v, g, tx in loads if v not in resident}
    if None in steps.values():
        return
    for v, g, tx in loads:
        if v in resident:
            continue
        ty = MemRefType(v.type)
        cty = MemRefType.get(shape[:-1] + [chunk], ty.element_type, memory_space=ty.memory_space)
        if g.name != "memref.get_global":
            new_bufs[v] = _allocated_like(g, cty)
            continue
        old = FlatSymbolRefAttr(g.attributes["name"]).value
        name = old.replace("_spad_", "_chunk_spad_") if "_spad_" in old else old + "_chunk"
        glob = globals_[old].clone(ip=InsertionPoint(globals_[old]))
        glob.attributes["sym_name"] = StringAttr.get(name)
        glob.attributes["type"] = TypeAttr.get(cty)
        get = g.clone(ip=InsertionPoint(g))
        get.attributes["name"] = FlatSymbolRefAttr.get(name)
        get.results[0].set_type(cty)
        new_bufs[v] = get.results[0]
    comb = _combiner(gen, n_in)
    if comb is None:
        return
    out = gen.operands[n_in]
    oty = MemRefType(out.type)
    pty = MemRefType.get(shape[:-1] + [_GRAIN], oty.element_type, memory_space=oty.memory_space)
    if loads[0][1].name != "memref.get_global":
        part = _allocated_like(loads[0][1], pty)
    else:
        part_name = (FlatSymbolRefAttr(producer(ins[0]).attributes["name"]).value
                     .replace("_spad_", "_part_spad_"))
        first = globals_[FlatSymbolRefAttr(loads[0][1].attributes["name"]).value]
        pglob = first.clone(ip=InsertionPoint(first))
        pglob.attributes["sym_name"] = StringAttr.get(part_name)
        pglob.attributes["type"] = TypeAttr.get(pty)
        pget = loads[0][1].clone(ip=InsertionPoint(loads[0][1]))
        pget.attributes["name"] = FlatSymbolRefAttr.get(part_name)
        pget.results[0].set_type(pty)
        part = pget.results[0]
    with InsertionPoint(gen):
        _fill(part, comb, oty.element_type)
        lo = arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 0)).result
        hi = arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, trips)).result
        one = arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 1)).result
        loop = scf.ForOp(lo, hi, one)
    with InsertionPoint(loop.body):
        iv = loop.induction_variable
        for v, _g, tx in loads:
            if v in resident:
                continue
            stride = arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, steps[v])).result
            off = arith.AddIOp(tx.operands[1], arith.MulIOp(iv, stride).result).result
            t = tx.clone()
            t.operands[1] = off
            t.operands[2] = new_bufs[v]
            _clamp_last_axis(t, chunk, rank)
        views = [_split_last(_chunk_view(v, iv, chunk) if v in resident else new_bufs[v],
                             chunk // _GRAIN) for v in ins]
        _generic(gen.regions[0].blocks[0], views, part, rank + 1, rank - 1)
        scf.YieldOp([])
    with InsertionPoint(gen):
        _fold(part, out, comb, rank)
    for v, g, tx in loads:
        if v not in resident:
            tx.erase()
    gen.erase()
    for v, g, tx in loads:
        if v in resident:
            continue
        if g.name != "memref.get_global":
            ends = [op_of(u.owner) for u in v.uses]
            if all(ends_lifetime(e, v) for e in ends):
                for e in ends:
                    e.erase()
                g.erase()
            continue
        name = FlatSymbolRefAttr(g.attributes["name"]).value
        g.erase()
        if not any(FlatSymbolRefAttr(o.attributes["name"]).value == name
                   for o in walk(module.operation) if o.name == "memref.get_global"):
            globals_[name].erase()


def _allocated_like(alloc, ty):
    """A buffer of type `ty` allocated where `alloc` is, its life ended wherever `alloc`'s is."""
    from mlir.ir import InsertionPoint
    got = alloc.clone(ip=InsertionPoint(alloc))
    got.results[0].set_type(ty)
    for use in list(alloc.results[0].uses):
        end = op_of(use.owner)
        if ends_lifetime(end, alloc.results[0]):
            twin = end.clone(ip=InsertionPoint(end))
            twin.operands[use.operand_number] = got.results[0]
    return got.results[0]


_IDENTITY = {"arith.addf": 0.0, "arith.maxnumf": float("-inf"), "arith.minnumf": float("inf"),
             "arith.maximumf": float("-inf"), "arith.minimumf": float("inf")}


def _combiner(gen, n_in):
    """The op that combines into the output, when it is one this knows the identity of and the
    body uses no index (a chunk's index is not the tile's)."""
    body = gen.regions[0].blocks[0]
    steps = [x.operation for x in body.operations]
    if any(o.name == "linalg.index" and list(o.results[0].uses) for o in steps) or len(steps) < 2:
        return None
    comb = steps[-2]
    if comb.name not in _IDENTITY or steps[-1].operands[0] != comb.results[0] \
            or body.arguments[n_in] not in list(comb.operands):
        return None
    return comb.name


def _fill(buf, comb, elem):
    from mlir.dialects import arith, linalg
    from mlir.ir import FloatAttr
    c = arith.ConstantOp(elem, FloatAttr.get(elem, _IDENTITY[comb])).result
    linalg.fill(c, outs=[buf])


def _rows_share_dram(tx):
    """A load whose rows all read the same DRAM (stride 0 along its first axis): a vector
    broadcast over the rows. Its chunks are one DRAM range a trip, so it is loaded once, whole,
    ahead of the loop, and each trip reads its chunk of it -- one DMA instead of one a trip."""
    strides = list(int_array(tx.attributes["dram_stride"]))
    return len(strides) == 2 and strides[0] == 0


def _chunk_step(tx, chunk, trips):
    """What a trip adds to the load's dram_idx: the translation of its last axis by a chunk, read
    through that axis's cut digits (a plain stride being one affine digit); None when it is none."""
    from ...dialects.pytorchsim import Digit, cut_translation
    from .lib_descriptor import read_cut
    strides = list(int_array(tx.attributes["dram_stride"]))
    digits = (read_cut(tx) or {}).get(len(strides) - 1, (Digit(0, strides[-1], 0, 0),))
    return cut_translation(digits, chunk, trips)


def _chunk_view(buf, iv, chunk):
    """`buf` [.., k] seen at its trip-`iv` chunk: [.., chunk] from column iv * chunk."""
    from mlir.dialects import arith, memref
    from mlir.ir import (DenseI64ArrayAttr, IndexType, IntegerAttr, MemRefType, ShapedType,
                         StridedLayoutAttr)
    ty = MemRefType(buf.type)
    shape = list(ty.shape)
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    dyn = ShapedType.get_dynamic_stride_or_offset()
    idx = IndexType.get()
    off = arith.MulIOp(iv, arith.ConstantOp(idx, IntegerAttr.get(idx, chunk)).result).result
    rty = MemRefType.get(shape[:-1] + [chunk], ty.element_type,
                         layout=StridedLayoutAttr.get(dyn, strides), memory_space=ty.memory_space)
    return memref.SubViewOp(rty, buf, [off], [], [],
                            DenseI64ArrayAttr.get([0] * (len(shape) - 1) + [dyn]),
                            DenseI64ArrayAttr.get(shape[:-1] + [chunk]),
                            DenseI64ArrayAttr.get([1] * len(shape))).result


def _split_last(buf, n):
    """`buf` [.., k] viewed as [.., n, k / n]."""
    from mlir.dialects import memref
    from mlir.ir import ArrayAttr, IntegerAttr, IntegerType, MemRefType
    ty = MemRefType(buf.type)
    shape = list(ty.shape)
    out = shape[:-1] + [n, shape[-1] // n]
    i64 = IntegerType.get_signless(64)
    groups = [[i] for i in range(len(shape) - 1)] + [[len(shape) - 1, len(shape)]]
    reassoc = ArrayAttr.get([ArrayAttr.get([IntegerAttr.get(i64, i) for i in g]) for g in groups])
    layout = None
    if "strided" in str(ty.layout):
        from mlir.ir import ShapedType, StridedLayoutAttr
        sl = StridedLayoutAttr(ty.layout)
        st = list(sl.strides)
        layout = StridedLayoutAttr.get(sl.offset, st[:-1] + [st[-1] * (shape[-1] // n), st[-1]])
    rty = MemRefType.get(out, ty.element_type, layout=layout, memory_space=ty.memory_space)
    return memref.ExpandShapeOp(rty, buf, reassoc, [], out).result


def _generic(body, ins, out, rank, red):
    """A generic over `rank` dims reducing dim `red`: identity maps for `ins`, the out without
    `red`, and a copy of `body`."""
    from mlir.ir import (AffineDimExpr, AffineMap, AffineMapAttr, ArrayAttr, Attribute,
                         DenseI32ArrayAttr, InsertionPoint, Operation)
    from .lib_ir import attrs_of, parallel_iterator
    kept = AffineMap.get(rank, 0, [AffineDimExpr.get(d) for d in range(rank) if d != red])
    gen = Operation.create(
        "linalg.generic", results=[], operands=list(ins) + [out],
        attributes={
            "indexing_maps": ArrayAttr.get([AffineMapAttr.get(AffineMap.get_identity(rank))] * len(ins)
                                           + [AffineMapAttr.get(kept)]),
            "iterator_types": ArrayAttr.get(
                [Attribute.parse("#linalg.iterator_type<reduction>") if d == red
                 else parallel_iterator() for d in range(rank)]),
            "operandSegmentSizes": DenseI32ArrayAttr.get([len(ins), 1])},
        regions=1)
    blk = gen.regions[0].blocks.append(*[a.type for a in body.arguments])
    env = dict(zip(list(body.arguments), list(blk.arguments)))
    with InsertionPoint(blk):
        for inner in body.operations:
            o = inner.operation
            got = Operation.create(o.name, results=[r.type for r in o.results],
                                   operands=[env.get(v, v) for v in o.operands],
                                   attributes=attrs_of(o))
            env.update(zip(o.results, got.results))
    return gen


def _fold(part, out, comb, rank):
    """The partials [.., _GRAIN] folded over their last dim into `out`, which holds the init."""
    from mlir.ir import Block, InsertionPoint, MemRefType, Operation
    elem = MemRefType(part.type).element_type
    import mlir.ir as ir
    region_op = Operation.create("scf.execute_region", results=[], regions=1)
    blk = region_op.regions[0].blocks.append(elem, elem)
    with InsertionPoint(blk):
        c = Operation.create(comb, results=[elem], operands=[blk.arguments[0], blk.arguments[1]])
        Operation.create("linalg.yield", operands=[c.results[0]])
    gen = _generic(blk, [part], out, rank, rank - 1)
    region_op.erase()
    return gen


def _whole_load(tx, buf):
    """A load of the whole of `buf` at its start: no gather, subtile or piece, its last axis unmasked or
    clamped to its full extent from 0. A cut may stay: a trip moves only the last axis (_chunk_step)."""
    from mlir.ir import MemRefType
    from ...dialects.pytorchsim import is_load, operand_number
    if not is_load(tx) or tx.operands[operand_number(tx, "sram")] != buf:
        return False
    for a in ("subtile_size", "indirect_dim", "dyn_fill", "async", "piece_box"):
        if a in tx.attributes:
            return False
    if _const(tx.operands[operand_number(tx, "sram_idx")]) != 0:
        return False
    rank = len(MemRefType(buf.type).shape)
    if "masked_axes" in tx.attributes:
        axes = list(int_array(tx.attributes["masked_axes"]))
        if rank - 1 in axes:
            clamp = list(tx.opview.clamp)
            i = axes.index(rank - 1)
            if _const(clamp[2 * i]) != 0 or _const(clamp[2 * i + 1]) != MemRefType(buf.type).shape[-1]:
                return False
    return True


def _clamp_last_axis(tx, chunk, rank):
    """A chunk load's clamp of the last axis is the chunk: the dim is a multiple of it."""
    from mlir.dialects import arith
    from mlir.ir import IndexType, InsertionPoint, IntegerAttr
    if "masked_axes" not in tx.attributes:
        return
    axes = list(int_array(tx.attributes["masked_axes"]))
    if rank - 1 not in axes:
        return
    from ...dialects.pytorchsim import operand_number
    i = axes.index(rank - 1)
    at = operand_number(tx, "clamp", 2 * i + 1)
    with InsertionPoint(tx):
        tx.operands[at] = arith.ConstantOp(IndexType.get(),
                                           IntegerAttr.get(IndexType.get(), chunk)).result


def _const(v):
    from .lib_ir import const_int
    c = const_int(v)
    if c is not None:
        return c
    o = producer(v)
    if o is not None and o.name in ("arith.index_cast", "arith.extsi"):
        return _const(o.operands[0])
    return None
