"""tritonpytorchsim-lower-runtime-attributes: each record a store into the program's row of the hidden
output, the external global ROW the wrapper hands every program, and each read point its value; the
slot table goes on the module as TABLE, which the binary step writes into the manifest."""
import json
import math

from ...contract.kernel_object import RUNTIME_ATTRIBUTE_HEAD, RUNTIME_ATTRIBUTE_ROW
from ...dialects.torchsim import TransferOp, moved_shape
from .lib_ir import walk
from .p11a_mark_runtime_attributes import PLUMBING, RECORD, SOURCES, erase_records, records

NAME = "tritonpytorchsim-lower-runtime-attributes"
MARKERS = (RECORD,)

TABLE = "torchsim.runtime_attributes"
"""The module attribute stating the row: {"row_words", "attributes": [{id, rank, offset, capacity,
fill, elem_bits, sources}]}, as JSON; `sources` the function arguments (by number) a record is read from."""


def run(module, *, timing=False, **_):
    """Lower every record of `module`; the timed binary (`timing`) has none, as if never marked."""
    recs = records(module)
    if not recs:
        return
    if timing:
        erase_records(module)
        return
    table, row_words = slot_table(recs)
    _Lowering(module, table, row_words).lower(recs)
    import mlir.ir as ir
    module.operation.attributes[TABLE] = ir.StringAttr.get(json.dumps(
        {"row_words": row_words, "attributes": [table[k] for k in sorted(table)]}))


def slot_table(recs):
    """{id: slot} of every record id and the row's words: past the head, per id in id order, its
    count and fill words, its extents, then the values its capacity holds, each `elem_bits` wide."""
    import mlir.ir as ir
    by_id = {}
    for o in recs:
        i = ir.IntegerAttr(o.attributes["id"]).value
        form = ir.StringAttr(o.attributes["form"]).value
        got = by_id.setdefault(i, {"id": i, "rank": ir.IntegerAttr(o.attributes["rank"]).value,
                                   "capacity": [], "fill": None, "elem_bits": 64, "sources": []})
        if SOURCES not in o.attributes:
            raise RuntimeError(f"runtime attribute {i} has a record that states no {SOURCES}")
        got["sources"] = sorted(set(got["sources"]) | set(ir.DenseI64ArrayAttr(o.attributes[SOURCES])))
        if form in ("A", "G"):
            got["capacity"] = list(ir.DenseI64ArrayAttr(o.attributes["capacity"]))
        if form == "G":
            got["fill"] = len(list(o.opview.extents))
            got["elem_bits"] = ir.IntegerType(ir.MemRefType(o.opview.source.type).element_type).width
    if sorted(by_id) != list(range(len(by_id))):
        raise RuntimeError(f"runtime attribute ids {sorted(by_id)} are not 0..{len(by_id) - 1}")
    offset = RUNTIME_ATTRIBUTE_HEAD
    for i in sorted(by_id):
        by_id[i]["offset"] = offset
        offset += 2 + by_id[i]["rank"] + -(-math.prod(by_id[i]["capacity"]) * by_id[i]["elem_bits"] // 64)
    return by_id, offset


class _Lowering:
    """The stores one module's records lower to, through the row global."""

    def __init__(self, module, table, row_words):
        import mlir.ir as ir
        self.ir = ir
        self.module = module
        self.table = table
        self.i64 = ir.IntegerType.get_signless(64)
        self.idx = ir.IndexType.get()
        self.bytes_type = ir.MemRefType.get([row_words * 8], ir.IntegerType.get_signless(8))
        self.row_type = ir.MemRefType.get([row_words], self.i64)
        with ir.InsertionPoint.at_block_begin(module.body), ir.Location.unknown():
            ir.Operation.create("memref.global", attributes={
                "sym_name": ir.StringAttr.get(RUNTIME_ATTRIBUTE_ROW),
                "type": ir.TypeAttr.get(self.bytes_type)})

    def lower(self, recs):
        for o in recs:
            form = self.ir.StringAttr(o.attributes["form"]).value
            slot = self.table[self.ir.IntegerAttr(o.attributes["id"]).value]
            with self.ir.InsertionPoint(o), o.location:
                if form == "S":
                    row = self.row()
                    self.store(row, slot["offset"] + 2, self.as_i64(o.operands[0]))
                    self.count(row, slot["offset"])
                elif form == "A":
                    self.array(o, slot)
                elif form == "G":
                    self.snapshot(o, slot)
            if len(o.results):
                o.results[0].replace_all_uses_with(o.operands[0])
            o.erase()
        for o in reversed([o for o in walk(self.module.operation) if PLUMBING in o.attributes]):
            if all(not list(r.uses) for r in o.results):
                o.erase()

    def bytes(self):
        from mlir.dialects import memref
        return memref.GetGlobalOp(self.bytes_type, RUNTIME_ATTRIBUTE_ROW).result

    def row(self):
        from mlir.dialects import memref
        return memref.ViewOp(self.row_type, self.bytes(), self.cst(0), []).result

    def cst(self, n, ty=None):
        from mlir.dialects import arith
        ty = ty or self.idx
        return arith.ConstantOp(ty, self.ir.IntegerAttr.get(ty, n)).result

    def as_i64(self, v):
        from mlir.dialects import arith
        if isinstance(v.type, self.ir.IndexType):
            return arith.IndexCastOp(self.i64, v).result
        width = self.ir.IntegerType(v.type).width
        if width == 64:
            return v
        return arith.ExtSIOp(self.i64, v).result if width < 64 else arith.TruncIOp(self.i64, v).result

    def store(self, row, at, value):
        from mlir.dialects import memref
        memref.StoreOp(value, row, [self.cst(at) if isinstance(at, int) else at])

    def count(self, row, at):
        """One more execution of the record at word `at`."""
        from mlir.dialects import arith, memref
        k = self.cst(at)
        got = memref.LoadOp(row, [k]).result
        memref.StoreOp(arith.AddIOp(got, self.cst(1, self.i64)).result, row, [k])

    def array(self, o, slot):
        """The record counted, its extents written, and each element of `source[base + sum(t_d *
        strides_d)]` below its capacity written packed row-major by those extents."""
        from mlir.dialects import arith, memref, scf
        ir = self.ir
        rec = o.opview
        strides = list(ir.DenseI64ArrayAttr(o.attributes["strides"]))
        cap = math.prod(slot["capacity"])
        extents = list(rec.extents)
        row = self.row()
        self.count(row, slot["offset"])
        for d, e in enumerate(extents):
            self.store(row, slot["offset"] + 2 + d, arith.IndexCastOp(self.i64, e).result)
        values = slot["offset"] + 2 + slot["rank"]

        def body(trips):
            lin, at = self.cst(0), rec.base
            for t, e, s in zip(trips, extents, strides):
                lin = arith.AddIOp(arith.MulIOp(lin, e).result, t).result
                at = arith.AddIOp(at, arith.MulIOp(t, self.cst(s)).result).result
            guard = scf.IfOp(arith.CmpIOp(arith.CmpIPredicate.slt, lin, self.cst(cap)).result)
            with ir.InsertionPoint(guard.then_block):
                v = memref.LoadOp(rec.source, [at]).result
                self.store(row, arith.AddIOp(lin, self.cst(values)).result, self.as_i64(v))
                scf.YieldOp([])

        def nest(k, trips):
            if k == len(extents):
                body(trips)
                return
            loop = scf.ForOp(self.cst(0), extents[k], self.cst(1))
            with ir.InsertionPoint(loop.body):
                nest(k + 1, trips + [loop.induction_variable])
                scf.YieldOp([])

        nest(0, [])

    def snapshot(self, o, slot):
        """At trip t of the loops around a gather: on the first, the record counted and its extents
        written; on every one, its fill counted and row t written: the index tile by one DMA, or, for
        a masked gather, entry by entry with 0 where the clamp masks the element."""
        from mlir.dialects import arith, memref, scf
        ir = self.ir
        rec = o.opview
        src = ir.MemRefType(rec.source.type)
        shape, bits = list(src.shape), slot["elem_bits"]
        trips, extents = list(rec.indices), list(rec.extents)
        tail = slot["capacity"][len(trips):]
        n = math.prod(tail)
        tx = self.gather_of(o)
        row = self.row()
        first = arith.ConstantOp(ir.IntegerType.get_signless(1), ir.IntegerAttr.get(
            ir.IntegerType.get_signless(1), 1)).result
        for t in trips:
            first = arith.AndIOp(first, arith.CmpIOp(arith.CmpIPredicate.eq, t, self.cst(0)).result).result
        once = scf.IfOp(first)
        with ir.InsertionPoint(once.then_block):
            self.count(row, slot["offset"])
            for d, e in enumerate(extents):
                self.store(row, slot["offset"] + 2 + d, arith.IndexCastOp(self.i64, e).result)
            for k, e in enumerate(slot["capacity"][len(trips):]):
                self.store(row, slot["offset"] + 2 + len(extents) + k, self.cst(e, self.i64))
            scf.YieldOp([])
        self.count(row, slot["offset"] + 1)
        lin = self.cst(0)
        for t, e in zip(trips, extents):
            lin = arith.AddIOp(arith.MulIOp(lin, e).result, t).result
        fits = arith.CmpIOp(arith.CmpIPredicate.slt, lin,
                            self.cst(math.prod(slot["capacity"][:len(trips)]))).result
        guard = scf.IfOp(fits)
        with ir.InsertionPoint(guard.then_block):
            at = arith.AddIOp(self.cst((slot["offset"] + 2 + slot["rank"]) * 8),
                              arith.MulIOp(lin, self.cst(n * bits // 8)).result).result
            view = memref.ViewOp(ir.MemRefType.get([n], src.element_type), self.bytes(), at, []).result
            i64 = self.i64
            axis = None if tx.indirect_dim is None else tx.indirect_dim.value
            head = n if axis is None else tail[axis] if len(tail) > 1 else tail[0]
            attrs = {
                "dma_kind": ir.StringAttr.get("MVOUT"), "dram_arg": ir.IntegerAttr.get(i64, -1),
                "dram_stride": ir.ArrayAttr.get([ir.IntegerAttr.get(i64, math.prod(shape[k + 1:]))
                                                 for k in range(len(shape))]),
                "operandSegmentSizes": ir.DenseI32ArrayAttr.get([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])}
            if math.prod(shape) != head:
                big = next(d for d, k in enumerate(shape) if k != 1)
                attrs["piece_box"] = ir.ArrayAttr.get([ir.DenseI64ArrayAttr.get([0, head] if d == big else [])
                                                       for d in range(len(shape))])
            ir.Operation.create(TransferOp.OPERATION_NAME, results=[], operands=[
                view, self.cst(0), rec.source, self.cst(0), self.cst(0)], attributes=attrs)
            if tx.masked_axes is not None:
                self.masked_entries(tx, tail, view)
            scf.YieldOp([])

    def gather_of(self, o):
        """The indirect transfer the snapshot `o` sits right before."""
        ops = [x.operation for x in o.block.operations]
        tx = ops[next(k for k, x in enumerate(ops) if x == o.operation) + 1]
        if tx.name != TransferOp.OPERATION_NAME or tx.opview.index != o.opview.source:
            raise RuntimeError(f"an index snapshot does not sit right before its gather: {o}")
        return tx.opview

    def masked_entries(self, tx, tail, view):
        """`view`, a row of `tail` (the tile row-major, or the gather axis alone) whose head the DMA filled
        with the index tile, rewritten last entry first: entry e the index of e's position on the gather
        axis, or 0 where a masked axis of e lies outside its clamp."""
        from mlir.dialects import arith, memref, scf
        ir = self.ir
        axis = None if tx.indirect_dim is None else tx.indirect_dim.value
        clamp = list(tx.clamp)
        whole = axis is None or len(tail) > 1
        masked = [((a if whole else 0), clamp[2 * k], clamp[2 * k + 1])
                  for k, a in enumerate(ir.IntegerAttr(x).value for x in tx.masked_axes)
                  if whole or a == axis]
        n = math.prod(tail)
        dims = list(moved_shape(tx.operation)) if whole else tail
        loop = scf.ForOp(self.cst(0), self.cst(n), self.cst(1))
        with ir.InsertionPoint(loop.body):
            e = arith.SubIOp(self.cst(n - 1), loop.induction_variable).result
            at = [arith.RemUIOp(arith.DivUIOp(e, self.cst(math.prod(dims[d + 1:]))).result,
                                self.cst(dims[d])).result for d in range(len(dims))]
            v = memref.LoadOp(view, [at[axis] if axis is not None and whole else e]).result
            off = None
            for d, low, high in masked:
                out = arith.OrIOp(arith.CmpIOp(arith.CmpIPredicate.slt, at[d], low).result,
                                  arith.CmpIOp(arith.CmpIPredicate.sge, at[d], high).result).result
                off = out if off is None else arith.OrIOp(off, out).result
            if off is not None:
                v = arith.SelectOp(off, arith.ConstantOp(v.type, ir.IntegerAttr.get(v.type, 0)).result, v).result
            memref.StoreOp(v, view, [e])
            scf.YieldOp([])
