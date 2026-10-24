"""pytorchsim-lower-runtime-attributes: each record a call of the wrapper's write with its extents and
values, staged in one buffer the largest record fits; the table and its hash on the module as TABLE, which
the binary step writes into the manifest and the wrapper into the record file's header."""
import hashlib
import json
import math

from ...contract.kernel_object import RUNTIME_ATTRIBUTE_WRITE
from ...dialects.torchsim import TransferOp, moved_shape
from .lib_ir import walk
from .p11a_mark_runtime_attributes import PLUMBING, RECORD, SOURCES, erase_records, records

NAME = "pytorchsim-lower-runtime-attributes"
MARKERS = (RECORD,)

TABLE = "torchsim.runtime_attributes"
"""The module attribute stating the table: {"table_hash", "attributes": [{id, rank, capacity, fill,
elem_bits, sources}]}, as JSON; `sources` the function arguments (by number) a record is read from."""

STAGING = "__torchsim_runtime_attribute_staging"


def run(module, *, timing=False, records_kept=True, **_):
    """Lower every record of `module`; the timed binary (`timing`) and the build the binary step
    judges spills on (not `records_kept`) have none, as if never marked."""
    recs = records(module)
    if not recs:
        return
    if timing or not records_kept:
        erase_records(module)
        return
    table = attribute_table(recs)
    attributes = [table[k] for k in sorted(table)]
    digest = table_hash(attributes)
    _Lowering(module, table, staging_bytes(attributes)).lower(recs)
    import mlir.ir as ir
    module.operation.attributes[TABLE] = ir.StringAttr.get(json.dumps(
        {"table_hash": digest, "attributes": attributes}))


def table_hash(attributes):
    """The u64 a record file's header and the manifest both state for `attributes`."""
    text = json.dumps(attributes, sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def record_bytes(a):
    """The most bytes one record of attribute `a` writes: its extents, then its values -- an S or A
    record its capacity of i64, a G record one trip's row of `elem_bits` entries."""
    rows = a["capacity"][a["fill"]:] if a["fill"] is not None else a["capacity"]
    return 8 * a["rank"] + -(-math.prod(rows) * a["elem_bits"] // 64) * 8


def staging_bytes(attributes):
    """The staging buffer's size: the largest record any attribute writes."""
    return max(record_bytes(a) for a in attributes)


def attribute_table(recs):
    """{id: attribute} of every record id: its rank, capacity, trip dims (`fill`, a G record's),
    element width and the function arguments it is read from."""
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
    return by_id


class _Lowering:
    """The calls one module's records lower to, through the staging global."""

    def __init__(self, module, table, nbytes):
        import mlir.ir as ir
        self.ir = ir
        self.module = module
        self.table = table
        self.i64 = ir.IntegerType.get_signless(64)
        self.i32 = ir.IntegerType.get_signless(32)
        self.idx = ir.IndexType.get()
        self.bytes_type = ir.MemRefType.get([nbytes], ir.IntegerType.get_signless(8))
        self.temps = {}
        with ir.InsertionPoint.at_block_begin(module.body), ir.Location.unknown():
            ir.Operation.create("memref.global", attributes={
                "sym_name": ir.StringAttr.get(STAGING), "sym_visibility": ir.StringAttr.get("private"),
                "type": ir.TypeAttr.get(self.bytes_type), "initial_value": ir.UnitAttr.get(),
                "alignment": ir.IntegerAttr.get(self.i64, 64)})
            ir.Operation.create("func.func", regions=1, attributes={
                "sym_name": ir.StringAttr.get(RUNTIME_ATTRIBUTE_WRITE), "sym_visibility": ir.StringAttr.get("private"),
                "function_type": ir.TypeAttr.get(ir.FunctionType.get([self.i32, self.i64, self.i64], []))})

    def lower(self, recs):
        for o in recs:
            form = self.ir.StringAttr(o.attributes["form"]).value
            slot = self.table[self.ir.IntegerAttr(o.attributes["id"]).value]
            with self.ir.InsertionPoint(o), o.location:
                if form == "S":
                    self.scalar(o, slot)
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
        return memref.GetGlobalOp(self.bytes_type, STAGING).result

    def words(self, n):
        """The staging buffer's head as `n` i64 words."""
        from mlir.dialects import memref
        return memref.ViewOp(self.ir.MemRefType.get([n], self.i64), self.bytes(), self.cst(0), []).result

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

    def write(self, slot, buffer, nbytes):
        """The call writing record `slot` of `nbytes` (an i64 value or a constant) from `buffer`."""
        from mlir.dialects import arith, memref
        at = memref.ExtractAlignedPointerAsIndexOp(buffer).result
        n = self.cst(nbytes, self.i64) if isinstance(nbytes, int) else nbytes
        self.ir.Operation.create("func.call", results=[], operands=[
            self.cst(slot["id"], self.i32), arith.IndexCastOp(self.i64, at).result, n],
            attributes={"callee": self.ir.FlatSymbolRefAttr.get(RUNTIME_ATTRIBUTE_WRITE)})

    def temp(self, o):
        """The i64 stack temporary of `o`'s function, made at its entry."""
        from mlir.dialects import memref
        f = o.parent
        while f.name != "func.func":
            f = f.parent
        key = f.attributes["sym_name"]
        if str(key) not in self.temps:
            entry = f.regions[0].blocks[0]
            with self.ir.InsertionPoint.at_block_begin(entry):
                self.temps[str(key)] = memref.AllocaOp(self.ir.MemRefType.get([1], self.i64), [], []).result
        return self.temps[str(key)]

    def scalar(self, o, slot):
        """The value stored to the stack temporary and written: rank 0, so no extents."""
        t = self.temp(o)
        self.store(t, 0, self.as_i64(o.operands[0]))
        self.write(slot, t, 8)

    def array(self, o, slot):
        """The extents staged, then each element of `source[base + sum(t_d * strides_d)]` below the
        capacity packed row-major by those extents, and the record written: as many values as the
        extents hold, at most the capacity, which the conversion holds the extents to."""
        from mlir.dialects import arith, memref, scf
        ir = self.ir
        rec = o.opview
        strides = list(ir.DenseI64ArrayAttr(o.attributes["strides"]))
        cap = math.prod(slot["capacity"])
        extents = list(rec.extents)
        row = self.words(slot["rank"] + cap)
        for d, e in enumerate(extents):
            self.store(row, d, arith.IndexCastOp(self.i64, e).result)
        values = slot["rank"]

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
        count = self.cst(1)
        for e in extents:
            count = arith.MulIOp(count, e).result
        count = arith.MinSIOp(count, self.cst(cap)).result
        nbytes = arith.MulIOp(arith.AddIOp(count, self.cst(values)).result, self.cst(8)).result
        self.write(slot, row, arith.IndexCastOp(self.i64, nbytes).result)

    def snapshot(self, o, slot):
        """At every trip of the loops around a gather, one record: the extents (the trips', then the
        row's) staged, then the trip's row -- the index tile by one DMA, or, for a masked gather, entry by
        entry with 0 where the clamp masks the element -- and written."""
        from mlir.dialects import arith, memref
        ir = self.ir
        rec = o.opview
        src = ir.MemRefType(rec.source.type)
        shape, bits = list(src.shape), slot["elem_bits"]
        trips, extents = list(rec.indices), list(rec.extents)
        tail = slot["capacity"][len(trips):]
        n = math.prod(tail)
        tx = self.gather_of(o)
        head_words = self.words(slot["rank"])
        for d, e in enumerate(extents):
            self.store(head_words, d, arith.IndexCastOp(self.i64, e).result)
        for k, e in enumerate(tail):
            self.store(head_words, len(extents) + k, self.cst(e, self.i64))
        view = memref.ViewOp(ir.MemRefType.get([n], src.element_type), self.bytes(),
                             self.cst(8 * slot["rank"]), []).result
        i64 = self.i64
        axis = None if tx.indirect_dim is None else tx.indirect_dim.value
        if axis is None:
            box = list(moved_shape(tx.operation))
        else:
            head = tail[axis] if len(tail) > 1 else tail[0]
            big = next((d for d, k in enumerate(shape) if k != 1), len(shape) - 1)
            box = [head if d == big else k for d, k in enumerate(shape)]
        attrs = {
            "dma_kind": ir.StringAttr.get("MVOUT"), "dram_arg": ir.IntegerAttr.get(i64, -1),
            "dram_stride": ir.ArrayAttr.get([ir.IntegerAttr.get(i64, math.prod(box[k + 1:]))
                                             for k in range(len(box))]),
            "operandSegmentSizes": ir.DenseI32ArrayAttr.get([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])}
        if box != shape:
            attrs["piece_box"] = ir.ArrayAttr.get([ir.DenseI64ArrayAttr.get([0, b] if b != k else [])
                                                   for b, k in zip(box, shape)])
        ir.Operation.create(TransferOp.OPERATION_NAME, results=[], operands=[
            view, self.cst(0), rec.source, self.cst(0), self.cst(0)], attributes=attrs)
        if tx.masked_axes is not None:
            self.masked_entries(tx, tail, view)
        self.write(slot, self.bytes(), 8 * slot["rank"] + n * bits // 8)

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
