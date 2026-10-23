"""tritonpytorchsim-lower-runtime-attributes: each record a store into the program's row of the hidden
output, the external global ROW the wrapper hands every program, and each read point its value; the
slot table goes on the module as TABLE, which the binary step writes into the manifest."""
import json
import math

from ...contract.kernel_object import RUNTIME_ATTRIBUTE_HEAD, RUNTIME_ATTRIBUTE_ROW
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
    count and fill words, its extents, then the values its capacity holds."""
    import mlir.ir as ir
    by_id = {}
    for o in recs:
        i = ir.IntegerAttr(o.attributes["id"]).value
        form = ir.StringAttr(o.attributes["form"]).value
        got = by_id.setdefault(i, {"id": i, "rank": ir.IntegerAttr(o.attributes["rank"]).value,
                                   "capacity": [], "fill": None, "sources": []})
        if SOURCES not in o.attributes:
            raise RuntimeError(f"runtime attribute {i} has a record that states no {SOURCES}")
        got["sources"] = sorted(set(got["sources"]) | set(ir.DenseI64ArrayAttr(o.attributes[SOURCES])))
        if form in ("A", "G"):
            got["capacity"] = list(ir.DenseI64ArrayAttr(o.attributes["capacity"]))
        if form == "G":
            got["fill"] = len(list(o.opview.extents))
    if sorted(by_id) != list(range(len(by_id))):
        raise RuntimeError(f"runtime attribute ids {sorted(by_id)} are not 0..{len(by_id) - 1}")
    offset = RUNTIME_ATTRIBUTE_HEAD
    for i in sorted(by_id):
        by_id[i]["offset"] = offset
        offset += 2 + by_id[i]["rank"] + math.prod(by_id[i]["capacity"])
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
        self.row_type = ir.MemRefType.get([row_words], self.i64)
        with ir.InsertionPoint.at_block_begin(module.body), ir.Location.unknown():
            ir.Operation.create("memref.global", attributes={
                "sym_name": ir.StringAttr.get(RUNTIME_ATTRIBUTE_ROW),
                "type": ir.TypeAttr.get(self.row_type)})

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
                    raise NotImplementedError(f"an index snapshot reaches the ELF lowering: {o}")
            if len(o.results):
                o.results[0].replace_all_uses_with(o.operands[0])
            o.erase()
        for o in reversed([o for o in walk(self.module.operation) if PLUMBING in o.attributes]):
            if all(not list(r.uses) for r in o.results):
                o.erase()

    def row(self):
        from mlir.dialects import memref
        return memref.GetGlobalOp(self.row_type, RUNTIME_ATTRIBUTE_ROW).result

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
