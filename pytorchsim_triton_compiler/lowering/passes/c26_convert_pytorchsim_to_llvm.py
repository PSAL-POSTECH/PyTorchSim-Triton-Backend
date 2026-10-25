"""convert-pytorchsim-to-llvm: the bank views and strided accesses lowered, the registered MLIR ->
LLVM-dialect passes run in process through the bindings PassManager, then the scratchpad reserved.
Runs after the lower-dma slot's pass; lower.TO_LLVM is the order."""

import os

SPAD_SECTION = ".spad"
SPAD_TOP_SECTION = ".spadtop"
SPAD_TOP_SYMBOL = "spad_bank_top"

SPAD_POISON_ENV = "TORCHSIM_COMPILE_SPAD_POISON"
SPAD_POISON_BYTES_SYMBOL = "torchsim_spad_poison_bytes"
SPAD_POISON_WORD_SYMBOL = "torchsim_spad_poison_word"


class SpadReservationRefused(Exception):
    """No storage can be reserved for a global: a compile error, never a bank slice of the wrong
    size."""


def _refuse_a_scratchpad_life_left(module):
    from .lib_bank_layout import scratchpad_lives_left
    left = scratchpad_lives_left(module)
    if left:
        raise SpadReservationRefused(
            f"a scratchpad buffer is still allocated or freed by an op at LLVM lowering "
            f"({left[0].name}); finalize-memref-to-llvm would make it a malloc or a free on the "
            f"heap, not a slot in {SPAD_SECTION}. pytorchsim-materialize-spad-allocs turns "
            f"each into a global first: {left[0]}")


STANDARD_STEPS = [
    "convert-linalg-to-loops",
    "convert-vector-to-scf{full-unroll=true}",
    "lower-affine",
    "expand-strided-metadata",
    "finalize-memref-to-llvm",
    "func.func(lower-vector-multi-reduction)",
    "convert-vector-to-llvm",
    "convert-arith-to-llvm",
    "convert-math-to-llvm",
    "convert-scf-to-cf",
    "convert-cf-to-llvm",
    "convert-func-to-llvm",
    "convert-arith-to-llvm",
    "convert-index-to-llvm",
    "reconcile-unrealized-casts",
]

STANDARD_PIPELINE = "builtin.module(" + ",".join(STANDARD_STEPS) + ")"

def _flatten(ty):
    """(element count, element type, lanes) of a nested `!llvm.array`: its dims are the memref's, so
    its lanes are lanes_of_shape's."""
    from mlir.dialects.llvm import ArrayType
    from .lib_bank_layout import lanes_of_shape
    shape = []
    while isinstance(ty, ArrayType):
        shape.append(ty.num_elements)
        ty = ty.element_type
    numel = 1
    for d in shape:
        numel *= d
    return numel, ty, lanes_of_shape(shape)


def _zero_tile(count, ety):
    from mlir.ir import (DenseElementsAttr, FloatAttr, FloatType, IntegerAttr,
                         IntegerType, RankedTensorType)
    if isinstance(ety, IntegerType):
        return DenseElementsAttr.get_splat(RankedTensorType.get([count], ety),
                                           IntegerAttr.get(ety, 0))
    if isinstance(ety, FloatType):
        return DenseElementsAttr.get_splat(RankedTensorType.get([count], ety),
                                           FloatAttr.get(ety, 0.0))
    raise SpadReservationRefused(
        f"a scratchpad global of `{ety}` has no zero this pass can write, so it "
        f"would stay an `external` declaration and the link would fail with an "
        f"undefined reference to it. Every tile this compiler banks is an integer or "
        f"a float -- a new element type needs its zero added to _zero_tile.")


def spad_poison():
    """The byte every .spad slot starts as, SPAD_POISON_ENV's; 0, the zero _zero_tile writes, when
    it is unset -- a non-zero one shows a kernel reading what nothing wrote."""
    said = os.environ.get(SPAD_POISON_ENV, "").strip()
    try:
        got = int(said, 0) if said else 0
    except ValueError:
        got = -1
    if not 0 <= got <= 0xFF:
        raise SpadReservationRefused(f"{SPAD_POISON_ENV}={said!r} is not a byte (0 is off)")
    return got


def _poison_tile(count, ety, byte):
    """`count` elements of `ety`, each `byte` in every one of its bytes."""
    import numpy as np
    from mlir.ir import DenseElementsAttr, FloatType, IntegerAttr, IntegerType, RankedTensorType
    width = ety.width
    bits = int.from_bytes(bytes([byte]) * -(-width // 8), "little") & ((1 << width) - 1)
    if isinstance(ety, IntegerType):
        signed = bits - (1 << width) if bits >> (width - 1) else bits
        return DenseElementsAttr.get_splat(RankedTensorType.get([count], ety),
                                           IntegerAttr.get(ety, signed))
    if isinstance(ety, FloatType) and width in (8, 16, 32, 64):
        word = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}[width]
        return DenseElementsAttr.get(np.full([1], bits, dtype=word), type=ety, shape=[count])
    return _zero_tile(count, ety)


def _mint_the_bank_top_anchor(top):
    """SPAD_TOP_SYMBOL in .spadtop, the op made."""
    from mlir.ir import (Attribute, DenseElementsAttr, InsertionPoint, IntegerAttr,
                         IntegerType, Operation, RankedTensorType, StringAttr, Type,
                         TypeAttr)
    i32 = IntegerType.get_signless(32)
    with InsertionPoint.at_block_begin(top):
        return Operation.create(
            "llvm.mlir.global", results=[], operands=[], regions=1,
            attributes={
                "sym_name": StringAttr.get(SPAD_TOP_SYMBOL),
                "global_type": TypeAttr.get(Type.parse("!llvm.array<4 x i32>")),
                "linkage": Attribute.parse("#llvm.linkage<external>"),
                "addr_space": IntegerAttr.get(i32, 0),
                "section": StringAttr.get(SPAD_TOP_SECTION),
                "value": DenseElementsAttr.get_splat(
                    RankedTensorType.get([4], i32), IntegerAttr.get(i32, 0))})


def reserve_spad_storage(module, vectorlane, vlen_bits, poison=0):
    """Every scratchpad global as a dso_local DEFINITION of ONE lane's share in `.spad` (llc reaches
    it PC-relatively), starting on a vector-register boundary (a whole-register load never straddles
    two), plus the bank-top anchor; every slot `poison` in each byte when it is not 0."""
    from mlir.dialects.llvm import ArrayType
    from mlir.ir import IntegerAttr, IntegerType, StringAttr, TypeAttr, UnitAttr
    from .lib_bank_layout import SPAD_SPACE, need_per_lane
    top = module.operation.regions[0].blocks[0]
    reserved = []
    for op in [o.operation for o in top.operations
               if o.operation.name == "llvm.mlir.global"
               and IntegerAttr(o.operation.attributes["addr_space"]).value == SPAD_SPACE]:
        name = StringAttr(op.attributes["sym_name"]).value
        numel, elem, lanes = _flatten(TypeAttr(op.attributes["global_type"]).value)
        count = need_per_lane(numel, lanes, vectorlane)
        value = _poison_tile(count, elem, poison) if poison else _zero_tile(count, elem)
        op.attributes["global_type"] = TypeAttr.get(ArrayType.get(elem, count))
        op.attributes["value"] = value
        op.attributes["section"] = StringAttr.get(SPAD_SECTION)
        op.attributes["dso_local"] = UnitAttr.get()
        op.attributes["alignment"] = IntegerAttr.get(IntegerType.get_signless(64), vlen_bits // 8)
        reserved.append((name, count, elem))
    if reserved:
        anchor = _mint_the_bank_top_anchor(top)
        if poison:
            _mint_the_poisoned_extent(top, reserved, vlen_bits // 8, anchor, poison)
    return reserved


def _mint_the_poisoned_extent(top, reserved, align, anchor, byte=0xFF):
    """SPAD_POISON_BYTES_SYMBOL (how many bytes from a bank's start the slots reach, each aligned)
    and SPAD_POISON_WORD_SYMBOL (`byte` in each of 8), i64s at `anchor`'s linkage: the wrapper's one
    statement that this module is poisoned, and what it refills every lane with before each program."""
    from mlir.ir import (InsertionPoint, IntegerAttr, IntegerType, Operation, StringAttr,
                         TypeAttr, UnitAttr)
    from .lib_ir import _i64_signed, elem_bytes
    i64 = IntegerType.get_signless(64)
    reach = sum(-(-count * elem_bytes(elem) // align) * align for _, count, elem in reserved)
    word = int.from_bytes(bytes([byte]) * 8, "little")
    with InsertionPoint.at_block_begin(top):
        for name, value in ((SPAD_POISON_BYTES_SYMBOL, reach),
                            (SPAD_POISON_WORD_SYMBOL, _i64_signed(word))):
            Operation.create(
                "llvm.mlir.global", results=[], operands=[], regions=1,
                attributes={"sym_name": StringAttr.get(name),
                            "global_type": TypeAttr.get(i64),
                            "linkage": anchor.attributes["linkage"],
                            "constant": UnitAttr.get(),
                            "value": IntegerAttr.get(i64, value)})


def poisoned(lowered):
    """Whether the LLVM-dialect module in file `lowered` defines SPAD_POISON_BYTES_SYMBOL: the one
    reading of whether its .spad is poisoned, by its symbol table, whatever the environment says now."""
    from mlir.ir import Context, Location, Module, SymbolTable
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown(), open(lowered) as f:
        return SPAD_POISON_BYTES_SYMBOL in SymbolTable(Module.parse(f.read()).operation)


def lower_strided_accesses(module):
    """Every pytorchsim.strided_load/store -> llvm.intr.experimental.vp.strided.load/store at the
    first element's byte address, `stride` elements apart, reading no more than `count`; the
    pointer is in SPAD_SPACE when the memref is, so a scratchpad access is a space-1 one."""
    from mlir.dialects import arith, llvm, memref
    from mlir.ir import (DenseElementsAttr, IndexType, InsertionPoint, IntegerAttr, IntegerType,
                         Operation, ShapedType, VectorType)
    from .lib_bank_layout import SPAD_SPACE, in_scratchpad
    from .lib_ir import elem_bytes, strides_and_offset, walk
    idx, i1, i32, i64 = (IndexType.get(), IntegerType.get_signless(1),
                         IntegerType.get_signless(32), IntegerType.get_signless(64))
    for op in [o for o in walk(module.operation)
               if o.name in ("pytorchsim.strided_load", "pytorchsim.strided_store")]:
        load = op.name == "pytorchsim.strided_load"
        vec = op.results[0] if load else op.operands[0]
        mem = op.operands[0 if load else 1]
        indices = list(op.operands)[1 if load else 2:]
        n = VectorType(vec.type).shape[0]
        esize = elem_bytes(VectorType(vec.type).element_type)
        so = strides_and_offset(mem)
        if so is None or ShapedType.get_dynamic_stride_or_offset() in list(so[0]) + [so[1]]:
            raise NotImplementedError(f"a strided access through a memref with no static layout: {op}")
        with InsertionPoint(op):
            def cst(ty, v):
                return arith.ConstantOp(ty, IntegerAttr.get(ty, v)).result
            off = cst(idx, so[1])
            for i, st in zip(indices, so[0]):
                off = arith.AddIOp(off, arith.MulIOp(i, cst(idx, st)).result).result
            addr = arith.AddIOp(memref.ExtractAlignedPointerAsIndexOp(mem).result,
                                arith.MulIOp(off, cst(idx, esize)).result).result
            space = SPAD_SPACE if in_scratchpad(mem.type) else 0
            ptr = llvm.IntToPtrOp(llvm.PointerType.get(space),
                                  arith.IndexCastOp(i64, addr).result).result
            mask = arith.ConstantOp(VectorType.get([n], i1), DenseElementsAttr.get_splat(
                VectorType.get([n], i1), IntegerAttr.get(i1, 1))).result
            tail = [cst(i64, IntegerAttr(op.attributes["stride"]).value * esize), mask,
                    cst(i32, min(IntegerAttr(op.attributes["count"]).value, n))]
            if load:
                got = Operation.create("llvm.intr.experimental.vp.strided.load",
                                       results=[vec.type], operands=[ptr] + tail).results[0]
                op.results[0].replace_all_uses_with(got)
            else:
                Operation.create("llvm.intr.experimental.vp.strided.store", results=[],
                                 operands=[vec, ptr] + tail)
        op.operation.erase()


def promote_short_loops(module, vlen):
    """Every loop of constant bounds that runs once, or whose trips each step at least one register
    of its widest element and together no more than one register group at the ISA's widest LMUL (a
    one-trip loop the spill retry narrowed), becomes its body once per trip in place: the stores
    one op leaves and the loads of the next meet in one block, and each trip's values stay registers."""
    from mlir.dialects import arith
    from mlir.ir import IndexType, InsertionPoint, IntegerAttr
    from ...machine import ISA_MAX_LMUL
    from .lib_ir import clone_ops, const_index, loop_bounds, trip_count, walk
    for loop in reversed([o for o in walk(module.operation)
                          if o.name in ("scf.for", "affine.for") and trip_count(o)]):
        trips = trip_count(loop)
        first, _, step = loop_bounds(loop)
        widest = _widest_bits(loop, step) if trips != 1 else 0
        if trips != 1 and not vlen <= step * widest <= trips * step * widest <= vlen * ISA_MAX_LMUL:
            continue
        body = loop.regions[0].blocks[0]
        ops = [o.operation for o in body.operations]
        iv, carried = body.arguments[0], list(loop.operands)[len(loop.operands) - len(loop.results):]
        for k in range(trips):
            with InsertionPoint(loop), loop.location:
                at = first + k * step
                mapping = {iv: const_index(at) if isinstance(iv.type, IndexType)
                           else arith.ConstantOp(iv.type, IntegerAttr.get(iv.type, at)).result}
            mapping.update(zip(list(body.arguments)[1:], carried))
            clone_ops(ops[:-1], mapping, InsertionPoint(loop))
            carried = [mapping.get(v, v) for v in ops[-1].operands]
        for r, v in zip(loop.results, carried):
            r.replace_all_uses_with(v)
        loop.erase()


def _widest_bits(loop, step):
    """The widest element of the vectors of `step` elements a trip of `loop` makes or reads; 0 for a
    loop that steps no such vector, which only one trip promotes."""
    from mlir.ir import VectorType
    from .lib_ir import elem_bits
    widest = 0
    for o in loop.regions[0].blocks[0].operations:
        for v in list(o.results) + list(o.operands):
            try:
                vt = VectorType(v.type)
            except (TypeError, ValueError):
                continue
            if vt.shape == [step]:
                widest = max(widest, elem_bits(vt.element_type))
    return widest


_SPAD_READS = {"affine.vector_load": 0, "vector.transfer_read": 0}
_SPAD_WRITES = {"affine.vector_store": 1, "vector.transfer_write": 1}


def _spad_global(v):
    """(the scratchpad global `v` views, the view's (strides, offset) in elements or None when not
    static); None when `v` is no get_global or a reinterpret_cast of one."""
    from mlir.ir import FlatSymbolRefAttr, MemRefType, ShapedType
    from .lib_ir import producer
    p = producer(v)
    if p is not None and p.name == "memref.reinterpret_cast":
        q = producer(p.operands[0])
        if q is None or q.name != "memref.get_global":
            return None
    elif p is None or p.name != "memref.get_global":
        return None
    root = p if p.name == "memref.get_global" else producer(p.operands[0])
    strides, offset = MemRefType(v.type).get_strides_and_offset()
    dyn = ShapedType.get_dynamic_stride_or_offset()
    known = None if dyn in strides or offset == dyn else (strides, offset)
    return FlatSymbolRefAttr(root.attributes["name"]).value, known


def _spad_access(op):
    """(the global, the place `(first byte, end byte, vector type)` or None when it is not known)
    of a whole-vector read or write of a scratchpad global; None for any other op."""
    import mlir.ir as ir
    from .lib_ir import const_int, elem_bits, eval_affine
    at = _SPAD_READS.get(op.name, _SPAD_WRITES.get(op.name))
    if at is None:
        return None
    found = _spad_global(op.operands[at])
    if found is None:
        return None
    name, layout = found
    vec = ir.VectorType(op.results[0].type if op.name in _SPAD_READS else op.operands[0].type)
    rank = ir.MemRefType(op.operands[at].type).rank
    rest = list(op.operands)[at + 1:]
    if op.name.startswith("affine."):
        m = ir.AffineMapAttr(op.attributes["map"]).value
        vals = [const_int(v) for v in rest]
        idx = None if None in vals else [eval_affine(e, vals[:m.n_dims], vals[m.n_dims:])
                                         for e in m.results]
    else:
        pad = 1 if op.name == "vector.transfer_read" else 0
        perm = ir.AffineMapAttr(op.attributes["permutation_map"]).value
        whole = (len(rest) == rank + pad
                 and perm == ir.AffineMap.get_minor_identity(perm.n_dims, len(perm.results))
                 and all(ir.BoolAttr(b).value for b in ir.ArrayAttr(op.attributes["in_bounds"])))
        idx = [const_int(v) for v in rest[:rank]] if whole else None
    bits = elem_bits(vec.element_type)
    if layout is None or idx is None or None in idx or vec.rank != 1 or bits % 8:
        return name, None
    strides, offset = layout
    first = (offset + sum(i * k for i, k in zip(idx, strides))) * bits // 8
    return name, (first, first + vec.shape[0] * bits // 8, str(vec))


def _escaped_globals(module, access):
    """The scratchpad globals some op other than a whole-vector access (`access`, by op) reaches (a
    DMA's pointer, a view this pass does not read): what they hold is never assumed."""
    from .lib_ir import walk
    out = set()
    for o in walk(module.operation):
        if o.name != "memref.get_global":
            continue
        name = _spad_global(o.results[0])[0]
        todo = [o.results[0]]
        while todo:
            v = todo.pop()
            for use in v.uses:
                u = use.owner.operation if hasattr(use.owner, "operation") else use.owner
                if u.name == "memref.reinterpret_cast" and _spad_global(u.results[0]):
                    todo.append(u.results[0])
                elif u not in access or use.operand_number != (
                        _SPAD_READS.get(u.name, _SPAD_WRITES.get(u.name))):
                    out.add(name)
    return out


def forward_spad_values(module):
    """A read of a scratchpad place written or read before it in the same block, with nothing
    between that may touch that global, takes that value; a write a later one of the same place
    replaces unread, and every write of a global nothing reads, go. Escaped globals stay as they are."""
    from .lib_ir import walk
    access = {}
    for o in walk(module.operation):
        a = _spad_access(o)
        if a is not None:
            access[o] = a
    escaped = _escaped_globals(module, access)

    def scan(block):
        """Forwards in `block`; returns the globals its ops, nested ones too, touch."""
        known, pending, seen = {}, {}, set()
        for o in list(block.operations):
            o = o.operation
            inner = set()
            for region in o.regions:
                for b in region.blocks:
                    inner |= scan(b)
            acc = access.get(o)
            if acc is None:
                for g in inner:
                    known.pop(g, None)
                    pending.pop(g, None)
                seen |= inner
                continue
            g, place = acc
            seen.add(g)
            if g in escaped:
                continue
            if o.name in _SPAD_READS:
                if place is None:
                    pending.pop(g, None)
                    continue
                if g in pending:
                    pending[g].take_overlapping(place)
                hit = known[g].get(place) if g in known else None
                if hit is not None:
                    o.results[0].replace_all_uses_with(hit)
                    o.erase()
                    continue
                known.setdefault(g, _Places()).put(place, o.results[0])
                continue
            if place is None:
                known.pop(g, None)
                pending.pop(g, None)
                continue
            waiting = pending.setdefault(g, _Places())
            for q, w in waiting.take_overlapping(place):
                if q == place:
                    w.erase()
            waiting.put(place, o)
            if g in known:
                known[g].take_overlapping(place)
            known.setdefault(g, _Places()).put(place, o.operands[0])
        return seen

    for f in [o for o in walk(module.operation) if o.name == "func.func"]:
        for b in f.regions[0].blocks:
            scan(b)
    live = [(o, a) for o in walk(module.operation) if o.name in _SPAD_READS or o.name in _SPAD_WRITES
            for a in [_spad_access(o)] if a is not None]
    read = {a[0] for o, a in live if o.name in _SPAD_READS}
    for o, a in live:
        if o.name in _SPAD_WRITES and a[0] not in read | escaped:
            o.erase()


class _Places:
    """What a block knows of one global, by place `(first byte, end byte, type)`: found exactly, or
    every place a byte range overlaps, by the places' first bytes kept in order."""

    def __init__(self):
        self.at, self.firsts, self.longest = {}, [], 0

    def get(self, place):
        return self.at.get(place)

    def put(self, place, value):
        import bisect
        if place not in self.at:
            bisect.insort(self.firsts, (place[0], place))
            self.longest = max(self.longest, place[1] - place[0])
        self.at[place] = value

    def take_overlapping(self, place):
        """Every place overlapping `place`, removed, with its value."""
        import bisect
        lo = bisect.bisect_left(self.firsts, (place[0] - self.longest + 1,))
        hi = bisect.bisect_left(self.firsts, (place[1],))
        got = [(q, self.at[q]) for _, q in self.firsts[lo:hi] if _overlaps(q, place)]
        if got:
            drop = {q for q, _ in got}
            self.firsts = [x for x in self.firsts if x[1] not in drop]
            for q in drop:
                del self.at[q]
        return got


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def lower_steps(module, vlen):
    """vector.step as the core's vid.v: lowered to a dense constant, a step that a promoted loop's
    constant offsets or an xor turn into a non-affine one would be read from a constant pool in DRAM,
    which no vector load of this machine reaches. vid.v is at the width its index casts all take. A
    length no register group holds whole (not a power of two, or wider than one group at the ISA's
    widest LMUL) is left a constant, which LLVM splits or builds itself."""
    import mlir.ir as ir
    from mlir.dialects import arith, llvm
    from ...machine import ISA_MAX_LMUL
    from .lib_ir import walk
    i64 = ir.IntegerType.get_signless(64)
    for op in [o for o in walk(module.operation) if o.name == "vector.step"]:
        vt = ir.VectorType(op.results[0].type)
        n = vt.shape[0]
        if n & (n - 1):
            continue
        casts = [u.owner for u in op.results[0].uses]
        kinds = {str(ir.VectorType(c.results[0].type).element_type) for c in casts
                 if c.name in ("arith.index_cast", "arith.index_castui")}
        narrow = len(kinds) == 1 and len(casts) == sum(
            c.name in ("arith.index_cast", "arith.index_castui") for c in casts)
        ity = ir.VectorType(casts[0].results[0].type).element_type if narrow else (
            vt.element_type if isinstance(vt.element_type, ir.IntegerType) else i64)
        width = ir.IntegerType(ity).width
        if n * width > vlen * ISA_MAX_LMUL:
            continue
        groups = max(1, -(-n * width // vlen))
        with ir.InsertionPoint(op), op.location:
            sc = ir.VectorType.get([groups * 64 // width], ity, scalable=[True])
            got = llvm.call_intrinsic(sc, "llvm.riscv.vid",
                                      [llvm.PoisonOp(sc).result, arith.ConstantOp(i64, n).result], [], [])
            got = ir.Operation.create("llvm.intr.vector.extract", results=[ir.VectorType.get([n], ity)],
                                      operands=[got],
                                      attributes={"pos": ir.IntegerAttr.get(i64, 0)}).results[0]
            if narrow:
                for c in casts:
                    c.results[0].replace_all_uses_with(got)
                    c.erase()
            else:
                if got.type != vt:
                    got = arith.IndexCastOp(vt, got).result
                op.results[0].replace_all_uses_with(got)
        op.erase()


LOOPS_PROMOTED = "pytorchsim.loops_promoted"
"""The module attribute run() sets when it promotes the one-trip loops: the retry's widths stood."""


def run(module, *, vectorlane, target, dump=None, timing=False, promote=True, **_):
    """Lower `module` to the LLVM dialect in place and return the scratchpad reservations;
    `dump(name, module)`, when given, runs after each standard step. SPAD_POISON_ENV is read here
    alone, and never for the timed binary (timing=True), whose data no one reads."""
    from mlir.ir import UnitAttr
    from mlir.passmanager import PassManager
    from .lib_bank_layout import lower_bank_views
    from .lib_pack import lower_packs
    ctx = module.context
    _refuse_a_scratchpad_life_left(module)
    lower_bank_views(module)
    lower_packs(module)
    lower_strided_accesses(module)
    lower_steps(module, target.vlen_bits)
    if promote:
        promote_short_loops(module, target.vlen_bits)
        forward_spad_values(module)
        module.operation.attributes[LOOPS_PROMOTED] = UnitAttr.get(ctx)
    if dump is None:
        PassManager.parse(STANDARD_PIPELINE, ctx).run(module.operation)
    else:
        for step in STANDARD_STEPS:
            PassManager.parse(f"builtin.module({step})", ctx).run(module.operation)
            dump(step.split("{")[0].split("(")[-1].rstrip(")"), module)
    reserved = reserve_spad_storage(module, vectorlane, target.vlen_bits,
                                    0 if timing else spad_poison())
    if dump is not None:
        dump("torchsim-reserve-spad-storage", module)
    return reserved
