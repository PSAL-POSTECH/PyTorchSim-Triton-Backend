"""lower_dma: every pytorchsim.transfer -> the model's DMA, CONFIG_DESC +
MVIN/MVOUT as custom-1 inline asm over its 20-slot descriptor; fit-transfers-to-descriptor leaves an uncut
transfer at rank <= DESCRIPTOR_DIMS, and a cut one is masked boxes here."""

from collections import namedtuple

from ....dialects.pytorchsim import POINTER_TILE_BASE, TransferOp, WaitOp, walked_shape
from ...passes.lib_ir import elem_bytes, strides_and_offset, walk_ops, _i64_signed
from ...passes.lib_bank_layout import is_uniform, row_major
from ...passes.lib_descriptor import TransferFacts

NAME = "pytorchsim-lower-dma"

SLOT = "lower-dma"

DESCRIPTOR_DIMS = 4

CONSTRAINTS = "r,r,~{memory},~{dirflag},~{fpsr},~{flags}"

CONFIG_DESC = 7
INDEX_KEY = 8
MVIN, MVOUT = 2, 3
FUNC7 = {"MVIN": MVIN, "MVOUT": MVOUT}
DESC_SLOTS = 20

S_FLAGS = 14
S_INDIRECT_ADDR = 15
S_INDIRECT = 16
S_FILL = 17
S_DRAM_BASE = 18
W_DIM_LOW, W_DIM_HIGH = 4, 8
W_MM_STRIDE = 12
W_INDIRECT_ADDR = 2 * S_INDIRECT_ADDR
W_INDIRECT = 2 * S_INDIRECT
W_FILL = 2 * S_FILL
W_DRAM_BASE = 2 * S_DRAM_BASE
F_INDIRECT, F_MASKED = 1, 2
F_SKIP_SHIFT = 4
CONFIG_TYPE = {MVIN: 0, MVOUT: 3}
VLANE_STRIDE = 1

_Desc4 = namedtuple("_Desc4", "expand shape dram spad vsa lo hi skip")

_Indirect = namedtuple("_Indirect", "addr stride esize dim lanes")

def _erase_globals_nothing_reads(module):
    from mlir.ir import FlatSymbolRefAttr, StringAttr
    top = module.operation.regions[0].blocks[0]
    for g in [o for o in walk_ops(top)
              if o.operation.name == "memref.get_global"
              and not list(o.operation.results[0].uses)]:
        g.operation.erase()
    read = {FlatSymbolRefAttr(o.operation.attributes["name"]).value for o in walk_ops(top)
            if o.operation.name == "memref.get_global"}
    for g in [o for o in walk_ops(top)
              if o.operation.name == "memref.global"
              and StringAttr(o.operation.attributes["sym_name"]).value not in read]:
        g.operation.erase()


def run(module, timing=False, vectorlane=None, **_):
    """Lower every pytorchsim.transfer in `module`."""
    if vectorlane is None:
        raise ValueError(
            f"{NAME} needs the lane count: a gather's descriptor names how "
            "many lanes hold its index. Pass vectorlane=machine.VECTORLANE_SIZE.")
    for w in [o for o in walk_ops(module.operation.regions[0].blocks[0])
              if o.operation.name == WaitOp.OPERATION_NAME]:
        w.operation.erase()
    descs = _Descriptors(module.operation.regions[0].blocks[0])
    for op in _find_transfers(module):
        tx = _Tx(op, descs, vectorlane)
        if tx.cut:
            raise RuntimeError(
                f"a cut transfer reached lower-dma: pytorchsim-fit-transfers-to-descriptor splits "
                f"every cut into its pieces: {op}")
        if len(tx.tile_shape) > DESCRIPTOR_DIMS:
            _unfitted(tx)
        if not timing:
            _emit_direct(tx)
        op.erase()
    _erase_globals_nothing_reads(module)


class _Descriptors:

    def __init__(self, module_top):
        from mlir.ir import IndexType, IntegerType, MemRefType
        self.module_top = module_top
        self.by_slots = {}
        self.n = 0
        self.i64 = IntegerType.get_signless(64)
        self.i32 = IntegerType.get_signless(32)
        self.idx_ty = IndexType.get()
        self.desc_ty = MemRefType.get([DESC_SLOTS], self.i64)
        self.i32_view_ty = MemRefType.get([DESC_SLOTS * 2], self.i32)

    def global_for(self, slots):
        from mlir.ir import (DenseElementsAttr, InsertionPoint, Operation,
                             RankedTensorType, StringAttr, TypeAttr)
        key = tuple(slots)
        if key not in self.by_slots:
            name = f"dma_desc_{self.n}"
            self.n += 1
            import numpy as np
            init = DenseElementsAttr.get(
                np.array([_i64_signed(v) for v in slots], dtype=np.int64),
                type=RankedTensorType.get([DESC_SLOTS], self.i64))
            with InsertionPoint.at_block_begin(self.module_top):
                Operation.create("memref.global", results=[], operands=[], attributes={
                    "sym_name": StringAttr.get(name),
                    "sym_visibility": StringAttr.get("private"),
                    "type": TypeAttr.get(self.desc_ty),
                    "initial_value": init})
            self.by_slots[key] = name
        return self.by_slots[key]

    def _byte_ptr(self, g):
        from mlir.dialects import arith, memref
        return arith.IndexCastOp(
            self.i64,
            memref.ExtractAlignedPointerAsIndexOp(g).result).result

    def _mint_i32_view(self, slots, masked_pairs, fill):
        from mlir.dialects import memref
        from mlir.ir import (DenseElementsAttr, InsertionPoint, Operation,
                             RankedTensorType, StringAttr, TypeAttr)
        import numpy as np
        slots = list(slots)
        if masked_pairs:
            slots[S_FLAGS] |= F_MASKED << 48
            slots[S_FILL] = fill & 0xFFFFFFFFFFFFFFFF
        words = []
        for v in slots:
            v &= 0xFFFFFFFFFFFFFFFF
            words.append(_i32_signed(v & 0xFFFFFFFF))
            words.append(_i32_signed((v >> 32) & 0xFFFFFFFF))
        name = f"dma_desc_{self.n}"
        self.n += 1
        init = DenseElementsAttr.get(
            np.array(words, dtype=np.int32),
            type=RankedTensorType.get([DESC_SLOTS * 2], self.i32))
        with InsertionPoint.at_block_begin(self.module_top):
            Operation.create("memref.global", results=[], operands=[],
                             attributes={
                                 "sym_name": StringAttr.get(name),
                                 "sym_visibility": StringAttr.get("private"),
                                 "type": TypeAttr.get(self.i32_view_ty),
                                 "initial_value": init})
        return memref.GetGlobalOp(self.i32_view_ty, name).result

    def _store_runtime_fields(self, g, masked_pairs, ind_addr_i64,
                              dram_base_i64, fill_i64, stride_pairs,
                              ostride_i64, ind_esize):
        from mlir.dialects import arith, memref
        from mlir.ir import IntegerAttr

        def store_word(v32, i32_idx):
            memref.StoreOp(v32, g, [arith.ConstantOp(
                self.idx_ty, IntegerAttr.get(self.idx_ty, i32_idx)).result])

        def store_i64(v64, word):
            """A 64-bit field is two i32 stores, low half first."""
            store_word(arith.TruncIOp(self.i32, v64).result, word)
            shamt = arith.ConstantOp(
                self.i64, IntegerAttr.get(self.i64, 32)).result
            store_word(
                arith.TruncIOp(self.i32, arith.ShRUIOp(v64, shamt).result).result,
                word + 1)

        for axis4, low_val, high_val in masked_pairs:
            store_word(arith.IndexCastOp(self.i32, low_val).result,
                       W_DIM_LOW + axis4)
            store_word(arith.IndexCastOp(self.i32, high_val).result,
                       W_DIM_HIGH + axis4)
        if ind_addr_i64 is not None:
            store_i64(ind_addr_i64, W_INDIRECT_ADDR)
        if dram_base_i64 is not None:
            store_i64(dram_base_i64, W_DRAM_BASE)
        if fill_i64 is not None:
            store_i64(fill_i64, W_FILL)
        for axis4, val in stride_pairs:
            store_i64(val, W_MM_STRIDE + 2 * axis4)
        if ostride_i64 is not None:
            lo = arith.AndIOp(
                arith.TruncIOp(self.i32, ostride_i64).result,
                arith.ConstantOp(self.i32,
                                 IntegerAttr.get(self.i32, 0xFFFF)).result)
            hi = arith.ConstantOp(
                self.i32,
                IntegerAttr.get(self.i32, (ind_esize & 0xFFFF) << 16))
            store_word(arith.OrIOp(lo.result, hi.result).result, W_INDIRECT)

    def ptr(self, slots, masked_pairs=(), fill=0, ind_addr_i64=None,
            fill_i64=None, stride_pairs=(), ostride_i64=None, ind_esize=0,
            dram_base_i64=None):
        """Byte pointer to the DMA descriptor global: a DEDUPED i64 one when every field is static,
        a UNIQUE i32-view one (same bytes) otherwise, so the runtime fields can be stored a word at
        a time."""
        from mlir.dialects import memref
        if not masked_pairs and ind_addr_i64 is None and fill_i64 is None \
                and not stride_pairs and ostride_i64 is None \
                and dram_base_i64 is None:
            return self._byte_ptr(memref.GetGlobalOp(
                self.desc_ty, self.global_for(slots)).result)
        g = self._mint_i32_view(slots, masked_pairs, fill)
        self._store_runtime_fields(g, masked_pairs, ind_addr_i64,
                                   dram_base_i64, fill_i64, stride_pairs,
                                   ostride_i64, ind_esize)
        return self._byte_ptr(g)


class _Tx(TransferFacts):

    def __init__(self, op, descs, vectorlane):
        from mlir.ir import IndexType, IntegerType
        kind = op.opview.dma_kind.value
        self.dma_type = FUNC7.get(kind)
        if self.dma_type is None:
            raise NotImplementedError(
                f"transfer carries dma_kind={kind!r}, which has no instruction "
                f"here; FUNC7 is the whole set this pass can emit")
        super().__init__(op)
        self.descs = descs
        self.vectorlane = vectorlane
        self.i64 = IntegerType.get_signless(64)
        self.idx_ty = IndexType.get()
        self.box = self.piece_box
        self.skip_axes = tuple(sorted(self.box)) if self.box else ()
        boxed = set(self.box or ())
        self.fill_only = any(d in boxed for d, _lo, _hi in self.masked_pairs)
        self.masked_pairs = [p for p in self.masked_pairs if p[0] not in boxed]

    def const(self, v):
        from mlir.ir import IntegerAttr
        from mlir.dialects import arith
        return arith.ConstantOp(self.idx_ty, IntegerAttr.get(self.idx_ty,
                                                             v)).result

    def _four_dims(self, cfg_shape, dram_strides, spad_strides, box, skip_axes,
                   subtile_shape):
        expand = DESCRIPTOR_DIMS - len(cfg_shape)
        shape4 = [1] * expand + list(cfg_shape)
        dram4 = [0] * expand + list(dram_strides)
        spad4 = [0] * expand + list(spad_strides)
        vsa = _lane_dim(self, shape4, dram4, spad4, expand)
        lo4 = hi4 = None
        if box:
            if subtile_shape is not None:
                raise NotImplementedError(
                    f"a wrapped tile that is also sub-tiled: the descriptor "
                    f"describes the subtile and the box names tile rows: "
                    f"{self.op}")
            lo4, hi4 = [0] * DESCRIPTOR_DIMS, list(shape4)
            for d, (lo, hi) in box.items():
                lo4[expand + d], hi4[expand + d] = lo, hi
            if self.fill_only:
                empty = _empty_dim(skip_axes, self.masked_pairs, expand, self.op)
                lo4[empty] = hi4[empty] = 0
        return _Desc4(expand, shape4, dram4, spad4, vsa,
                      lo4, hi4, sum(1 << (expand + d) for d in skip_axes))

    def _indirect_fields(self, gather_axis, subtile_shape, expand):
        from mlir.ir import IntegerAttr, MemRefType
        from mlir.dialects import arith, memref
        if self.gather_axis is not None and subtile_shape is not None:
            raise NotImplementedError(
                f"the index buffer holds one entry per position on axis "
                f"{self.gather_axis}, and this descriptor cannot address it "
                f"that way -- the axis is sub-tiled, so the "
                f"coordinate the DMA reads restarts: {self.op}")
        if not self.indirect:
            return _Indirect(None, 0, 0, 0, 0)
        off_ty = MemRefType(self.offset_buf.type)
        addr = arith.IndexCastOp(
            self.i64,
            memref.ExtractAlignedPointerAsIndexOp(self.offset_buf).result).result
        esize = elem_bytes(off_ty.element_type)
        stride = IntegerAttr(self.op.attributes["offset_stride"]).value
        dim = 0 if gather_axis is None else expand + gather_axis + 1
        if not dim:
            return _Indirect(addr, stride, esize, 0, 0)
        lanes = _index_lanes(self)
        if not 0 < lanes <= 0xFFFF:
            raise NotImplementedError(
                f"the index buffer is held by {lanes} lanes, which the "
                f"descriptor's uint16 field cannot name: {self.op}")
        return _Indirect(addr, stride, esize, dim, lanes)

    def _dram_bound_fields(self):
        from mlir.dialects import arith, memref
        if self.pointer_tile:
            return _emit_i64_const(0), 1 << 63
        return (arith.IndexCastOp(
            self.i64,
            memref.ExtractAlignedPointerAsIndexOp(self.dram).result).result,
            self.dram_ty.shape[0] * elem_bytes(self.dram_ty.element_type))

    def asm(self, d4, sram_mem, sram_indices, dram_idx_val, subtile_shape,
            desc_masked_pairs=None, desc_stride_pairs=None, gather_axis=None,
            dram_bound=False):
        """The config descriptor `d4` and the one DMA instruction that reads it."""
        from mlir.ir import MemRefType
        expand = d4.expand
        dram_addr = (_emit_i64_const(POINTER_TILE_BASE) if self.pointer_tile else
                     _emit_elem_addr_i64(self.dram, [dram_idx_val], self.dram_ty,
                                         self.bytes))
        spad_addr = _emit_elem_addr_i64(sram_mem, sram_indices,
                                        MemRefType(sram_mem.type), self.bytes)
        ind = self._indirect_fields(gather_axis, subtile_shape, expand)
        dram_base_i64, dram_bytes = (
            self._dram_bound_fields() if dram_bound else (None, 0))
        slots = _pack_desc(d4.shape, d4.dram, d4.spad, self.bytes,
                           VLANE_STRIDE, d4.vsa,
                           CONFIG_TYPE[self.dma_type], self.indirect,
                           ind.stride, ind.esize, hi=d4.hi, lo=d4.lo,
                           ind_dim=ind.dim, ind_lanes=ind.lanes, skip_axes=d4.skip,
                           dram_bytes=dram_bytes)
        if self.fill_only:
            slots[S_FILL] = self.masked_fill & 0xFFFFFFFFFFFFFFFF
        pairs4 = [(expand + d, lo, hi)
                  for d, lo, hi in (desc_masked_pairs or ())]
        strides4 = [(expand + d, v) for d, v in (desc_stride_pairs or ())]
        desc_ptr_val = self.descs.ptr(slots, pairs4, self.masked_fill, ind.addr,
                                      self.dyn_fill, strides4, self.dyn_ostride,
                                      ind.esize, dram_base_i64)
        _emit_dma_asm(CONFIG_DESC, desc_ptr_val, _emit_i64_const(0))
        _emit_dma_asm(self.dma_type, dram_addr, spad_addr)


def sram_footprints(op, vectorlane, interval_of):
    """(memref, [lo, hi) of its elements from its buffer's start, or None) of each scratchpad side
    the descriptor `_emit_direct` writes for transfer `op` reaches in one lane, the DMA's own walk
    (dma.hpp): the tile from its address, and a gather's index buffer; a cut's box only narrows it."""
    tx = _Tx(op, None, vectorlane)
    if len(walked_shape(op)) > DESCRIPTOR_DIMS:
        return [(tx.sram, None)] + ([(tx.offset_buf, None)] if tx.indirect else [])
    d4, sidx = _direct(tx)
    depth = _descriptor_depth(d4.shape, d4.spad, d4.vsa, vectorlane)
    at = _elem_offset(tx.sram_ty, sidx)
    reach = None if at is None else [(i, i) if isinstance(i, int) else interval_of(i)
                                     for i, _ in at[1]]
    out = [(tx.sram, None if reach is None or None in reach else (
        at[0] + sum(min(a * s, b * s) for (a, b), (_, s) in zip(reach, at[1])),
        at[0] + sum(max(a * s, b * s) for (a, b), (_, s) in zip(reach, at[1])) + depth))]
    if tx.indirect:
        axis = None if tx.gather_axis is None else d4.expand + tx.gather_axis
        rows = depth if axis is None else (d4.shape[axis] - 1) // _index_lanes(tx) + 1
        out.append((tx.offset_buf, (0, rows)))
    return out


def walk(op, vectorlane):
    """(dims, DRAM strides, lead) the descriptor `_emit_direct` writes for transfer `op` walks, its
    box narrowing them as moved_shape does, a uniform MVIN's lane copies included; leading extent-1
    pad dims are left out, and `lead` counts the pad dims kept ahead of the tile's axis 0."""
    tx = _Tx(op, None, vectorlane)
    if len(tx.tile_shape) > DESCRIPTOR_DIMS:
        _unfitted(tx)
    d4, _ = _direct(tx)
    shape = list(d4.shape)
    for d, (lo, hi) in (tx.box or {}).items():
        shape[d4.expand + d] = hi - lo
    start = next((i for i in range(d4.expand) if shape[i] != 1), d4.expand)
    return shape[start:], list(d4.dram[start:]), d4.expand - start


def _descriptor_depth(shape4, spad4, vsa, lanes):
    """How many elements of one lane's bank, from the spad address on, the DMA walks for a
    descriptor of `shape4` and `spad4` that spreads dim `vsa` over `lanes`."""
    outer = -(-shape4[vsa] // (VLANE_STRIDE * lanes))
    block = [s // shape4[vsa] * VLANE_STRIDE * outer if s > spad4[vsa] else s for s in spad4]
    dims = [VLANE_STRIDE if i == vsa else d for i, d in enumerate(shape4)]
    last = block[vsa] * VLANE_STRIDE * (outer - 1)
    return last + sum(b * (d - 1) for b, d in zip(block, dims)) + 1


def _index_lanes(tx):
    """How many lanes hold a gather's index buffer, as the descriptor names them."""
    from mlir.ir import MemRefType
    if is_uniform(tx.offset_buf):
        return 1
    return min(MemRefType(tx.offset_buf.type).shape[0], tx.vectorlane)


def _find_transfers(module):
    return [op.operation
            for region in module.operation.regions for b in region.blocks
            for op in walk_ops(b)
            if op.operation.name == TransferOp.OPERATION_NAME]


def _direct(tx):
    """The descriptor a direct transfer `tx` packs and the sram indices (an int, or tx's index
    value) its address is taken at: the one statement of both, emitted by _emit_direct and walked
    by sram_footprints."""
    d4 = tx._four_dims(walked_shape(tx.op), tx.dram_stride, tx.tile_stride,
                       tx.box, tx.skip_axes, tx.subtile)
    return d4, ([0] * (tx.spad_rank - 1) + [tx.sram_idx] if tx.spad_rank else [])


def _emit_direct(tx):
    from mlir.dialects import arith
    from mlir.ir import InsertionPoint
    with InsertionPoint(tx.op):
        d4, sidx = _direct(tx)
        sidx = [tx.const(i) if isinstance(i, int) else i for i in sidx]
        back = sum(lo * tx.dram_stride[d] for d, (lo, _hi) in (tx.box or {}).items())
        dram_idx = (tx.dram_idx if not back
                    else arith.AddIOp(tx.dram_idx, tx.const(-back)).result)
        tx.asm(d4, tx.sram, sidx, dram_idx, tx.subtile, tx.masked_pairs, tx.dyn_strides,
               gather_axis=tx.gather_axis, dram_bound=tx.dram_bound)


def _empty_dim(skip_axes, masked_pairs, expand, op):
    """A descriptor dim no skip and no runtime clamp names: an empty [0, 0) there puts every element
    outside the box but inside the skipped axes' boxes, which an MVIN fills (dma.hpp)."""
    named = {expand + d for d in skip_axes} | {expand + d for d, _lo, _hi in masked_pairs}
    for d in range(DESCRIPTOR_DIMS):
        if d not in named:
            return d
    raise NotImplementedError(
        f"a fill piece needs a descriptor dim no skip and no runtime clamp names, to be empty "
        f"on, and every dim is named: {op}")


def _unfitted(tx):
    """A tile of more than four axes that pytorchsim-fit-transfers-to-descriptor left uncut."""
    raise RuntimeError(
        f"a transfer of rank {len(tx.tile_shape)} reached the DMA lowering: "
        f"pytorchsim-fit-transfers-to-descriptor runs before this pass and leaves every uncut "
        f"transfer at rank <= {DESCRIPTOR_DIMS}: {tx.op}")


def _i32_signed(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def _lane_dim(tx, shape4, dram4, spad4, expand):
    """The dim the lanes ride: a spread side's axis 0; a uniform side's free extent-1 dim, which an
    MVIN widens in place to every lane at DRAM stride 0 and a spad stride no other dim exceeds, so
    each bank gets the whole tile and dma.hpp rescales no block stride; an MVOUT reads lane 0 alone."""
    if not tx.uniform:
        return expand
    named = ({expand + d for d in (tx.box or {})} | {expand + d for d in tx.skip_axes}
             | {expand + d for d, _lo, _hi in tx.masked_pairs}
             | {expand + d for d, _v in tx.dyn_strides}
             | ({expand + tx.gather_axis} if tx.indirect and tx.gather_axis is not None else set()))
    slot = _one_lane_slot(shape4, named, tx.op)
    assert slot not in named, (slot, named)
    if tx.dma_type != MVIN:
        return slot
    shape4[slot], dram4[slot], spad4[slot] = tx.vectorlane, 0, max(spad4)
    return slot


def _one_lane_slot(shape4, named, op):
    """The first extent-1 dim of `shape4` no runtime field or box names, refused by name if none is."""
    for d, extent in enumerate(shape4):
        if extent == 1 and d not in named:
            return d
    raise NotImplementedError(
        f"this tile sits in one bank, but its descriptor {shape4} has no extent-1 dim that a box, "
        f"a clamp, a runtime stride, a skip or a gather leaves free (named: {sorted(named)}), so "
        f"there is no dim to name as the lane axis -- for an MVIN, none to spread its copies over "
        f"every lane: {op}")


def _emit_i64_const(value):
    from mlir.dialects import arith
    from mlir.ir import IntegerAttr, IntegerType
    i64 = IntegerType.get_signless(64)
    return arith.ConstantOp(i64, IntegerAttr.get(i64, _i64_signed(value))).result


def _elem_offset(mtype, indices):
    """(the offset `mtype` states, [(index, its row-major stride)] of each index that moves it) of
    the element `indices` name: where a DMA end's address is taken. None when no offset is stated."""
    from mlir.ir import ShapedType
    so = strides_and_offset(mtype)
    if so is None or so[1] == ShapedType.get_dynamic_stride_or_offset():
        return None
    return so[1], [(i, st) for i, st in zip(indices, row_major(list(mtype.shape))) if st != 0]


def _emit_elem_addr_i64(memref_val, indices, mtype, elem_bytes):
    from mlir.dialects import arith, memref
    from mlir.ir import IndexType, IntegerAttr, IntegerType
    i64 = IntegerType.get_signless(64)
    idx_ty = IndexType.get()
    base = memref.ExtractAlignedPointerAsIndexOp(memref_val).result
    at = _elem_offset(mtype, indices)
    if at is None:
        raise NotImplementedError(
            f"a DMA end whose offset its type does not state: {mtype}")
    off = (arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, at[0])).result
           if at[0] else None)
    for ival, stride in at[1]:
        term = ival
        if stride != 1:
            term = arith.MulIOp(ival, arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, stride)).result).result
        off = term if off is None else arith.AddIOp(off, term).result
    if off is not None:
        byte = arith.MulIOp(off, arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, elem_bytes)).result).result
        base = arith.AddIOp(base, byte).result
    return arith.IndexCastOp(i64, base).result


def _pack_desc(shape4, dram4, spad4, elem_bytes, vlstride, vsa4, cfg_type, indirect,
               ind_stride, ind_esize, hi=None, lo=None, ind_dim=0, ind_lanes=0,
               skip_axes=0, dram_bytes=0):
    lo = [0, 0, 0, 0] if lo is None else list(lo)
    if hi is None:
        hi = list(shape4)
    masked = any(h < s for h, s in zip(hi, shape4)) or any(l > 0 for l in lo)
    def s2(a, b): return (a & 0xFFFFFFFF) | ((b & 0xFFFFFFFF) << 32)
    m = 0xFFFFFFFFFFFFFFFF
    slots = [
        s2(shape4[0], shape4[1]), s2(shape4[2], shape4[3]),
        s2(lo[0], lo[1]), s2(lo[2], lo[3]),
        s2(hi[0], hi[1]), s2(hi[2], hi[3]),
        dram4[0] & m, dram4[1] & m, dram4[2] & m, dram4[3] & m,
        spad4[0] & m, spad4[1] & m, spad4[2] & m, spad4[3] & m,
        (elem_bytes & 0xFFFF) | ((vlstride & 0xFFFF) << 16) | ((vsa4 & 0xFF) << 32)
            | ((cfg_type & 0xFF) << 40)
            | (((F_INDIRECT if indirect else 0) | (F_MASKED if masked else 0)
                | ((skip_axes & 0xF) << F_SKIP_SHIFT)) << 48),
        0,
        (ind_stride & 0xFFFF) | ((ind_esize & 0xFFFF) << 16)
            | ((ind_dim & 0xFF) << 32) | ((ind_lanes & 0xFFFF) << 48),
        0,
        0,
        dram_bytes & m,
    ]
    return slots


def _emit_dma_asm(func7, rs1, rs2):
    from mlir.dialects import llvm
    llvm.InlineAsmOp(None, [rs1, rs2], insn(func7, "$0", "$1"), CONSTRAINTS,
                     has_side_effects=True, asm_dialect=0)


def insn(func7, rs1, rs2):
    """The one spelling of a DMA instruction: custom-1, funct3 3, `func7`, its two sources as the
    caller's asm names them -- this pass's MLIR inline asm and the ELF wrapper's C alike."""
    return f".insn r CUSTOM_1, 0x3, {func7}, x0, {rs1}, {rs2}"
