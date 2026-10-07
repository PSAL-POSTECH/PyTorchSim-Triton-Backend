"""torchsim-lower-to-dma: every torchsim.transfer -> CONFIG_DESC + MVIN/MVOUT
inline asm, with no memref.dma_start in between. NOT Gemmini, despite the
upstream names. timing=True erases the transfer instead (the gem5 cycle path)."""

from collections import namedtuple

from ...dialects.torchsim import (DRAM_ARG_POINTER_TILE, POINTER_TILE_BASE, TransferOp,
                                  WaitOp)
from .p01_convert_accesses_to_transfer import CUT_WRAP
from .lib_ir import (elem_bytes, int_array,
                            strides_and_offset, walk_ops, _i64_signed)
from .lib_bank_layout import bank_strides, lanes_of, why_no_lane_axis, row_major

CONSTRAINTS = "r,r,~{memory},~{dirflag},~{fpsr},~{flags}"

CONFIG_DESC = 7
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
MAX_TENSOR_DIM = 4
VLANE_STRIDE = 1

_Desc4 = namedtuple("_Desc4", "expand shape dram spad vsa lo hi skip")

_Indirect = namedtuple("_Indirect", "addr stride esize dim lanes")

_Peel = namedtuple("_Peel", "peeled inner masked_pairs stride_pairs vlane")


def _erase_globals_nothing_reads(module):
    """Every global whose last reader left with the waits and the transfers. An
    `external` declaration still reaches p11's reservation, which would give a bank
    slice to storage no instruction touches; symbol-dce keeps public symbols."""
    top = module.operation.regions[0].blocks[0]
    for g in [o for o in walk_ops(top)
              if o.operation.name == "memref.get_global"
              and not list(o.operation.results[0].uses)]:
        g.operation.erase()
    read = {str(o.operation.attributes["name"]).strip('@" ') for o in walk_ops(top)
            if o.operation.name == "memref.get_global"}
    for g in [o for o in walk_ops(top)
              if o.operation.name == "memref.global"
              and str(o.operation.attributes["sym_name"]).strip('" ') not in read]:
        g.operation.erase()


def run(module, timing=False, vectorlane=None, **_):
    """Lower every torchsim.transfer in `module`. `vectorlane` HAS NO DEFAULT ON
    PURPOSE: a default stands in for the machine's lane count silently, and a
    gather's index buffer names how many lanes hold it."""
    if vectorlane is None:
        raise ValueError(
            "lower_to_dma needs the lane count: a gather's descriptor names how "
            "many lanes hold its index. Pass vectorlane=machine.VECTORLANE_SIZE.")
    for w in [o for o in walk_ops(module.operation.regions[0].blocks[0])
              if o.operation.name == WaitOp.OPERATION_NAME]:
        w.operation.erase()
    descs = _Descriptors(module.operation.regions[0].blocks[0])
    for op in _find_transfers(module):
        tx = _Tx(op, descs, vectorlane)
        if not timing:
            for piece in _wrap_pieces(tx):
                (_emit_direct if len(tx.tile_shape) <= MAX_TENSOR_DIM else
                 _emit_collapsed if _collapse_groups(tx)[0] is not None else
                 _emit_peeled)(piece)
        op.erase()
    _erase_globals_nothing_reads(module)


class _Descriptors:
    """The DMA descriptor globals of one module, and pointers into them.
    `by_slots` dedupes fully static descriptors by their slot tuple; one with a
    runtime-written field is unique and never enters it."""

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
        """`g`'s aligned pointer as an i64 -- what the CONFIG_DESC instruction
        takes. Both paths below end here, deduped global or freshly minted."""
        from mlir.dialects import arith, memref
        return arith.IndexCastOp(
            self.i64,
            memref.ExtractAlignedPointerAsIndexOp(g).result).result

    def _mint_i32_view(self, slots, masked_pairs, fill):
        """A UNIQUE global holding these slots as i32 words, and a get_global
        reading it. Unique because the caller is about to store runtime values
        into it, and i32 because a field is written a word at a time."""
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
        """Every field whose value is only known at run time, stored into `g`.
        ONE WORD AT A TIME: the global's initialiser can only carry constants,
        so a runtime field is written over its own words after the fact."""
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
        """Byte pointer to the DMA descriptor global: a DEDUPED i64 one when
        every field is static, a UNIQUE i32-view one (same bytes) otherwise, so
        the runtime fields can be stored a word at a time."""
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


class _Tx:
    """One `torchsim.transfer`, read once. The three emitters below differ only in
    HOW MANY DIMS they hand a four-dim descriptor, so everything they share --
    the fixed slots, the tail, and the tile's own shape and strides -- is here."""

    def __init__(self, op, descs, vectorlane):
        from mlir.ir import IndexType, IntegerAttr, IntegerType, MemRefType
        self.op, self.descs = op, descs
        self.vectorlane = vectorlane
        self.i64 = IntegerType.get_signless(64)
        self.idx_ty = IndexType.get()
        tx = op.opview
        self.dram, self.dram_idx = tx.dram, tx.dram_idx
        self.sram, self.sram_idx = tx.sram, tx.sram_idx
        self.offset_buf = tx.index
        self.indirect = self.offset_buf is not None
        kind = tx.dma_kind.value
        self.dma_type = FUNC7.get(kind)
        if self.dma_type is None:
            raise NotImplementedError(
                f"transfer carries dma_kind={kind!r}, which has no instruction "
                f"here; FUNC7 is the whole set this pass can emit")
        self.one_bank = _is_one_bank(op, self.sram)
        self.dram_bound = tx.dram_bound
        self.pointer_tile = (IntegerAttr(tx.dram_arg).value
                             == DRAM_ARG_POINTER_TILE)
        self.cut = _read_cut(op)
        self.box, self.skip_axes = None, ()
        self.gather_axis = (None if tx.indirect_dim is None
                            else IntegerAttr(tx.indirect_dim).value)
        self.dram_stride = int_array(tx.dram_stride)
        self.subtile = (None if tx.subtile_size is None
                        else int_array(tx.subtile_size))
        masked = [] if tx.masked_axes is None else int_array(tx.masked_axes)
        clamp = list(tx.clamp)
        self.masked_pairs = [(a, clamp[2 * i], clamp[2 * i + 1])
                             for i, a in enumerate(masked)]
        self.masked_fill = (0 if tx.masked_axes is None
                            else IntegerAttr(tx.masked_fill).value)
        self.dyn_fill = tx.dyn_fill
        strided = ([] if tx.dyn_stride_axes is None
                   else int_array(tx.dyn_stride_axes))
        self.dyn_strides = list(zip(strided, tx.dyn_strides))
        self.dyn_ostride = tx.dyn_offset_stride
        self.sram_ty = MemRefType(self.sram.type)
        self.elem = self.sram_ty.element_type
        self.space = self.sram_ty.memory_space
        self.bytes = elem_bytes(self.elem)
        self.dram_ty = MemRefType(self.dram.type)
        self.tile_shape = list(self.sram_ty.shape)
        self.spad_rank = len(self.tile_shape)
        self.tile_stride = bank_strides(self.sram)
        self.eff = [i for i, e in enumerate(self.tile_shape) if e > 1]

    def const(self, v):
        from mlir.ir import IntegerAttr
        from mlir.dialects import arith
        return arith.ConstantOp(self.idx_ty, IntegerAttr.get(self.idx_ty,
                                                             v)).result

    def _four_dims(self, cfg_shape, dram_strides, spad_strides, vsa_val,
                   box, skip_axes, subtile_shape):
        """Everything an emitter hands over, PADDED TO FOUR FROM THE LEFT, and
        how many were added. The descriptor always has four dims and every
        emitter has at most four, so every per-axis field steps by `expand`."""
        expand = MAX_TENSOR_DIM - len(cfg_shape)
        shape4 = [1] * expand + list(cfg_shape)
        lo4 = hi4 = None
        if box:
            if subtile_shape is not None:
                raise NotImplementedError(
                    f"a wrapped tile that is also sub-tiled: the descriptor "
                    f"describes the subtile and the box names tile rows: "
                    f"{self.op}")
            lo4, hi4 = [0] * MAX_TENSOR_DIM, list(shape4)
            for d, (lo, hi) in box.items():
                lo4[expand + d], hi4[expand + d] = lo, hi
        return _Desc4(
            expand, shape4,
            [0] * expand + list(dram_strides),
            [0] * expand + list(spad_strides),
            (_choose_one_lane_slot(shape4, self.op) if vsa_val is None
             else vsa_val + expand),
            lo4, hi4, sum(1 << (expand + d) for d in skip_axes))

    def _indirect_fields(self, gather_axis, subtile_shape, expand):
        """The index buffer as the descriptor names it, all zero when this
        transfer is not a gather. REFUSED, NOT DEGRADED: an index holding one
        entry per position cannot be walked per element off a peeled axis."""
        from mlir.ir import IntegerAttr, MemRefType
        from mlir.dialects import arith, memref
        if self.gather_axis is not None and (gather_axis is None
                                             or subtile_shape is not None):
            raise NotImplementedError(
                f"the index buffer holds one entry per position on axis "
                f"{self.gather_axis}, and this descriptor cannot address it "
                f"that way -- the axis is peeled or sub-tiled, so the "
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
        lanes = (1 if _is_one_bank(self.op, self.offset_buf)
                 else min(off_ty.shape[0], self.vectorlane))
        if not 0 < lanes <= 0xFFFF:
            raise NotImplementedError(
                f"the index buffer is held by {lanes} lanes, which the "
                f"descriptor's uint16 field cannot name: {self.op}")
        return _Indirect(addr, stride, esize, dim, lanes)

    def _dram_bound_fields(self):
        """`(the argument's base as an i64, its size in bytes)` -- THE ARGUMENT, NOT THE
        TILE: `self.dram` is the view over all of it. A tile of pointers' transfer is bound
        by all of memory from 0, so only what a mask parked below 0 falls outside."""
        from mlir.dialects import arith, memref
        if self.pointer_tile:
            return _emit_i64_const(0), 1 << 63
        return (arith.IndexCastOp(
            self.i64,
            memref.ExtractAlignedPointerAsIndexOp(self.dram).result).result,
            self.dram_ty.shape[0] * elem_bytes(self.dram_ty.element_type))

    def asm(self, sram_mem, sram_indices, dram_idx_val, vsa_val, desc_shape,
            desc_dram_strides, desc_spad_strides, subtile_shape,
            desc_masked_pairs=None, desc_stride_pairs=None, gather_axis=None,
            desc_box=None, skip_axes=(), dram_bound=False):
        """The config descriptor and the one DMA instruction that reads it.
        EXPANDED TO FOUR DIMS HERE and nowhere else: the descriptor has four
        and every emitter has at most four, so the padding is this one place."""
        from mlir.ir import MemRefType
        d4 = self._four_dims(
            subtile_shape if subtile_shape is not None else desc_shape,
            desc_dram_strides, desc_spad_strides, vsa_val, desc_box,
            skip_axes, subtile_shape)
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
        pairs4 = [(expand + d, lo, hi)
                  for d, lo, hi in (desc_masked_pairs or ())]
        strides4 = [(expand + d, v) for d, v in (desc_stride_pairs or ())]
        desc_ptr_val = self.descs.ptr(slots, pairs4, self.masked_fill, ind.addr,
                                      self.dyn_fill, strides4, self.dyn_ostride,
                                      ind.esize, dram_base_i64)
        _emit_dma_asm(CONFIG_DESC, desc_ptr_val, _emit_i64_const(0))
        _emit_dma_asm(self.dma_type, dram_addr, spad_addr)


def _find_transfers(module):
    """Every torchsim.transfer in the module, collected before any is rewritten."""
    return [op.operation
            for region in module.operation.regions for b in region.blocks
            for op in walk_ops(b)
            if op.operation.name == TransferOp.OPERATION_NAME]


def _wrap_pieces(tx):
    """This transfer as the strided pieces its cuts make of it, or itself.
    The axes cut independently, so the pieces are their PRODUCT -- each a box.
    They share the tile, the extent and the fill; only the rows differ."""
    import copy, itertools
    from mlir.dialects import arith
    from mlir.ir import InsertionPoint
    if not tx.cut:
        yield tx
        return
    if len(tx.tile_shape) > MAX_TENSOR_DIM:
        raise NotImplementedError(
            f"a wrapped tile of rank {len(tx.tile_shape)} would be collapsed or "
            f"peeled, and a piece's box does not survive either remap: {tx.op}")
    clamped = {d for d, _lo, _hi in tx.masked_pairs}
    if clamped & set(tx.cut):
        raise NotImplementedError(
            f"axis {sorted(clamped & set(tx.cut))} is both cut and clamped: "
            f"outside the box the clamp wants a fill and the piece wants a "
            f"skip, and one box cannot say both: {tx.op}")
    axes = sorted(tx.cut)
    runs = {d: _runs_of(*tx.cut[d]) for d in axes}
    with InsertionPoint(tx.op):
        for combo in itertools.product(*(runs[d] for d in axes)):
            piece = copy.copy(tx)
            piece.box = {d: (start, start + count)
                         for d, (start, count, _delta) in zip(axes, combo)}
            piece.skip_axes = axes
            delta = sum(c[2] for c in combo)
            piece.dram_idx = (tx.dram_idx if delta == 0 else
                              arith.AddIOp(tx.dram_idx, tx.const(delta)).result)
            yield piece


def _emit_direct(tx):
    """A tile the descriptor can already describe: at most four dims, handed
    over as they are. The scratchpad offset is linear, so only the last index
    carries it -- ON THE TILE'S OWN MEMREF."""
    from mlir.ir import InsertionPoint
    with InsertionPoint(tx.op):
        sidx = ([tx.const(0)] * (tx.spad_rank - 1) + [tx.sram_idx]
                if tx.spad_rank else [])
        tx.asm(tx.sram, sidx, tx.dram_idx, None if tx.one_bank else 0, tx.tile_shape,
               tx.dram_stride, tx.tile_stride, tx.subtile, tx.masked_pairs,
               tx.dyn_strides, gather_axis=tx.gather_axis,
               desc_box=tx.box, skip_axes=tx.skip_axes,
               dram_bound=tx.dram_bound)


def _emit_collapsed(tx):
    """More dims than the descriptor holds, but at most four with extent > 1:
    collapse the unit dims into their neighbours and describe the result. Each
    group contributes ONE stride, its `keep` dim's, so nothing else may."""
    from mlir.ir import ArrayAttr, InsertionPoint, IntegerAttr, MemRefType, Operation
    shape = tx.tile_shape
    groups, target = _collapse_groups(tx)

    def groups_of(d):
        return next(gi for gi, g in enumerate(groups) if d in g)

    reassoc = ArrayAttr.get(
        [ArrayAttr.get([IntegerAttr.get(tx.i64, d)
                        for d in g]) for g in groups])
    collapsed_ty = MemRefType.get(target, tx.elem, memory_space=tx.space)
    keep = [next((d for d in reversed(g) if shape[d] > 1), g[-1])
            for g in groups]
    dr = [tx.dram_stride[i] for i in keep]
    tl = [tx.tile_stride[i] for i in keep]
    st = [tx.subtile[i] for i in keep] if tx.subtile is not None else None
    mp = [(groups_of(d), lo, hi) for d, lo, hi in tx.masked_pairs]
    for d, _v in tx.dyn_strides:
        if d not in keep:
            raise NotImplementedError(
                f"runtime stride on tile axis {d}, which is squeezed "
                f"into a collapsed group whose stride comes from "
                f"axis {keep[groups_of(d)]}")
    sp = [(keep.index(d), v) for d, v in tx.dyn_strides]
    with InsertionPoint(tx.op):
        sram_c = Operation.create(
            "memref.collapse_shape", results=[collapsed_ty],
            operands=[tx.sram],
            attributes={"reassociation": reassoc}).results[0]
        sidx = [tx.const(0)] * (len(target) - 1) + [tx.sram_idx]
        tx.asm(sram_c, sidx, tx.dram_idx, None if tx.one_bank else groups_of(0), target,
               dr, tl, st, mp, sp,
               gather_axis=None if tx.gather_axis is None
               else groups_of(tx.gather_axis), dram_bound=tx.dram_bound)


def _collapse_groups(tx):
    """How this tile collapses, or (None, None) if it does not fit the descriptor.
    TWO RULES, ONE QUESTION: unit dims fold into a neighbour, and past four
    effective dims adjacent dims fold where both sides are contiguous."""
    if len(tx.eff) <= MAX_TENSOR_DIM:
        return _choose_squeeze_groups(tx.tile_shape)
    groups, target = _contiguous_groups(tx)
    if groups is None or len(target) > MAX_TENSOR_DIM:
        return None, None
    return groups, target


def _emit_peeled(tx):
    """More than four dims with extent > 1: the outermost become an affine.for
    nest and the innermost four go to the descriptor. THE LANE AXIS MAY NOT BE
    PEELED -- an outer loop iterates it, and the DMA spreads it."""
    from mlir.ir import (AffineExpr, AffineMap, AffineMapAttr, DenseI32ArrayAttr,
                         DenseI64ArrayAttr, InsertionPoint, Operation)
    from mlir.dialects import affine
    shape, stride = tx.tile_shape, tx.tile_stride
    cut = _peel_split(tx)
    peeled, inner = cut.peeled, cut.inner
    ndim = tx.spad_rank
    inner_shape = [shape[d] for d in inner]
    dr = [tx.dram_stride[d] for d in inner]
    tl = [stride[d] for d in inner]
    st = [tx.subtile[d] for d in inner] if tx.subtile is not None else None

    mshape = list(tx.sram_ty.shape)
    keep = list(inner)
    static_sizes = [1] * ndim
    for m in keep:
        static_sizes[m] = mshape[m]
    res_ty = _peel_subview_type(tx.sram_ty, keep, [mshape[m] for m in keep],
                                tx.elem, tx.space)

    cur_ip = InsertionPoint(tx.op)
    ivs = []
    for d in peeled:
        floop = affine.AffineForOp(0, shape[d], 1, ip=cur_ip)
        ivs.append(floop.induction_variable)
        with InsertionPoint(floop.body):
            affine.AffineYieldOp([])
        cur_ip = InsertionPoint.at_block_terminator(floop.body)

    npeel = len(peeled)
    with cur_ip:
        sub = Operation.create(
            "memref.subview", results=[res_ty], operands=[tx.sram],
            attributes={"static_offsets": DenseI64ArrayAttr.get([0] * ndim),
                        "static_sizes": DenseI64ArrayAttr.get(static_sizes),
                        "static_strides": DenseI64ArrayAttr.get([1] * ndim),
                        "operandSegmentSizes": DenseI32ArrayAttr.get([1, 0, 0, 0])}
        ).results[0]
        sram_expr = AffineExpr.get_dim(0) * stride[peeled[0]]
        for k in range(1, npeel):
            sram_expr = sram_expr + AffineExpr.get_dim(k) * stride[peeled[k]]
        sram_off_val = Operation.create(
            "affine.apply", results=[tx.idx_ty], operands=list(ivs),
            attributes={"map": AffineMapAttr.get(AffineMap.get(npeel, 0, [sram_expr]))}
        ).results[0]
        dram_expr = AffineExpr.get_dim(0)
        for k in range(npeel):
            dram_expr = dram_expr + AffineExpr.get_dim(k + 1) * tx.dram_stride[peeled[k]]
        dram_idx_val = Operation.create(
            "affine.apply", results=[tx.idx_ty], operands=[tx.dram_idx, *ivs],
            attributes={"map": AffineMapAttr.get(AffineMap.get(npeel + 1, 0, [dram_expr]))}
        ).results[0]
        sidx = [tx.const(0)] * (len(keep) - 1) + [sram_off_val]
        tx.asm(sub, sidx, dram_idx_val, cut.vlane, inner_shape, dr, tl, st,
               cut.masked_pairs, cut.stride_pairs,
               gather_axis=(inner.index(tx.gather_axis)
                            if tx.gather_axis in inner else None),
               dram_bound=tx.dram_bound)


def _i32_signed(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def _is_one_bank(op, sram):
    """Does this scratchpad buffer sit in one bank, off its type? Refused outside the
    scratchpad."""
    n = lanes_of(sram)
    if n is None:
        raise RuntimeError(
            f"the transfer's scratchpad side has no recorded lane axis, and "
            f"this pass does not decide layouts: {why_no_lane_axis(sram)}: {op}")
    return n == 1


def _read_cut(op):
    """{axis: (kind, a, b, stride, extent)} for a cut transfer, else None.
    p03 mints the FACT because it is the pass that knows the wrap and the step;
    the runs are built here because only now is the lane axis fixed."""
    cut = op.opview.cut
    if cut is None:
        return None
    out = {}
    for d, a in enumerate(cut):
        v = list(int_array(a))
        if v:
            out[d] = tuple(v)
    return out or None


def _choose_one_lane_slot(shape4, op):
    """The descriptor dim a one-bank tile names as its lane axis: the DMA has no
    "one bank" encoding, so one lane is spelled by an extent-1 dim."""
    for d, extent in enumerate(shape4):
        if extent == 1:
            return d
    raise NotImplementedError(
        f"this tile is not spread over the lanes, but its descriptor is "
        f"{shape4} -- four dims that all hold something, so there is no "
        f"extent-1 dim to name as the lane axis and no other way to say "
        f"'one lane'. The DMA would spread {shape4[0]} of them: {op}")


def _emit_i64_const(value):
    """An i64 constant. The type is derived here rather than closed over --
    the context is active wherever this is called."""
    from mlir.dialects import arith
    from mlir.ir import IntegerAttr, IntegerType
    i64 = IntegerType.get_signless(64)
    return arith.ConstantOp(i64, IntegerAttr.get(i64, _i64_signed(value))).result


def _emit_elem_addr_i64(memref_val, indices, mtype, elem_bytes):
    """The byte address of element `indices` of `memref_val`, as i64. The aligned pointer
    is the BUFFER's, so a view's own offset is added: a subview read from its buffer's
    start is read at the wrong elements, in silence."""
    from mlir.dialects import arith, memref
    from mlir.ir import IndexType, IntegerAttr, IntegerType, ShapedType
    i64 = IntegerType.get_signless(64)
    idx_ty = IndexType.get()
    base = memref.ExtractAlignedPointerAsIndexOp(memref_val).result
    strides = row_major(list(mtype.shape))
    so = strides_and_offset(mtype)
    if so is None or so[1] == ShapedType.get_dynamic_stride_or_offset():
        raise NotImplementedError(
            f"a DMA end whose offset its type does not state: {mtype}")
    off = (arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, so[1])).result
           if so[1] else None)
    for k, ival in enumerate(indices):
        if strides[k] == 0:
            continue
        term = ival
        if strides[k] != 1:
            term = arith.MulIOp(ival, arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, strides[k])).result).result
        off = term if off is None else arith.AddIOp(off, term).result
    if off is not None:
        byte = arith.MulIOp(off, arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, elem_bytes)).result).result
        base = arith.AddIOp(base, byte).result
    return arith.IndexCastOp(i64, base).result


def _pack_desc(shape4, dram4, spad4, elem_bytes, vlstride, vsa4, cfg_type, indirect,
               ind_stride, ind_esize, hi=None, lo=None, ind_dim=0, ind_lanes=0,
               skip_axes=0, dram_bytes=0):
    """The 20 i64 slots of the C dma_descriptor, little-endian. `hi` defaults to
    the extent and `lo` to 0; a box is either end moved off that, and moving one
    is what raises F_MASKED and makes the box visible to the DMA at all."""
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
    llvm.InlineAsmOp(None, [rs1, rs2], _asm(func7), CONSTRAINTS,
                     has_side_effects=True, asm_dialect=0)


def _asm(func7):
    return f".insn r CUSTOM_1, 0x3, {func7}, x0, $0, $1"


def _runs_of(kind, a, b, st, n):
    """One axis's runs from its fact: (first tile row, how many, DRAM delta). A WRAP's lap `j(p)
    = (o + p*st) // W` is monotone, so a run is a lap and it carries `-j*W`. A STEP's run j
    is `p in [jK, min((j+1)K, n))` and carries `j*st`, the address being constant in p."""
    if kind == CUT_WRAP:
        W, o = a, b
        runs, p0 = [], 0
        for p in range(1, n + 1):
            if p == n or (o + p * st) // W != (o + p0 * st) // W:
                runs.append((p0, p - p0, -((o + p0 * st) // W) * W))
                p0 = p
        return runs
    return [(j, min(a, n - j), j // a * st) for j in range(0, n, a)]


def _choose_squeeze_groups(shape):
    """Group source dims so each group's product is one effective (non-unit)
    dim; unit dims attach to a neighbour, trailing ones to the last group, and
    an all-ones shape to a single group. Returns (groups, target_shape)."""
    groups, cur = [], []
    for i, e in enumerate(shape):
        cur.append(i)
        if e > 1:
            groups.append(cur)
            cur = []
    if cur:
        if groups:
            groups[-1] += cur
        else:
            groups.append(cur)
    import math
    target = [math.prod(shape[d] for d in g) for g in groups]
    return groups, target


def _contiguous_groups(tx):
    """Adjacent dims merged where BOTH sides walk them contiguously, and NEVER across the lane
    axis -- a merged lane axis spreads over a different number of lanes. (None, None) when
    nothing merges."""
    import math
    if tx.subtile is not None:
        return None, None
    shape, dram, tile = tx.tile_shape, tx.dram_stride, tx.tile_stride
    fixed = ({d for d, _lo, _hi in tx.masked_pairs}
             | {d for d, _v in tx.dyn_strides})
    vlane = None if tx.one_bank else 0
    groups = [[0]]
    for i in range(1, len(shape)):
        p = groups[len(groups) - 1][-1]
        if (i != vlane and p != vlane and i not in fixed and p not in fixed
                and dram[p] == dram[i] * shape[i]
                and tile[p] == tile[i] * shape[i]):
            groups[len(groups) - 1].append(i)
        else:
            groups.append([i])
    if len(groups) == len(shape):
        return None, None
    return groups, [math.prod(shape[d] for d in g) for g in groups]


def _peel_split(tx):
    """The axis split for a tile with more than four dims. THE OUTER LOOP IS
    NOT A DESCRIPTOR AXIS, so anything the descriptor would have to say about a
    peeled axis -- a clamp, a runtime stride -- is refused."""
    peeled, inner = _seats_for_the_descriptor(tx)
    if any(d in peeled for d, _lo, _hi in tx.masked_pairs):
        raise NotImplementedError("masked-DMA clamp on a peeled (outer-loop) axis")
    if any(d in peeled for d, _v in tx.dyn_strides):
        raise NotImplementedError("runtime stride on a peeled (outer-loop) axis")
    return _Peel(
        peeled, inner,
        [(inner.index(d), lo, hi) for d, lo, hi in tx.masked_pairs if d in inner],
        [(inner.index(d), v) for d, v in tx.dyn_strides if d in inner],
        None if tx.one_bank else 0)


def _seats_for_the_descriptor(tx):
    """(what an outer loop iterates, what the descriptor gets). Axis 0 rides the lanes and only
    the DMA spreads it, so a banked tile keeps it among the descriptor's four and the outermost
    of the rest yields; the others are the innermost, being the contiguous ones."""
    peeled, inner = tx.eff[:-MAX_TENSOR_DIM], tx.eff[-MAX_TENSOR_DIM:]
    if tx.one_bank or not peeled:
        return peeled, inner
    inner = [0] + tx.eff[-(MAX_TENSOR_DIM - 1):]
    return [d for d in tx.eff if d not in inner], inner


def _peel_subview_type(src_ty, keep, keep_shape, elem, space):
    """The peel's rank-reducing memref.subview type. Its strides are the SOURCE's,
    not the tile's banked ones -- the verifier compares against the memref being
    viewed. See tests/cases/mechanism/tile/tile_five_dim_peel_lane_inner.py."""
    from mlir.ir import MemRefType, StridedLayoutAttr
    src = _memref_strides(src_ty)
    return MemRefType.get(keep_shape, elem,
                          layout=StridedLayoutAttr.get(0, [src[d] for d in keep]),
                          memory_space=space)


def _memref_strides(ty):
    """What a memref type itself says its strides are (identity layout included).
    A type stating no strided layout has none, and this lowering has no second
    description of where its elements are -- so it is refused by name."""
    so = strides_and_offset(ty)
    if so is None:
        raise NotImplementedError(
            f"a memref whose layout is not strided reached the DMA lowering, "
            f"which has nothing else that says where its elements are: {ty}")
    return so[0]
