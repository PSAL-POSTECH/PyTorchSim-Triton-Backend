"""tts.{load,store,gather,scatter}, and loads and stores through a tile of pointers ->
pytorchsim.transfer."""

from collections import namedtuple

from ...contract.stop import Reason, Refusal

from ...dialects.pytorchsim import (DRAM_ARG_POINTER_TILE, PER_AXIS, POINTER_TILE_BASE, Digit,
                                  TransferOp, cut_digits, cut_entry, cut_offset, is_load,
                                  is_load_kind, operand_number)
from .lib_ir import (GATHER_MASK, GATHER_OFF, GATHER_OTHER, GATHER_PTR, GS_BASE_SEG, GS_MASK_SEG,
                     GS_OFFSET_SEG, GS_SHAPE_SEG, GS_STRIDE_SEG, attrs_of, elem_bits, elem_bytes,
                     elem_type, elementwise_generic, funcs, gather, holds_type, int_array,
                     int_attr, is_shaped, is_type, name_of, op_of, operand_segments,
                     parallel_iterator, producer, rank_of, splat, walk, zero_of)

FORBIDS = ("tts.*", "tt.*", "ttx.*")

KDYN = -9223372036854775808
BOXED_ACCESS = ("tts.load", "tts.store")
_GATHER_PTR = "tts.make_gather_scatter_tptr"
_TPTR_BASE_OFFSET_SEG = 4
_TPTR_FLAT_SHAPE_SEG = 5
_GS_ATTRS = ("gather_scatter_dim", "sizes", "static_offsets",
             "static_shape", "static_strides")
MAX_WRAP_PIECES = 12
_PTR_CARRIERS = ("linalg.fill", "linalg.generic", "tensor.empty")
_PTR_IDENTITY = ("tptr.inttoptr", "tptr.ptrtoint")
_ADDRESS_AGNOSTIC = ("arith.select",)

_Access = namedtuple("_Access", "op ptr kind sizes elem dram arg_index")
_PointerAccess = namedtuple("_PointerAccess", "kind addr value mask fill")


class _Fatal(Exception):
    pass


class BaseUntraced(_Fatal, Refusal):
    """A make_tptr base that does not trace to a function argument."""
    reason = Reason.POINTER_BASE_UNTRACED


class FillVaries(_Fatal, Refusal):
    """A tensor load whose other= varies per element."""
    reason = Reason.FILL_VARIES


class PointerTileUnconverted(_Fatal, Refusal):
    """A tile of pointers through an op this pass does not rewrite."""
    reason = Reason.POINTER_TILE_UNCONVERTED


def run(module, arg_numels=None, **_opts):
    """The table of contents, in the file's order: 1 read masks into selects (module-wide), then per
    function 2 tiles of pointers, 3 write masks, 4 every access into a transfer, 5 the scalar
    pointers left."""
    from mlir.ir import Location

    if arg_numels is None:
        raise _Fatal("arg_numels required: {arg index -> element count} for the "
                     "DRAM args (the memref<*xf32> params are unranked).")

    _empty_read_mask_carriers(module, arg_numels)

    loc = Location.unknown()
    with loc:
        for func in funcs(module):
            cx = _Ctx(func, arg_numels)
            _rewrite_pointer_tiles(cx, func, loc)
            _empty_write_mask_carriers(cx, func, loc)
            for op in _get_ordered_accesses(func):
                _rewrite_access(cx, op)
            _lower_ptr_to_int(cx, func)
            _drop_dead_ptr_casts(func)

    from mlir.passmanager import PassManager
    PassManager.parse(
        "builtin.module(func.func(canonicalize,cse))").run(module.operation)
    return module


class _Ctx:

    def __init__(self, func, arg_numels):
        from mlir.ir import IndexType, IntegerType
        self.arg_numels = arg_numels
        self.entry = func.operation.regions[0].blocks[0]
        self.block_args = list(self.entry.arguments)
        # A DRAM argument is numbered among the TENSORS, in `dram_arg`, `arg_numels` and TOGSim's
        # layout alike: a scalar parameter between two tensors is no argument of the DRAM.
        self.tensor_args = [a for a in self.block_args if str(a.type).startswith("memref<")]
        self.i64 = IntegerType.get_signless(64)
        self.idx = IndexType.get()
        self._consts, self._flats = {}, {}
        self.wide_index = {}
        self.flat_index = {}
        self.staged = {}
        self.claimed_index = set()

    def _i64_arr(self, vals):
        from mlir.ir import AttrBuilder
        return AttrBuilder.get("I64ArrayAttr")([int(v) for v in vals], context=None)

    def const(self, v):
        """An index constant, hoisted to the entry block and reused."""
        from mlir.ir import InsertionPoint, IntegerAttr
        from mlir.dialects import arith
        if v not in self._consts:
            with InsertionPoint.at_block_begin(self.entry):
                self._consts[v] = arith.ConstantOp(
                    self.idx, IntegerAttr.get(self.idx, v)).result
        return self._consts[v]

    def flat_view(self, memref_arg, arg_index):
        """rank-1 memref<Nxelem> over the whole DRAM tensor."""
        from mlir.ir import InsertionPoint, MemRefType, ShapedType
        from mlir.dialects import memref
        if arg_index in self._flats:
            return self._flats[arg_index]
        numel = self.arg_numels[arg_index]
        elem = ShapedType(memref_arg.type).element_type
        with InsertionPoint.at_block_begin(self.entry):
            v = memref.ReinterpretCastOp(
                MemRefType.get([numel], elem), memref_arg, [], [], [],
                static_offsets=[0], static_sizes=[numel],
                static_strides=[1]).result
        self._flats[arg_index] = v
        return v

    def any_argument(self):
        """The DRAM operand a tile of pointers' transfer carries because the op needs one: the first
        argument's view."""
        i, arg = next((i, a) for i, a in enumerate(self.block_args)
                      if str(a.type).startswith("memref<"))
        return self.flat_view(arg, i)

    def base_memref(self, base):
        """A pointer base -> the memref arg behind its unrealized_conversion_cast, and that arg's
        index."""
        cast = getattr(base, "owner", None)
        cast = op_of(cast)
        if cast is None or cast.name != "builtin.unrealized_conversion_cast":
            raise _Fatal(f"make_tptr base is not an unrealized_conversion_cast "
                         f"from a memref arg: {base}")
        memref_arg = _get_through_casts(cast.operands[0])
        for i, a in enumerate(self.tensor_args):
            if a == memref_arg:
                return memref_arg, i
        raise BaseUntraced(f"make_tptr base does not trace to a function arg: "
                     f"{memref_arg}")

    def mask_and_fill(self, access_op, sizes, elem):
        """The clamp + `other=` fill of one tts.load/store -> (masked axes, clamp operands, runtime
        fill operand or None, static fill bits)."""
        from mlir.dialects import arith
        dims, lows, mask_ops, low_ops, other = _get_mask_box(access_op, sizes)
        covers = _covered_axes(dims, lows, sizes)
        fill, dyn_fill = 0, False
        if other is not None and dims and not all(covers):
            bits = _get_fill_bits(other, elem)
            fill, dyn_fill = (0, True) if bits is None else (bits, False)
        axes, clamp = self._clamp_operands(access_op, dims, lows, covers,
                                           mask_ops, low_ops)
        fill_op = (_build_fill_value_i64(other, elem, arith, self.i64)
                   if dyn_fill else None)
        return axes, clamp, fill_op, fill

    def _clamp_operands(self, access_op, dims, lows, covers, mask_ops, low_ops):
        from mlir.dialects import arith
        axes, clamp = [], []
        mit, lit = iter(mask_ops), iter(low_ops)
        for axis, m in enumerate(dims):
            if covers[axis]:
                continue
            lo = lows[axis]
            low = next(lit) if lo == KDYN else self.const(lo)
            if m == KDYN or lo == KDYN:
                extent = next(mit) if m == KDYN else self.const(m)
                high = arith.AddIOp(low, extent).result
            else:
                high = self.const(lo + m)
            axes.append(axis)
            clamp += [low, high]
        return axes, clamp

    def _runtime_attrs(self, attrs, masked_axes, fill, dyn_stride_axes):
        from mlir.ir import IntegerAttr
        from .lib_ir import _i64_signed
        if masked_axes:
            attrs["masked_axes"] = self._i64_arr(masked_axes)
            attrs["masked_fill"] = IntegerAttr.get(self.i64, _i64_signed(fill))
        if dyn_stride_axes:
            attrs["dyn_stride_axes"] = self._i64_arr(dyn_stride_axes)

    def stride_attrs(self, dstride):
        """Split a dram_stride list into (static list, dynamic axes, ops)."""
        from mlir.dialects import arith
        axes = [i for i, s in enumerate(dstride) if not isinstance(s, int)]
        ops = [arith.IndexCastOp(self.i64, dstride[i]).result for i in axes]
        return [0 if not isinstance(s, int) else s for s in dstride], axes, ops

    def _flat_offset(self, offsets):
        from mlir.dialects import arith
        if all(isinstance(o, int) for o in offsets):
            return self.const(sum(offsets))
        acc = None
        for o in offsets:
            if isinstance(o, int):
                if o == 0:
                    continue
                term = self.const(o)
            else:
                term = o
            acc = term if acc is None else arith.AddIOp(acc, term).result
        return acc if acc is not None else self.const(0)

    def base_attrs(self, kind, dstride, arg_index):
        """THE CONTRACT: every transfer carries dma_kind, dram_arg and dram_stride -- the three its
        ODS requires -- and no hardware."""
        from mlir.ir import IntegerAttr, StringAttr
        return {
            "dma_kind": StringAttr.get(kind),
            "dram_arg": IntegerAttr.get(self.i64, arg_index),
            "dram_stride": self._i64_arr(dstride),
        }

    def transfer(self, dram, dram_idx, spad, attrs, index=None, masked=(),
                 fill=None, strides=(), offset_stride=None):
        """One transfer."""
        from mlir.ir import RankedTensorType, StringAttr
        yields = (isinstance(spad.type, RankedTensorType) and "dma_kind" in attrs
                  and is_load_kind(StringAttr(attrs["dma_kind"]).value))
        return TransferOp(
            spad.type if yields else None, dram, dram_idx, spad, self.const(0),
            self.const(0), list(masked), list(strides), index=index,
            dyn_fill=fill, dyn_offset_stride=offset_stride, **attrs).operation

    def note_staging(self, tile, spad, tx):
        """Record that `tx` filled `spad`, which `tile` reads -- so that _widen_index_to_tile can
        ASK which transfer staged an index instead of pattern-matching the IR back for it (rule
        9/14)."""
        if tile is not None:
            self.staged[tile] = (spad, tx)


def _empty_read_mask_carriers(module, arg_numels):
    from mlir.ir import Location
    from mlir.passmanager import PassManager

    loc = Location.unknown()
    with loc:
        changed = 0
        for func in funcs(module):
            for op in walk(func.operation):
                name = op.name
                if name == "tts.gather":
                    changed += _rewrite_gather(op, loc)
                elif name == "tts.load":
                    if operand_segments(op)[4:] and operand_segments(op)[4]:
                        _fold_the_argument_into_the_box(op, arg_numels)
                    changed += (_rewrite_elem_mask(op, loc)
                                or _rewrite_gs_mask(op, loc))
        if changed:
            PassManager.parse(
                "builtin.module(func.func(canonicalize,cse))").run(
                    module.operation)


def _rewrite_gather(op, loc):
    from mlir.ir import InsertionPoint
    seg = operand_segments(op)
    if not seg[GATHER_MASK]:
        return False
    mask = op.operands[GATHER_MASK]
    other = op.operands[GATHER_OTHER] if seg[GATHER_OTHER] else None
    res = op.results[0]
    with InsertionPoint(op):
        off = op.operands[GATHER_OFF]
        off = _build_select(mask, off,
                      splat(zero_of(elem_type(off), loc), off, loc), loc)
        g = gather(op.operands[GATHER_PTR], off, res.type, loc)
        if other is None:
            other = zero_of(elem_type(g), loc)
        new = _build_select(mask, g, splat(other, g, loc), loc)
    res.replace_all_uses_with(new)
    op.operation.erase()
    return True


def _fold_the_argument_into_the_box(load_op, arg_numels):
    from mlir.ir import DenseI64ArrayAttr
    ptr = _get_ptr_op(load_op)
    if ptr.name != "tts.make_tptr":
        return False
    sizes, strides, offsets, wrap = _get_descriptor(ptr)
    if any(not isinstance(w, int) for w in wrap):
        return False
    if any(_get_steps(ptr)):
        return False
    if any(not isinstance(x, int) for x in list(sizes) + list(offsets)):
        return False
    base = _get_base_offset(ptr)
    if base is not None and not isinstance(base, int):
        return False
    dstride = _get_dram_strides(sizes, strides)
    if any(not isinstance(d, int) or d < 0 for d in dstride):
        return False
    idx = _get_argument_index(ptr)
    numel = None if idx is None else arg_numels.get(idx)
    if numel is None:
        return False
    start = sum(offsets) + (base or 0)
    far = [max((min(n, w) if w else n) - 1, 0) * d
           for n, d, w in zip(sizes, dstride, wrap)]
    reach = sum(far)
    if start < 0 or start + reach < numel:
        return False
    cands = [a for a in range(len(sizes))
             if not wrap[a] and dstride[a] > 0 and sizes[a] > 1]
    if not cands:
        return False
    axis = max(cands, key=lambda a: dstride[a])
    room = numel - 1 - start
    if room < 0:
        return False
    fits = room // dstride[axis] + 1
    if fits >= sizes[axis]:
        return False
    dims = (list(int_array(load_op.attributes["static_mask_dims"]))
            if "static_mask_dims" in load_op.attributes else list(sizes))
    if len(dims) != len(sizes) or fits >= dims[axis]:
        return False
    dims[axis] = fits
    load_op.attributes["static_mask_dims"] = DenseI64ArrayAttr.get(dims)
    return True


def _get_argument_index(ptr_op):
    cast = op_of(getattr(ptr_op.operands[0], "owner", None))
    if cast is None or cast.name != "builtin.unrealized_conversion_cast":
        return None
    arg = _get_through_casts(cast.operands[0])
    blk = op_of(ptr_op).parent
    while blk is not None and name_of(blk) != "func.func":
        blk = op_of(blk).parent
    if blk is None:
        return None
    tensors = [a for a in blk.regions[0].blocks[0].arguments if str(a.type).startswith("memref<")]
    for i, a in enumerate(tensors):
        if a == arg:
            return i
    return None


def _rewrite_elem_mask(load_op, loc):
    from mlir.ir import DenseI32ArrayAttr, InsertionPoint, Operation
    lseg = operand_segments(load_op)
    if len(lseg) < 5 or not lseg[4]:
        return False
    ops = list(load_op.operands)
    _, _, other, mask = _split_access_operands(load_op)
    res = load_op.results[0]
    with InsertionPoint(load_op):
        bare = Operation.create(
            "tts.load", results=[res.type], operands=ops[:-1],
            attributes=dict(
                {"operandSegmentSizes": DenseI32ArrayAttr.get(
                    list(lseg[:4]) + [0]),
                 "static_mask_dims": load_op.attributes["static_mask_dims"]},
                **({"static_mask_lows": load_op.attributes["static_mask_lows"]}
                   if "static_mask_lows" in load_op.attributes else {})),
            loc=loc).results[0]
        if other is None:
            other = zero_of(elem_type(bare), loc)
        new = _build_select(mask, bare, splat(other, bare, loc), loc)
    res.replace_all_uses_with(new)
    load_op.operation.erase()
    return True


def _rewrite_gs_mask(load_op, loc):
    from mlir.ir import DenseI32ArrayAttr, DenseI64ArrayAttr, InsertionPoint, Operation
    gs = op_of(load_op.operands[0].owner)
    if name_of(gs) != "tts.make_gather_scatter_tptr":
        return False
    gseg = operand_segments(gs)
    if not gseg[GS_MASK_SEG]:
        return False
    mask = _get_operand_at_segment(gs, GS_MASK_SEG)
    dim = int_attr(gs, "gather_scatter_dim")
    sizes = int_array(gs.attributes["sizes"])
    res = load_op.results[0]

    lseg = operand_segments(load_op)
    other = list(load_op.operands)[-1] if lseg[3] else None

    with InsertionPoint(load_op):
        attrs = {n: gs.attributes[n] for n in _GS_ATTRS}
        mask_at = sum(gseg[:GS_MASK_SEG])
        attrs["operandSegmentSizes"] = DenseI32ArrayAttr.get(
            gseg[:GS_MASK_SEG] + [0] + gseg[GS_MASK_SEG + 1:])
        ops = list(gs.operands)
        idx_at = sum(gseg[:1])
        idx = ops[idx_at]
        ops[idx_at] = _build_select(mask, idx,
                              splat(zero_of(elem_type(idx), loc), idx, loc), loc)
        bare_ptr = Operation.create(
            "tts.make_gather_scatter_tptr", results=[gs.results[0].type],
            operands=ops[:mask_at] + ops[mask_at + 1:],
            attributes=attrs, loc=loc).results[0]
        dims = int_array(load_op.attributes["static_mask_dims"])
        if dims:
            dims[dim] = sizes[dim]
        attrs = {"operandSegmentSizes": DenseI32ArrayAttr.get(lseg),
                 "static_mask_dims": DenseI64ArrayAttr.get(dims)}
        if "static_mask_lows" in load_op.attributes:
            lows = int_array(load_op.attributes["static_mask_lows"])
            if lows:
                lows[dim] = 0
            attrs["static_mask_lows"] = DenseI64ArrayAttr.get(lows)
        bare = Operation.create(
            "tts.load", results=[res.type],
            operands=[bare_ptr] + list(load_op.operands)[1:],
            attributes=attrs, loc=loc).results[0]
        if other is None:
            other = zero_of(elem_type(bare), loc)
        new = _build_select(mask, bare, splat(other, bare, loc), loc, mask_axes=[dim])
    res.replace_all_uses_with(new)
    load_op.operation.erase()
    if not any(len(list(r.uses)) for r in gs.results):
        gs.erase()
    return True


def _rewrite_pointer_tiles(cx, func, loc):
    from mlir.ir import IntegerType
    i64 = IntegerType.get_signless(64)
    ops = walk(func.operation)
    _refuse_other_carriers(ops)
    for op in ops:
        if op.name == "linalg.fill":
            _rewrite_ptr_fill(op, i64, loc)
            continue
        if op.name != "linalg.generic":
            continue
        access = _get_pointer_access(op)
        body = _get_generic_body_op(op)
        if access is not None:
            _rewrite_pointer_access(cx, op, access, loc)
        elif not (_rewrite_tptr_ptradd(op, i64, loc)
                  or _rewrite_address_move(op, i64, loc)
                  or (body is not None and body.name in _PTR_IDENTITY
                      and _rewrite_ptr_identity(op, body))
                  or (body is not None and body.name in _ADDRESS_AGNOSTIC
                      and _has_pointers(op)
                      and _rewrite_address_agnostic(op, body, i64, loc))):
            _refuse_undissolved(op)


def _rewrite_ptr_fill(op, i64, loc):
    from mlir.ir import InsertionPoint, Operation, ShapedType
    from mlir.dialects import tensor
    if not (op.results and _is_pointer_tile(op.results[0])
            and _is_pointer(op.operands[0].type)):
        return False
    shape = list(ShapedType(op.results[0].type).shape)
    from mlir.ir import RankedTensorType
    with InsertionPoint(op):
        addr = Operation.create("tptr.ptrtoint", results=[i64],
                                operands=[op.operands[0]], loc=loc).results[0]
        tmpl = Operation.create("tensor.empty", results=[RankedTensorType.get(
            shape, i64, RankedTensorType(op.results[0].type).encoding)]).results[0]
        new = splat(addr, tmpl, loc)
    tmpl.owner.erase()
    op.results[0].replace_all_uses_with(new)
    op.operation.erase()
    return True


def _rewrite_pointer_access(cx, op, access, loc):
    from mlir.ir import InsertionPoint, ShapedType
    load = access.kind == "load"
    ty = ShapedType((op.results[0] if load else access.value).type)
    elem, sizes = ty.element_type, list(ty.shape)
    fill = None
    with InsertionPoint(op):
        if access.fill is not None:
            fill = _build_uniform_scalar(access.fill)
            if fill is None:
                raise FillVaries(
                    "a tensor tt.load whose `other=` varies per element is not "
                    "wired; the fill is a single value and splatting the wrong one "
                    "would be silent: %s" % op)
        idx = _build_pointer_index(access.addr, elem, sizes, op, loc)
        acc = _Access(op, None, "MVIN" if load else "MVOUT", sizes, elem,
                      cx.any_argument(), DRAM_ARG_POINTER_TILE)
        tx = _rewrite_indirect(
            cx, acc, idx, [] if access.mask is None else [(access.mask, 1)], [],
            [0] * len(sizes), value=None if load else access.value)
    if fill is None:
        return
    got = tx.results[0]
    with InsertionPoint.after(tx):
        sel = _build_select(access.mask, got, splat(fill, got, loc), loc)
    got.replace_all_uses_except(sel, op_of(sel.owner).operation)


def _rewrite_tptr_ptradd(op, i64, loc):
    from mlir.ir import InsertionPoint, Operation, ShapedType
    from mlir.dialects import arith
    from .lib_ir import widen
    ops = _get_block_ops(op.regions[0])
    if not ops or ops[-1].name != "linalg.yield":
        return False
    body, yielded = ops[:-1], ops[-1].operands[0]
    if not any(o.name == "tptr.ptradd" for o in body):
        return False
    ins = list(op.operands)[:operand_segments(op)[0]]
    old = list(op.regions[0].blocks[0].arguments)

    def rebuilt_chain(*args):
        seen = dict(zip(old, args))
        for o in body:
            if o.name == "tptr.ptradd":
                a = seen.get(o.operands[0], o.operands[0])
                b = seen.get(o.operands[1], o.operands[1])
                seen[o.results[0]] = arith.AddIOp(a, widen(b, i64)).result
            else:
                seen[o.results[0]] = Operation.create(
                    o.name, results=[o.results[0].type],
                    operands=[seen.get(x, x) for x in o.operands],
                    attributes=attrs_of(o),
                    loc=loc).results[0]
        return seen.get(yielded, yielded)

    with InsertionPoint(op):
        new = elementwise_generic(ins, i64, rebuilt_chain, loc,
                                  list(ShapedType(op.results[0].type).shape),
                                  like=op)
    op.results[0].replace_all_uses_with(new)
    op.operation.erase()
    return True


def _rewrite_address_move(op, i64, loc):
    moved = _build_address_move(op, i64, loc)
    if moved is None:
        return False
    op.results[0].replace_all_uses_with(moved)
    op.operation.erase()
    return True


def _rewrite_ptr_identity(op, body):
    if not (_is_pointer(body.operands[0].type)
            or _is_pointer(body.results[0].type)):
        return False
    if operand_segments(op)[0] != 1:
        raise _Fatal("a pointer cast generic with %d inputs is not the "
                     "one-in-one-out form this rewrites: %s"
                     % (operand_segments(op)[0], op))
    src = op.operands[0]
    if not _is_i64_tensor(src):
        raise _Fatal(
            "a pointer cast generic whose input is not an i64 tile: the "
            "chain feeding it was not dissolved, and replacing its uses "
            "would hand a pointer tensor downstream: %s" % op)
    op.results[0].replace_all_uses_with(src)
    op.operation.erase()
    return True


def _rewrite_address_agnostic(op, body, i64, loc):
    from mlir.ir import InsertionPoint, Operation, ShapedType
    ins = list(op.operands)[:operand_segments(op)[0]]
    old = list(op.regions[0].blocks[0].arguments)
    if any(_is_pointer(o.type) and o not in old for o in body.operands):
        raise _Fatal("an address-agnostic body reads a pointer that is not one "
                     "of its own arguments, which this cannot retype: %s" % op)

    def rebuilt_op(*args):
        seen = dict(zip(old, args))
        return Operation.create(
            body.name, results=[i64],
            operands=[seen.get(o, o) for o in body.operands],
            attributes=attrs_of(body),
            loc=loc).results[0]

    with InsertionPoint(op):
        new = elementwise_generic(ins, i64, rebuilt_op, loc,
                                  list(ShapedType(op.results[0].type).shape),
                                  like=op)
    op.results[0].replace_all_uses_with(new)
    op.operation.erase()
    return True


def _refuse_other_carriers(ops):
    for op in ops:
        if op.name in _PTR_CARRIERS:
            continue
        vals = (list(op.operands) + list(op.results)
                + [a for r in op.regions for b in r.blocks for a in b.arguments])
        if any(_is_pointer_tile(v) for v in vals):
            raise PointerTileUnconverted("a tile of pointers passes through `%s`, and this pass "
                         "rewrites only linalg.fill and linalg.generic, so it would "
                         "reach bufferize still typed over !ptr.ptr: %s" % (op.name, op))


def _refuse_undissolved(op):
    if not _has_pointers(op):
        return
    names = ", ".join(o.name for o in _get_block_ops(op.regions[0]))
    raise _Fatal("a linalg.generic over a tensor of pointers whose body is "
                 "`%s` has no arm in this pass, so it would reach bufferize "
                 "still typed over !ptr.ptr: %s" % (names, op))


def _get_pointer_access(op):
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return None
    blk = op.regions[0].blocks[0]
    tile = dict(zip(blk.arguments, op.operands))
    body = _get_block_ops(op.regions[0])
    ops, yielded = body[:-1], list(body[-1].operands)
    mask = fill = None
    if len(ops) == 1 and ops[0].name == "scf.if" and not ops[0].results:
        mask, ops = ops[0].operands[0], _get_block_ops(ops[0].regions[0])[:-1]
    elif (len(ops) == 2 and ops[1].name == "scf.if"
          and len(ops[1].results) == 1 and yielded == [ops[1].results[0]]):
        taken = _get_block_ops(ops[1].regions[0])
        other = _get_block_ops(ops[1].regions[1])
        if len(taken) != 2 or len(other) != 1:
            return None
        mask, fill = ops[1].operands[0], other[0].operands[0]
        ops, yielded = [ops[0], taken[0]], list(taken[1].operands)
    if len(ops) != 2 or ops[0].name != "tptr.to_memref":
        return None
    conv, acc = ops
    if conv.operands[0] not in tile or (mask is not None and mask not in tile):
        return None
    addr = tile[conv.operands[0]]
    mask = None if mask is None else tile[mask]
    fill = tile.get(fill, fill)
    if (acc.name == "memref.load" and acc.operands[0] == conv.results[0]
            and yielded == [acc.results[0]]):
        return _PointerAccess("load", addr, None, mask, fill)
    if (acc.name == "memref.store" and acc.operands[1] == conv.results[0]
            and not yielded and acc.operands[0] in tile):
        return _PointerAccess("store", addr, tile[acc.operands[0]], mask, None)
    return None


def _get_generic_body_op(gen):
    if len(gen.regions) != 1:
        return None
    ops = _get_block_ops(gen.regions[0])
    if len(ops) != 2 or ops[-1].name != "linalg.yield":
        return None
    return ops[0]


def _get_block_ops(region):
    if len(region.blocks) != 1:
        return []
    return [o.operation for o in region.blocks[0].operations]


def _build_pointer_index(addr, elem, sizes, op, loc):
    from mlir.dialects import arith
    from mlir.ir import IntegerAttr, IntegerType
    size = elem_bytes(elem)
    shift = size.bit_length() - 1
    if 1 << shift != size:
        raise _Fatal("element size %d is not a power of two, so an address cannot "
                     "be turned into an element index by a shift: %s" % (size, op))
    i64 = IntegerType.get_signless(64)

    def const(v):
        return splat(arith.ConstantOp(i64, IntegerAttr.get(i64, v)).result, addr, loc)
    return elementwise_generic(
        [addr, const(POINTER_TILE_BASE), const(shift)], i64,
        lambda a, b, s: arith.ShRSIOp(arith.SubIOp(a, b).result, s).result, loc, sizes)


def _build_uniform_scalar(v):
    from mlir.dialects import arith
    from mlir.ir import DenseElementsAttr, ShapedType
    if not is_shaped(v):
        return v
    op = producer(v)
    name = name_of(op)
    if name in ("linalg.fill", "tensor.splat", "tt.splat"):
        return op.operands[0]
    if name != "arith.constant":
        return None
    try:
        attr = DenseElementsAttr(op.attributes["value"])
    except (KeyError, ValueError, TypeError):
        return None
    if not attr.is_splat:
        return None
    return arith.ConstantOp(ShapedType(v.type).element_type,
                            attr.get_splat_value()).result


def _build_address_move(op, i64, loc):
    from mlir.ir import InsertionPoint, ShapedType
    if not op.results or not _is_pointer_tile(op.results[0]):
        return None
    res = op.results[0]
    if list(operand_segments(op)) != [1, 1]:
        return None
    blk = op.regions[0].blocks[0]
    yld = _get_block_ops(op.regions[0])[-1]
    if len(yld.operands) != 1 or yld.operands[0] != blk.arguments[0]:
        return None
    src = op.operands[0]
    with InsertionPoint(op):
        return elementwise_generic([src], i64, lambda a: a, loc,
                                   list(ShapedType(res.type).shape), like=op)


def _has_pointers(op):
    args = [a for r in op.regions for b in r.blocks for a in b.arguments]
    return (any(_is_pointer_tile(v) for v in list(op.operands) + list(op.results))
            or any(_is_pointer(a.type) for a in args))


def _is_pointer_tile(v):
    from mlir.ir import ShapedType
    return rank_of(v) is not None and _is_pointer(ShapedType(v.type).element_type)


def _is_pointer(ty):
    return is_type(ty, "ptr", "ptr")


def _is_i64_tensor(v):
    from mlir.ir import IntegerType, ShapedType
    if rank_of(v) is None:
        return False
    try:
        return IntegerType(ShapedType(v.type).element_type).width == 64
    except (TypeError, ValueError):
        return False


def _empty_write_mask_carriers(cx, func, loc):
    n = 0
    for op in [o for o in walk(func.operation) if name_of(o) == "tts.store"]:
        n += _rewrite_store_elem_mask(cx, op, loc)
    return n


def _rewrite_store_elem_mask(cx, store_op, loc):
    from mlir.ir import DenseI32ArrayAttr, InsertionPoint, Operation
    seg = operand_segments(store_op)
    if len(seg) < 5 or not seg[4]:
        return False
    ptr_op = op_of(store_op.operands[0].owner)
    if ptr_op is None or name_of(ptr_op) != "tts.make_tptr":
        return False
    if _get_flat_shape(ptr_op) is not None:
        return False
    sizes, strides, offsets, wrap = _get_descriptor(ptr_op)
    if not _is_tile_injective(sizes, strides, wrap):
        return False
    bounds = _get_tile_bounds(sizes, strides, offsets, wrap,
                          _get_base_offset(ptr_op))
    arg_index = cx.base_memref(ptr_op.operands[0])[1]
    if bounds is None or not (0 <= bounds[0]
                              and bounds[1] < cx.arg_numels[arg_index]):
        return False
    ops = list(store_op.operands)
    box = ops[seg[0] + seg[1]:seg[0] + seg[1] + seg[2] + seg[3]]
    keep = dict({"static_mask_dims": store_op.attributes["static_mask_dims"]},
                **({"static_mask_lows": store_op.attributes["static_mask_lows"]}
                   if "static_mask_lows" in store_op.attributes else {}))
    value, mask = ops[seg[0]], ops[-1]
    with InsertionPoint(store_op):
        held = Operation.create(
            "tts.load", results=[value.type], operands=ops[:seg[0]] + box,
            attributes=dict({"operandSegmentSizes": DenseI32ArrayAttr.get(
                [seg[0], seg[2], seg[3], 0, 0])}, **keep), loc=loc).results[0]
        Operation.create(
            "tts.store", results=[],
            operands=ops[:seg[0]] + [_build_select(mask, value, held, loc)] + box,
            attributes=dict({"operandSegmentSizes": DenseI32ArrayAttr.get(
                [seg[0], seg[1], seg[2], seg[3], 0])}, **keep), loc=loc)
    store_op.operation.erase()
    return True


def _is_tile_injective(sizes, strides, wrap):
    if any(not (isinstance(w, int) and w == 0) for w in wrap):
        return False
    spans = []
    for n, st in zip(sizes, strides):
        n, st = _get_int(n), _get_int(st)
        if n is None or st is None:
            return False
        if n == 1:
            continue
        if st == 0:
            return False
        spans.append((abs(st), (n - 1) * abs(st)))
    reach = 0
    for st, span in sorted(spans):
        if st <= reach:
            return False
        reach += span
    return True


def _get_tile_bounds(sizes, strides, offsets, wrap, base):
    lo = hi = 0
    if base is not None:
        o = _get_int(base)
        if o is None:
            return None
        lo += o
        hi += o
    for d in range(len(sizes)):
        w = wrap[d] if d < len(wrap) else 0
        if not (isinstance(w, int) and w == 0):
            W = _get_int(w)
            if W is None or W <= 0:
                return None
            hi += W - 1
            continue
        n, st, off = (_get_int(sizes[d]), _get_int(strides[d]),
                      _get_int(offsets[d]))
        if None in (n, st, off):
            return None
        lo += off
        hi += off
        span = (n - 1) * st
        hi += span if span >= 0 else 0
        lo += span if span < 0 else 0
    return lo, hi


def _get_ordered_accesses(func):
    loads, stores, full_gs = [], [], []
    for op in walk(func.operation):
        n = op.operation.name
        if n == "tts.load":
            loads.append(op)
        elif n == "tts.store":
            stores.append(op)
        elif n in ("tts.gather", "tts.scatter"):
            full_gs.append(op)
    via = [_get_ptr_op(o).name == _GATHER_PTR for o in loads]
    return ([o for o, g in zip(loads, via) if not g]
            + [o for o, g in zip(loads, via) if g]
            + full_gs + stores)


def _rewrite_access(cx, access_op):
    kind = "MVIN" if len(access_op.results) else "MVOUT"
    name = access_op.operation.name
    if name in ("tts.gather", "tts.scatter"):
        _rewrite_unstructured(cx, access_op, kind)
        return
    ptr = _get_ptr_op(access_op)
    if ptr.name == _GATHER_PTR:
        _rewrite_partial_unstructured(cx, access_op, ptr, kind)
        return
    _rewrite_affine(cx, access_op, ptr, kind)


def _rewrite_affine(cx, access_op, make_tptr_op, kind):
    from mlir.ir import InsertionPoint
    sizes, strides, offsets, wrap = _get_descriptor(make_tptr_op)
    step = _get_steps(make_tptr_op)
    wdims = {d for d, w in enumerate(wrap)
             if not (isinstance(w, int) and w == 0)}
    sdims = {d for d, k in enumerate(step) if k}
    flat = _get_flat_shape(make_tptr_op)
    base = _get_base_offset(make_tptr_op)
    _, _, _, elem_mask = _split_access_operands(access_op)
    runs = None
    dstride = _get_dram_strides(sizes, strides)
    for d in sdims:
        dstride[d] = 0
    if wdims or sdims or flat is not None or elem_mask is not None:
        if flat is None and elem_mask is None:
            runs = _get_cut_facts(access_op, sizes, strides, offsets,
                              wrap, wdims, step, sdims, dstride)
        if runs is None:
            return _rewrite_per_element(cx, access_op, make_tptr_op, kind, sizes,
                                     strides, offsets, wrap, wdims, flat,
                                     elem_mask, sdims)
    if base is not None:
        offsets = list(offsets) + [base]
    acc = _get_access(cx, access_op, kind, make_tptr_op.operands[0],
                  make_tptr_op, sizes)
    with InsertionPoint(access_op):
        _rewrite_into_transfer(cx, acc, cx._flat_offset(offsets), dstride,
                note=True, cut=runs)


def _get_cut_facts(access_op, sizes, strides, offsets, wrap, wdims,
               step, sdims, dstride):
    if _narrowed_at_runtime(access_op) & (wdims | sdims):
        return None
    facts, total = [[] for _ in sizes], 1
    for d in sorted(wdims | sdims):
        got = _get_axis_cut(sizes, strides, offsets, wrap, step, d, dstride[d],
                            d in wdims, d in sdims)
        if got is None:
            return None
        facts[d], count = got
        total *= count
    if total > MAX_WRAP_PIECES:
        return None
    if name_of(access_op) == "tts.store" and _revisits_a_row(sizes, facts, wdims):
        return None
    return facts


def _narrowed_at_runtime(access_op):
    """The axes a mask bounds by a runtime value: a piece's box is static, so a cut axis takes a
    static clamp only (p11 intersects it with each piece's box)."""
    if "static_mask_dims" not in access_op.attributes:
        return set()
    hi = int_array(access_op.attributes["static_mask_dims"])
    lo = (int_array(access_op.attributes["static_mask_lows"])
          if "static_mask_lows" in access_op.attributes else [0] * len(hi))
    return {d for d in range(len(hi)) if KDYN in (lo[d], hi[d])}


def _get_axis_cut(sizes, strides, offsets, wrap, step, d, dstride, wraps, steps):
    """Axis `d`'s digits and how many runs they make: the rest of the axis at its stride, wrapping
    at W from its offset if it wraps, over a K-row digit of stride 0 if it steps once every K rows."""
    n, st = _get_int(sizes[d]), _get_int(strides[d])
    K = step[d] if steps else 1
    if n is None or st is None or n < 1 or (steps and K <= 1):
        return None
    walk_at = st if steps else dstride
    outer, runs = Digit(0, walk_at, 0, 0), -(-n // K)
    if wraps:
        W, o = _get_int(wrap[d]), _get_int(offsets[d])
        if None in (W, o) or W <= 0:
            return None
        outer = Digit(0, walk_at, W, o)
        if not steps:
            runs = abs((o + (n - 1) * st) // W - o // W) + 1
    return cut_entry([outer] + ([Digit(K, 0, 0, 0)] if steps else [])), runs


def _revisits_a_row(sizes, facts, wdims):
    for d in sorted(wdims):
        n = _get_int(sizes[d])
        if n is None or n < 1:
            continue
        digits = cut_digits(facts[d])
        if len({cut_offset(digits, p) for p in range(n)}) != n:
            return True
    return False


def _rewrite_per_element(cx, access_op, make_tptr_op, kind, sizes, strides,
                      offsets, wrap, wdims, flat, elem_mask,
                      sdims=frozenset()):
    from mlir.ir import InsertionPoint
    rank = len(sizes)
    acc = _get_access(cx, access_op, kind, make_tptr_op.operands[0],
                  make_tptr_op, sizes)
    inside = (set(range(rank)) if (flat is not None or elem_mask is not None)
              else set(wdims) | set(sdims))
    dstride = _get_dram_strides(sizes, strides)
    for d in inside:
        dstride[d] = 0
    with InsertionPoint(access_op):
        idx_t = _build_address_tile(sizes, strides, offsets, wrap, wdims, flat,
                              inside, stepv=_get_steps(make_tptr_op), sdims=sdims,
                              encoding=_get_access_encoding(access_op))
        _rewrite_indirect(
            cx, acc,
            idx_t, [(elem_mask, 1)] if elem_mask is not None else [],
            _outside_offsets(offsets, wdims, flat,
                             _get_base_offset(make_tptr_op)), dstride)


def _build_address_tile(sizes, strides, offsets, wrap, wdims, flat, inside,
                  gathered=None, stepv=None, sdims=frozenset(), encoding=None):
    from mlir.ir import (AffineMap, AffineMapAttr, ArrayAttr, DenseI32ArrayAttr, IndexType, IntegerAttr,
                         IntegerType, InsertionPoint, Operation,
                         RankedTensorType)
    from mlir.dialects import arith, linalg, tensor
    i64 = IntegerType.get_signless(64)
    rank = len(sizes)

    def as_i64(v):
        if isinstance(v, int):
            return arith.ConstantOp(i64, IntegerAttr.get(i64, v)).result
        return arith.IndexCastOp(i64, v).result

    from mlir.ir import ShapedType
    tile_ty = RankedTensorType.get(sizes, i64, encoding)
    empty = Operation.create("tensor.empty", results=[tile_ty]).results[0]
    ident = AffineMapAttr.get(AffineMap.get_identity(rank))
    ins = [] if gathered is None else [gathered[0]]
    gen = Operation.create(
        "linalg.generic", results=[tile_ty], operands=ins + [empty],
        attributes={
            "indexing_maps": ArrayAttr.get([ident] * (len(ins) + 1)),
            "iterator_types": ArrayAttr.get(
                [parallel_iterator()] * rank),
            "operandSegmentSizes": DenseI32ArrayAttr.get([len(ins), 1])},
        regions=1)
    arg_tys = ([ShapedType(gathered[0].type).element_type] if ins else []) + [i64]
    blk = gen.regions[0].blocks.append(*arg_tys)
    with InsertionPoint(blk):
        acc = None
        if gathered is not None:
            idx = blk.arguments[0]
            if elem_bits(ShapedType(gathered[0].type).element_type) < 64:
                idx = arith.ExtSIOp(i64, idx).result
            acc = arith.MulIOp(idx, as_i64(gathered[1])).result
        for d in sorted(inside):
            idx = Operation.create(
                "linalg.index", results=[IndexType.get()], operands=[],
                attributes={"dim": IntegerAttr.get(i64, d)}).results[0]
            pos = arith.IndexCastOp(i64, idx).result
            if d in sdims:
                pos = arith.DivSIOp(pos, as_i64(stepv[d])).result
            step = arith.MulIOp(pos, as_i64(strides[d])).result
            if d in wdims or flat is not None:
                step = arith.AddIOp(as_i64(offsets[d]), step).result
            if d in wdims:
                step = arith.RemSIOp(step, as_i64(wrap[d])).result
            acc = step if acc is None else arith.AddIOp(acc, step).result
        if acc is None:
            acc = arith.ConstantOp(i64, IntegerAttr.get(i64, 0)).result
        if flat is not None:
            acc = arith.RemSIOp(acc, as_i64(flat)).result
        linalg.YieldOp([acc])
    return gen.results[0]


def _outside_offsets(offsets, wdims, flat, base):
    rest = [0 if (flat is not None or d in wdims) else o
            for d, o in enumerate(offsets)]
    return rest + ([base] if base is not None else [])


def _rewrite_partial_unstructured(cx, access_op, gs_op, kind="MVIN"):
    from mlir.ir import InsertionPoint, IntegerAttr, ShapedType
    sizes, offsets, strides, wrap, wdims, gs_mask, gs_base = _gather_ptr_says(
        gs_op)
    _, _, _, elem_mask = _split_access_operands(access_op)
    _refuse_read_mask_on_pointer(gs_op, gs_mask, kind)
    gather_dim = IntegerAttr(gs_op.attributes["gather_scatter_dim"]).value
    acc = _get_access(cx, access_op, kind, gs_op.operands[0], gs_op, sizes)
    idx_t = gs_op.operands[1]
    parked = False
    if gs_mask is not None:
        if not wdims:
            with InsertionPoint(access_op):
                idx_t = _park_masked_offsets(
                    cx, idx_t, gs_mask, acc.arg_index, acc.elem,
                    access_op, stride=strides[gather_dim], kind=kind)
            parked = True
        _open_box_over_gathered(access_op, sizes, gather_dim)
    ish = ShapedType(idx_t.type)
    widen_idx = None
    one_per_row = _is_one_entry_per_row(cx, idx_t, ish, sizes, gather_dim)
    if not one_per_row and list(ish.shape) != sizes:
        if list(ish.shape) != [sizes[gather_dim]]:
            raise _Fatal(
                f"gather index shape {list(ish.shape)} is neither the "
                f"tile {sizes} nor one value per gathered row: {gs_op}")
        wide = None if wdims else _widen_index_load(
            cx, idx_t, sizes, gather_dim, _get_access_encoding(access_op))
        if wide is not None:
            idx_t = wide
            ish = ShapedType(idx_t.type)
        else:
            widen_idx = True
    dstride = _get_dram_strides(sizes, strides)
    dstride[gather_dim] = 0
    ostride = strides[gather_dim]
    flat = list(offsets)
    for d in wdims:
        dstride[d] = 0
        flat[d] = 0
    if gs_base is not None:
        flat.append(gs_base)
    with InsertionPoint(access_op):
        enc = _get_access_encoding(access_op)
        if widen_idx:
            idx_t = _broadcast_to_tile(idx_t, sizes, gather_dim, enc)
        if wdims:
            idx_t = _fold_wrap_into_index(sizes, strides, offsets, wrap,
                                          wdims, idx_t, ostride, enc)
            ostride = 1
        masks = _get_remaining_masks(gs_mask, elem_mask, sizes, gather_dim, wdims,
                                 ostride, enc)
        _rewrite_indirect(cx, acc, idx_t, masks, flat, dstride,
                       ostride=ostride, parked=parked,
                       gather_dim=gather_dim if one_per_row else None)


def _gather_ptr_says(gs_op):
    a = gs_op.attributes
    seg = int_array(a["operandSegmentSizes"])
    ops = list(gs_op.operands)
    p = 2
    stride_ops = ops[p:p + seg[GS_STRIDE_SEG]]; p += seg[GS_STRIDE_SEG]
    offset_ops = ops[p:p + seg[GS_OFFSET_SEG]]; p += seg[GS_OFFSET_SEG]
    shape_ops = ops[p:p + seg[GS_SHAPE_SEG]]
    strides = _resolve_strides(int_array(a["static_strides"]), stride_ops)
    offsets = _resolve_strides(int_array(a["static_offsets"]), offset_ops)
    wrap = _resolve_strides(int_array(a["static_shape"]), shape_ops)
    return (int_array(a["sizes"]), offsets, strides, wrap,
            [d for d, w in enumerate(wrap)
             if not (isinstance(w, int) and w == 0)],
            _get_operand_at_segment(gs_op, GS_MASK_SEG),
            _get_operand_at_segment(gs_op, GS_BASE_SEG))


def _refuse_read_mask_on_pointer(gs_op, gs_mask, kind):
    if gs_mask is not None and is_load_kind(kind):
        raise _Fatal(f"a read's mask reached the emitters on the POINTER, so "
                     f"_empty_read_mask_carriers left that carrier full: "
                     f"{gs_op}")


def _is_one_entry_per_row(cx, idx_t, ish, sizes, gather_dim):
    key = (idx_t, tuple(sizes), gather_dim)
    if key not in cx.flat_index:
        cx.flat_index[key] = (list(ish.shape) == [sizes[gather_dim]]
                              and _is_flat_index_ok(idx_t, sizes, gather_dim))
    return cx.flat_index[key]


def _is_flat_index_ok(idx_t, sizes, gather_dim):
    from mlir.ir import IntegerAttr
    for use in idx_t.uses:
        gs = op_of(use.owner)
        if name_of(gs) != "tts.make_gather_scatter_tptr":
            continue
        if int_array(gs.attributes["sizes"]) != sizes:
            return False
        if IntegerAttr(gs.attributes["gather_scatter_dim"]).value != gather_dim:
            return False
        if "static_shape" not in gs.attributes:
            return False
        if any(w != 0 for w in int_array(gs.attributes["static_shape"])):
            return False
        for r in gs.results:
            for u in r.uses:
                acc = op_of(u.owner)
                if name_of(acc) not in ("tts.load", "tts.store"):
                    return False
                if _split_access_operands(acc)[3] is not None:
                    return False
    return True


def _widen_index_load(cx, idx_t, sizes, gather_dim, enc):
    key = (idx_t, tuple(sizes), gather_dim)
    if key not in cx.wide_index:
        cx.wide_index[key] = _widen_index_to_tile(cx, idx_t, sizes, gather_dim, enc)
    return cx.wide_index[key]


def _widen_index_to_tile(cx, idx_t, sizes, gather_dim, enc):
    from mlir.ir import ShapedType

    src = op_of(idx_t.owner)
    elem = ShapedType(idx_t.type).element_type
    if idx_t in cx.staged:
        return _widen_staged_index(cx, src, idx_t, sizes, gather_dim, elem, enc)
    return None


def _has_axis_facts(tx):
    """Whether `tx` states a per-axis fact beyond the dram_stride every transfer carries."""
    for name in PER_AXIS:
        if name == "dram_stride" or name not in tx.attributes:
            continue
        try:
            if len(tx.attributes[name]):
                return True
        except TypeError:
            return True
    return False


def _widen_staged_index(cx, src, idx_t, sizes, gather_dim, elem, enc):
    from mlir.ir import InsertionPoint, RankedTensorType
    init, tx = cx.staged[idx_t]
    alloc = op_of(init.owner)
    ds = tx.attributes["dram_stride"]
    if len(ds) != 1 or _has_axis_facts(tx):
        return None
    strd = [0] * len(sizes)
    strd[gather_dim] = ds[0].value
    wide_t = RankedTensorType.get(sizes, elem, enc)
    if (len(list(init.uses)) == 1
            and _can_widen_in_place(idx_t, sizes, gather_dim)):
        tx.attributes["dram_stride"] = cx._i64_arr(strd)
        init.set_type(wide_t)
        idx_t.set_type(wide_t)
        return idx_t
    with InsertionPoint(src):
        a2 = alloc.clone()
        a2.results[0].set_type(wide_t)
        tx2 = tx.clone()
        tx2.operands[operand_number(tx2, "sram")] = a2.results[0]
        tx2.results[0].set_type(wide_t)
        tx2.attributes["dram_stride"] = cx._i64_arr(strd)
    wide = tx2.results[0]
    cx.note_staging(wide, a2.results[0], tx2)
    return wide


def _can_widen_in_place(idx_t, sizes, gather_dim):
    from mlir.ir import IntegerAttr
    for use in idx_t.uses:
        o = op_of(use.owner)
        if name_of(o) != "tts.make_gather_scatter_tptr":
            return False
        if int_array(o.attributes["sizes"]) != sizes:
            return False
        if IntegerAttr(o.attributes["gather_scatter_dim"]).value != gather_dim:
            return False
    return True


def _broadcast_to_tile(v, sizes, gather_dim, encoding=None):
    from mlir.dialects import linalg
    from mlir.ir import InsertionPoint, Operation, RankedTensorType, ShapedType
    elem = ShapedType(v.type).element_type
    ty = RankedTensorType.get(sizes, elem, encoding)
    empty = Operation.create("tensor.empty", results=[ty]).results[0]
    bop = linalg.BroadcastOp(
        result=[ty], input=v, init=empty,
        dimensions=[d for d in range(len(sizes)) if d != gather_dim])
    blk = bop.regions[0].blocks.append(elem, elem)
    with InsertionPoint(blk):
        linalg.YieldOp([blk.arguments[0]])
    return bop.results[0]


def _open_box_over_gathered(access_op, sizes, gather_dim):
    from mlir.ir import DenseI64ArrayAttr
    if "static_mask_dims" not in access_op.attributes:
        return
    dims = int_array(access_op.attributes["static_mask_dims"])
    if not dims:
        return
    dims[gather_dim] = sizes[gather_dim]
    access_op.attributes["static_mask_dims"] = DenseI64ArrayAttr.get(dims)
    if "static_mask_lows" in access_op.attributes:
        lo = int_array(access_op.attributes["static_mask_lows"])
        lo[gather_dim] = 0
        access_op.attributes["static_mask_lows"] = DenseI64ArrayAttr.get(lo)


def _fold_wrap_into_index(sizes, strides, offsets, wrap, wdims, idx_t,
                          scale, encoding=None):
    return _build_address_tile(sizes, strides, offsets, wrap, wdims, None,
                         set(wdims), gathered=(idx_t, scale), encoding=encoding)


def _get_remaining_masks(gs_mask, elem_mask, sizes, gather_dim, wdims, ostride,
                         encoding=None):
    masks = []
    if gs_mask is not None and wdims:
        masks.append((_broadcast_to_tile(gs_mask, sizes, gather_dim, encoding), 1))
    if elem_mask is not None:
        masks.append((elem_mask, ostride))
    return masks


def _rewrite_unstructured(cx, access_op, kind):
    from mlir.ir import InsertionPoint, ShapedType
    is_gather = is_load_kind(kind)
    if rank_of(access_op.operands[1]) is None:
        return _rewrite_scalar(cx, access_op, is_gather)
    if is_gather:
        seg = int_array(access_op.attributes["operandSegmentSizes"])
        if seg[2] or seg[3]:
            raise _Fatal(f"a read's mask reached the emitters, so "
                         f"_empty_read_mask_carriers left a tts.gather's own "
                         f"carrier full: {access_op}")
    scatter_mask = (None if is_gather or len(access_op.operands) <= 3
                    else access_op.operands[3])
    base = access_op.operands[0]
    if is_gather:
        offsets_t = access_op.operands[1]
        tile_ty = ShapedType(access_op.results[0].type)
    else:
        offsets_t = access_op.operands[1]
        value_t = access_op.operands[2]
        tile_ty = ShapedType(value_t.type)
    sizes = list(tile_ty.shape)
    rank = len(sizes)
    acc = _get_access(cx, access_op, kind, base, None, sizes,
                  None if is_gather else value_t)
    with InsertionPoint(access_op):
        _rewrite_indirect(
            cx, acc, offsets_t, [(scatter_mask, 1)] if scatter_mask is not None else [],
            [], [0] * rank, value=None if is_gather else value_t)


def _rewrite_scalar(cx, access_op, is_gather):
    """A scalar read or write is a one-element transfer through the scratchpad, as every DRAM access is."""
    from mlir.ir import InsertionPoint, IndexType, RankedTensorType
    from mlir.dialects import arith, bufferization, tensor
    if is_gather:
        seg = int_array(access_op.attributes["operandSegmentSizes"])
        if seg[2] or seg[3]:
            raise _Fatal(f"a read's mask reached the emitters, so "
                         f"_empty_read_mask_carriers left a scalar "
                         f"tts.gather's own carrier full: {access_op}")
    mask = (None if is_gather or len(access_op.operands) <= 3
            else access_op.operands[3])

    dram_arg, arg_index = cx.base_memref(access_op.operands[0])
    from mlir.ir import ShapedType
    want = (access_op.results[0].type if is_gather
            else access_op.operands[2].type)
    arg_elem = ShapedType(dram_arg.type).element_type
    if elem_bytes(want) != elem_bytes(arg_elem):
        raise _Fatal(f"scalar access wants a {want} out of a {arg_elem} buffer "
                     f"and a memref.load has no element size to scale: "
                     f"{access_op}")
    dram = cx.flat_view(dram_arg, arg_index)
    off = access_op.operands[1]
    scalar = RankedTensorType.get([], want)

    def read():
        spad = bufferization.AllocTensorOp(scalar, []).result
        tx = cx.transfer(dram, idx, spad, cx.base_attrs("MVIN", [1], arg_index))
        return tensor.ExtractOp(tx.results[0], []).result

    with InsertionPoint(access_op):
        idx = (off if isinstance(off.type, IndexType)
               else arith.IndexCastOp(cx.idx, off).result)
        if is_gather:
            got = read()
        else:
            # A write mask bounds the one element, [0, mask ? 1 : 0), as a tile's mask bounds its
            # axes: masked off, the transfer moves nothing -- no read of the element, no write past
            # the buffer -- and the TOG still sees one transfer, as it has no branch.
            spad = tensor.InsertOp(access_op.operands[2],
                                   bufferization.AllocTensorOp(scalar, []).result, []).result
            attrs, masked = cx.base_attrs("MVOUT", [1], arg_index), ()
            if mask is not None:
                one = arith.SelectOp(mask, cx.const(1), cx.const(0)).result
                masked = (cx.const(0), one)
                cx._runtime_attrs(attrs, [0], 0, [])
            cx.transfer(dram, idx, spad, attrs, masked=masked)
    if is_gather:
        access_op.results[0].replace_all_uses_with(got)
    access_op.operation.erase()


def _rewrite_indirect(cx, acc, idx_t, masks, rest, dstride, ostride=1,
                   value=None, gather_dim=None, parked=False):
    for mask, park_stride in masks:
        idx_t = _park_masked_offsets(cx, idx_t, mask, acc.arg_index, acc.elem,
                                     acc.op, stride=park_stride, kind=acc.kind)
        parked = True
    return _rewrite_into_transfer(cx, acc, cx._flat_offset(rest), dstride,
            index=_build_index_buffer(cx, idx_t),
            ostride=ostride, value=value,
            gather_dim=None if masks else gather_dim,
            dram_bound=parked and (not is_load_kind(acc.kind)
                                   or acc.arg_index == DRAM_ARG_POINTER_TILE))


def _park_masked_offsets(cx, offsets_t, mask, arg_index, tile_elem, access_op,
                         stride=1, kind="MVOUT"):
    from mlir.ir import (AffineMap, AffineMapAttr, ArrayAttr, DenseI32ArrayAttr, Operation, RankedTensorType,
                         ShapedType)
    from mlir.dialects import arith, linalg, tensor

    park = (0 if is_load_kind(kind) and arg_index != DRAM_ARG_POINTER_TILE
            else _park_offset(cx, arg_index, tile_elem, access_op, stride))
    st = ShapedType(offsets_t.type)
    shape, elem = list(st.shape), st.element_type
    if not (-2 ** 31 <= park < 2 ** 31) and elem_bits(elem) <= 32:
        raise _Fatal(f"parking offset {park} does not fit the index element "
                     f"{elem}: {access_op}")
    rank = len(shape)
    ident = AffineMapAttr.get(AffineMap.get_identity(rank))
    par = parallel_iterator()

    cst = arith.ConstantOp(elem, park).result
    parked = splat(cst, offsets_t, access_op.location)
    parked_ty = RankedTensorType.get(shape, elem, RankedTensorType(offsets_t.type).encoding)
    empty2 = Operation.create("tensor.empty", results=[parked_ty]).results[0]
    gen = Operation.create(
        "linalg.generic", results=[parked_ty],
        operands=[mask, offsets_t, parked, empty2],
        attributes={"indexing_maps": ArrayAttr.get([ident] * 4),
                    "iterator_types": ArrayAttr.get([par] * rank),
                    "operandSegmentSizes": DenseI32ArrayAttr.get([3, 1])},
        regions=1)
    i1 = ShapedType(mask.type).element_type
    blk = gen.regions[0].blocks.append(i1, elem, elem, elem)
    from mlir.ir import InsertionPoint as _IP
    with _IP(blk):
        linalg.YieldOp([arith.SelectOp(blk.arguments[0], blk.arguments[1],
                                       blk.arguments[2]).result])
    return gen.results[0]


def _park_offset(cx, arg_index, elem, access_op, stride=1):
    from mlir.ir import ShapedType
    if arg_index == DRAM_ARG_POINTER_TILE:
        return -(1 << 60)
    arg_elem = ShapedType(cx.tensor_args[arg_index].type).element_type
    total = cx.arg_numels[arg_index] * elem_bytes(arg_elem)
    park = (((total + 63) // 64) * 64) // elem_bytes(elem)
    if stride != 1:
        if not isinstance(stride, int) or stride < 1:
            raise _Fatal(f"cannot park a masked scatter whose gathered stride "
                         f"{stride} is not a positive constant: {access_op}")
        park = -(-park // stride)
    return park


def _build_index_buffer(cx, idx_t):
    from mlir.ir import IntegerType, ShapedType
    from mlir.dialects import bufferization
    ish = ShapedType(idx_t.type)
    elem = ish.element_type
    if not isinstance(elem, IntegerType):
        raise _Fatal(f"an indirect transfer's index tile is {elem}, which "
                     f"is not an integer, so the DMA's index field cannot "
                     f"describe it: {idx_t}")
    if elem_bytes(elem) not in (4, 8):
        raise _Fatal(
            f"an indirect transfer's index tile is {elem}, which the DMA "
            f"descriptor's indirect_esize cannot name -- it dispatches on "
            f"1, 2, 4 and 8 bytes and asserts otherwise. "
            f"tts.make_gather_scatter_tptr is supposed to have refused this "
            f"upstream, so its ODS has changed: {idx_t}")
    src = _already_a_tile(idx_t)
    if src is not None and src not in cx.claimed_index:
        cx.claimed_index.add(src)
        return src
    dest = bufferization.AllocTensorOp(idx_t.type, [], copy=idx_t).result
    cx.claimed_index.add(dest)
    return dest


def _already_a_tile(idx_t):
    o = producer(idx_t)
    if (name_of(o) != TransferOp.OPERATION_NAME
            or not is_load(o)):
        return None
    return idx_t


def _rewrite_into_transfer(cx, acc, dram_idx, dstride, index=None, ostride=None, value=None,
            note=False, gather_dim=None, cut=None, dram_bound=False):
    from mlir.ir import IntegerAttr, IntegerType
    from mlir.dialects import arith
    i64 = IntegerType.get_signless(64)
    if name_of(acc.op) in BOXED_ACCESS:
        masked_axes, clamp, fill_op, fill = cx.mask_and_fill(
            acc.op, acc.sizes, acc.elem)
    else:
        masked_axes, clamp, fill_op, fill = [], [], None, 0
    dstride, dyn_stride_axes, dyn_stride_ops = cx.stride_attrs(dstride)
    spad = _build_tile_buffer(acc.op, acc.kind, acc.sizes, acc.elem, value)
    attrs = cx.base_attrs(acc.kind, dstride, acc.arg_index)
    ostride_op = None
    if index is not None:
        if not isinstance(ostride, int):
            attrs["offset_stride"] = IntegerAttr.get(i64, 0)
            ostride_op = arith.IndexCastOp(i64, ostride).result
        else:
            if not 0 <= ostride <= 0xFFFF:
                raise _Fatal(f"offset_stride {ostride} does not fit the "
                             f"descriptor's uint16 field: {acc.op}")
            attrs["offset_stride"] = IntegerAttr.get(i64, ostride)
        if gather_dim is not None:
            attrs["indirect_dim"] = IntegerAttr.get(i64, gather_dim)
    cx._runtime_attrs(attrs, masked_axes, fill, dyn_stride_axes)
    tx = cx.transfer(acc.dram, dram_idx, spad, attrs, index=index, masked=clamp,
                     fill=fill_op, strides=dyn_stride_ops,
                     offset_stride=ostride_op)
    if dram_bound:
        tx.opview.dram_bound = True
    if cut:
        from mlir.ir import ArrayAttr, DenseI64ArrayAttr
        tx.opview.cut = ArrayAttr.get([DenseI64ArrayAttr.get(f) for f in cut])
    retired = _retire_access(acc.op, acc.kind, tx, acc.ptr)
    if note:
        cx.note_staging(retired, spad, tx)
    return tx


def _build_tile_buffer(access_op, kind, sizes, elem, value=None):
    from mlir.ir import RankedTensorType
    from mlir.dialects import bufferization
    if is_load_kind(kind):
        loaded = RankedTensorType(access_op.results[0].type)
        if list(loaded.shape) != list(sizes):
            raise _Fatal(f"an access whose result is not its tile {sizes}: {access_op}")
        return bufferization.AllocTensorOp(
            RankedTensorType.get(sizes, elem, loaded.encoding), []).result
    return access_op.operands[1] if value is None else value


def _retire_access(access_op, kind, tx, ptr_op=None):
    t = None
    if is_load_kind(kind):
        t = tx.results[0]
        access_op.results[0].replace_all_uses_with(t)
    access_op.operation.erase()
    if ptr_op is not None:
        _erase_ptr_if_dead(ptr_op)
    return t


def _erase_ptr_if_dead(ptr_op):
    if any(len(list(r.uses)) for r in ptr_op.results):
        return
    ptr_op.operation.erase()


def _get_access(cx, access_op, kind, base, ptr_op, sizes, value=None):
    dram_arg, arg_index = cx.base_memref(base)
    return _Access(access_op, ptr_op, kind, sizes,
                   _get_tile_elem(access_op, kind, dram_arg, value),
                   cx.flat_view(dram_arg, arg_index), arg_index)


def _get_tile_elem(access_op, kind, dram_arg, value=None):
    from mlir.ir import ShapedType
    if is_load_kind(kind):
        v = access_op.results[0]
    else:
        v = access_op.operands[1] if value is None else value
    return ShapedType(v.type).element_type


def _lower_ptr_to_int(cx, func):
    from mlir.ir import InsertionPoint, IntegerType
    from mlir.dialects import arith, memref
    i64 = IntegerType.get_signless(64)
    n = 0
    for op in walk(func.operation):
        if op.name not in ("tt.ptr_to_int", "tptr.ptrtoint"):
            continue
        if rank_of(op.results[0]) is not None:
            raise _Fatal(f"a TENSOR tt.ptr_to_int outlived _rewrite_pointer_tiles, "
                         f"which dissolves those first: {op}")
        dram_arg, arg_index = cx.base_memref(op.operands[0])
        dram = cx.flat_view(dram_arg, arg_index)
        with InsertionPoint(op):
            base = memref.ExtractAlignedPointerAsIndexOp(dram).result
            addr = arith.IndexCastOp(i64, base).result
        op.results[0].replace_all_uses_with(addr)
        op.operation.erase()
        n += 1
    return n


def _drop_dead_ptr_casts(func):
    from mlir.ir import InsertionPoint, Operation
    for op in [o for o in walk(func.operation)
               if name_of(o) == "builtin.unrealized_conversion_cast"
               and o.results and holds_type(o.results[0], "ptr", "ptr")]:
        up = op_of(getattr(op.operands[0], "owner", None))
        if (up is None
                or name_of(up) != "builtin.unrealized_conversion_cast"
                or not holds_type(op.operands[0], "tt", "ptr")):
            continue
        with InsertionPoint(op):
            new = Operation.create(
                "builtin.unrealized_conversion_cast",
                results=[op.results[0].type], operands=[up.operands[0]],
                loc=op.location).results[0]
        op.results[0].replace_all_uses_with(new)
        op.operation.erase()
    for op in [o for o in walk(func.operation)
               if name_of(o) == "builtin.unrealized_conversion_cast"
               and o.results and holds_type(o.results[0], "tt", "ptr")]:
        if not any(True for _ in op.results[0].uses):
            op.operation.erase()


def _get_ptr_op(op):
    return op_of(op.operands[0].owner)


def _get_descriptor(make_tptr_op):
    a = make_tptr_op.attributes
    sizes = int_array(a["sizes"])
    strides = int_array(a["static_strides"])
    static_offsets = int_array(a["static_offsets"])
    shape = int_array(a["static_shape"])
    seg = int_array(a["operandSegmentSizes"])
    ops = list(make_tptr_op.operands)
    p = 1
    stride_ops = ops[p:p + seg[1]]; p += seg[1]
    offset_ops = ops[p:p + seg[2]]; p += seg[2]
    shape_ops = ops[p:p + seg[3]]; p += seg[3]
    wrap = _resolve_strides(shape, shape_ops)
    resolved = _resolve_strides(strides, stride_ops)
    odyn = iter(offset_ops)
    offsets = [next(odyn) if o == KDYN else o for o in static_offsets]
    return sizes, resolved, offsets, wrap


def _get_steps(make_tptr_op):
    a = make_tptr_op.attributes
    n = len(int_array(a["sizes"]))
    return int_array(a["static_step"]) if "static_step" in a else [0] * n


def _get_base_offset(make_tptr_op):
    return _get_operand_at_segment(make_tptr_op, _TPTR_BASE_OFFSET_SEG)


def _get_flat_shape(make_tptr_op):
    return _get_operand_at_segment(make_tptr_op, _TPTR_FLAT_SHAPE_SEG)


def _get_operand_at_segment(op, seg):
    segs = operand_segments(op)
    if seg >= len(segs) or not segs[seg]:
        return None
    return list(op.operands)[sum(segs[:seg])]


def _split_access_operands(access_op):
    ops = list(access_op.operands)
    seg = int_array(access_op.attributes["operandSegmentSizes"])
    if name_of(access_op) == "tts.load":
        i = seg[0]
        dims = ops[i:i + seg[1]]; i += seg[1]
        lows = ops[i:i + seg[2]]; i += seg[2]
        other = ops[i] if seg[3] else None; i += seg[3]
        return dims, lows, other, (ops[i] if seg[4] else None)
    i = seg[0] + seg[1]
    dims = ops[i:i + seg[2]]; i += seg[2]
    lows = ops[i:i + seg[3]]; i += seg[3]
    return dims, lows, None, (ops[i] if seg[4] else None)


def _get_mask_box(access_op, sizes):
    dims = (int_array(access_op.attributes["static_mask_dims"])
            if "static_mask_dims" in access_op.attributes else [])
    lows = (int_array(access_op.attributes["static_mask_lows"])
            if "static_mask_lows" in access_op.attributes
            else [0] * len(dims))
    if len(lows) != len(dims):
        raise _Fatal(f"static_mask_lows has {len(lows)} entries for "
                     f"{len(dims)} extents: {access_op}")
    mask_ops, low_ops, other, _ = _split_access_operands(access_op)
    n_dyn = sum(1 for m in dims if m == KDYN)
    if len(mask_ops) != n_dyn:
        raise _Fatal(f"{len(mask_ops)} mask operands but static_mask_dims "
                     f"asks for {n_dyn}: {access_op}")
    n_low = sum(1 for m in lows if m == KDYN)
    if len(low_ops) != n_low:
        raise _Fatal(f"{len(low_ops)} mask_lows operands but static_mask_lows "
                     f"asks for {n_low}: {access_op}")
    if dims and len(dims) != len(sizes):
        raise _Fatal(f"static_mask_dims has {len(dims)} entries for a "
                     f"rank-{len(sizes)} tile: {access_op}")
    return dims, lows, mask_ops, low_ops, other


def _covered_axes(dims, lows, sizes):
    return [lo == 0 and m != KDYN and m >= s
            for lo, m, s in zip(lows, dims, sizes)]


def _get_fill_bits(other, elem_type):
    import struct
    from mlir.ir import FloatAttr, FloatType, IntegerAttr, IntegerType
    owner = getattr(other, "owner", None)
    owner = op_of(owner)
    if name_of(owner) != "arith.constant":
        return None
    attr = owner.attributes["value"]
    if isinstance(elem_type, IntegerType):
        width = IntegerType(elem_type).width
        return IntegerAttr(attr).value & ((1 << width) - 1)
    if not isinstance(elem_type, FloatType):
        raise _Fatal(f"`other=` of non-numeric type {elem_type} not supported")
    v = FloatAttr(attr).value
    if str(elem_type) == "bf16":
        return int.from_bytes(struct.pack("<f", v), "little") >> 16
    fmt = {16: "<e", 32: "<f", 64: "<d"}.get(FloatType(elem_type).width)
    if fmt is None:
        raise _Fatal(f"`other=` on element type {elem_type} not supported "
                     f"(no host encoding for its bit pattern)")
    return int.from_bytes(struct.pack(fmt, v), "little")


def _build_fill_value_i64(other, elem_type, arith, i64):
    from mlir.ir import IntegerType, FloatType
    if isinstance(elem_type, IntegerType):
        width, v = IntegerType(elem_type).width, other
    elif isinstance(elem_type, FloatType):
        width = FloatType(elem_type).width
        v = arith.BitcastOp(IntegerType.get_signless(width), other).result
    else:
        raise _Fatal(f"runtime `other=` of non-numeric type {elem_type}")
    return arith.ExtUIOp(i64, v).result if width < 64 else v


def _get_dram_strides(sizes, strides):
    out, acc = [], 1
    for size, stride in zip(reversed(sizes), reversed(strides)):
        if size == 1 and isinstance(stride, int) and stride == 0:
            out.append(acc)
        else:
            out.append(stride)
        acc *= size
    out.reverse()
    return out


def _resolve_strides(static_strides, stride_ops):
    it = iter(stride_ops)
    out = []
    for s in static_strides:
        if s != KDYN:
            out.append(s)
            continue
        v = next(it)
        c = _get_int(v)
        out.append(v if c is None else c)
    return out


def _get_int(v):
    from mlir.ir import IntegerAttr
    if isinstance(v, int):
        return v
    owner = op_of(getattr(v, "owner", None))
    if name_of(owner) != "arith.constant":
        return None
    return IntegerAttr(owner.attributes["value"]).value


def _get_through_casts(value):
    while not str(value.type).startswith("memref<"):
        up = op_of(getattr(value, "owner", None))
        if up is None or up.name != "builtin.unrealized_conversion_cast":
            break
        value = up.operands[0]
    return value


def _build_select(mask, a, b, loc, mask_axes=None):
    from mlir.ir import AffineExpr, AffineMap, AffineMapAttr, ShapedType
    from mlir.dialects import arith

    if not is_shaped(a):
        return arith.SelectOp(mask, a, b, loc=loc).result
    st = ShapedType(a.type)
    shape, elem = list(st.shape), st.element_type
    ident = AffineMapAttr.get(AffineMap.get_identity(len(shape)))
    mask_map = ident if mask_axes is None else AffineMapAttr.get(AffineMap.get(
        len(shape), 0, [AffineExpr.get_dim(d) for d in mask_axes]))
    return elementwise_generic(
        [mask, a, b], elem,
        lambda m, x, y: arith.SelectOp(m, x, y).result,
        loc, shape, maps=[mask_map] + [ident] * 3)


def _get_access_encoding(access_op):
    from mlir.ir import RankedTensorType
    tile = access_op.results[0] if len(access_op.results) else access_op.operands[1]
    return RankedTensorType(tile.type).encoding if isinstance(tile.type, RankedTensorType) else None
