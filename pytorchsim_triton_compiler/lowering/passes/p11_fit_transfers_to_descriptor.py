"""pytorchsim-fit-transfers-to-descriptor: a transfer of more axes than the provider's DESCRIPTOR_DIMS
is folded, peeled into affine.for loops, or both, whichever issues fewest DMAs; a cut transfer, so
fitted, becomes one transfer per run of its cut axes, moving its `piece_box` from that run's DRAM."""

import itertools
import math
from collections import namedtuple

from ...contract.stop import Reason, Refusal

from ...dialects.torchsim import (TransferOp, cut_offset, is_load, moved_shape, operand_number,
                                  renumber_per_axis, renumbered_per_axis, restated)
from .lib_descriptor import TransferFacts
from .lib_ir import int_array, strides_and_offset, walk

MARKERS = (TransferOp.OPERATION_NAME,)

Fit = namedtuple("Fit", "peeled groups dmas")


def run(module, *, descriptor_dims, **_):
    """Fit every transfer of `module` whose tile has more axes than the descriptor has dims, and
    split every cut transfer into its pieces."""
    for op in [o for o in walk(module.operation) if o.name == TransferOp.OPERATION_NAME]:
        tx = TransferFacts(op)
        if len(tx.tile_shape) > descriptor_dims:
            op = _apply(tx, choose_fit(tx, descriptor_dims))
        if tx.cut:
            _split_cut(TransferFacts(op), descriptor_dims)


def runs_of(digits, n, stride):
    """(start, count, DRAM delta) of each run an axis of extent `n` cut into `digits` splits into --
    one per value of its outer digits, split again where the inner one wraps -- the DRAM a descriptor
    walking the whole axis at `stride` from dram_idx + delta reads at the run's coordinates."""
    inner = digits[-1]
    block = inner.radix or n
    lap = (lambda j: (inner.phase + j * inner.stride) // inner.modulus) if inner.modulus else (
        lambda j: 0)
    runs = []
    for s in range(0, n, block):
        count, p0 = min(block, n - s), 0
        for p in range(1, count + 1):
            if p == count or lap(p) != lap(p0):
                runs.append((s + p0, p - p0, cut_offset(digits, s + p0) - (s + p0) * stride))
                p0 = p
    return runs


def _cut_runs(cut, shape, stride):
    """{cut axis: its runs_of} for a transfer of tile `shape` whose `cut` is {axis: Digits}."""
    return {d: runs_of(digits, shape[d], stride[d]) for d, digits in sorted(cut.items())}


def _split_cut(tx, dims):
    """One transfer per combination of the cut axes' runs, each moving that box of the tile; a static
    clamp on a cut axis narrows every box to it, and an MVIN fills what it leaves (_fill_pieces)."""
    from mlir.ir import InsertionPoint
    if tx.subtile is not None:
        raise NotImplementedError(
            f"a cut tile that is also sub-tiled: a piece's box names tile rows and the "
            f"subtile restarts them: {tx.op}")
    clamp = _static_clamps(tx)
    by_axis = _cut_runs(tx.cut, tx.tile_shape, tx.dram_stride)
    axes, runs = list(by_axis), list(by_axis.values())
    load = is_load(tx.op)
    whole = math.prod(n if load or d not in clamp else clamp[d][1] - clamp[d][0]
                      for d, n in enumerate(moved_shape(tx.op)))
    moved = 0
    with InsertionPoint(tx.op):
        for combo in itertools.product(*runs):
            box = {d: _meet((start, start + count), clamp.get(d))
                   for d, (start, count, _) in zip(axes, combo)}
            if any(lo >= hi for lo, hi in box.values()):
                continue
            origin = sum(delta for _, _, delta in combo) + sum(
                lo * tx.dram_stride[d] for d, (lo, _) in box.items())
            dram_idx = _plus(tx.dram_idx, origin) if origin else tx.dram_idx
            moved += _piece_of(tx, box, dram_idx, set(clamp))
        if load:
            for box, d in _fill_pieces(clamp, tx.tile_shape):
                moved += _piece_of(tx, box, tx.dram_idx, set(clamp) - {d})
    if moved != whole:
        raise NotImplementedError(
            f"the pieces of this cut move {moved} elements and the transfer {whole}, so the "
            f"wait on its semaphore would count neither: {tx.op}")
    tx.op.erase()


def _static_clamps(tx):
    """{cut axis: its clamp's [lo, hi) held to the axis}, refusing a clamp a runtime value bounds."""
    from .lib_ir import const_int
    out = {}
    for d, lo, hi in tx.masked_pairs:
        if d not in tx.cut:
            continue
        lo, hi = const_int(lo), const_int(hi)
        if lo is None or hi is None:
            raise NotImplementedError(
                f"axis {d} is cut and clamped by a runtime bound: a piece's box is static, so "
                f"no box can be met with that clamp: {tx.op}")
        lo = min(max(lo, 0), tx.tile_shape[d])
        out[d] = (lo, max(min(hi, tx.tile_shape[d]), lo))
    return out


def _meet(run, clamp):
    return run if clamp is None else (max(run[0], clamp[0]), min(run[1], clamp[1]))


def _fill_pieces(clamp, shape):
    """(box, axis) of each MVIN piece that takes only the fill: per clamped cut axis in turn, each
    range of it outside its clamp, the axes before it held to their clamps, so the boxes partition."""
    out, before = [], {}
    for d in sorted(clamp):
        lo, hi = clamp[d]
        for gap in ((0, lo), (hi, shape[d])):
            if gap[0] < gap[1] and all(a < b for a, b in before.values()):
                out.append(({**before, d: gap}, d))
        before[d] = (lo, hi)
    return out


def _piece_of(tx, box, dram_idx, unclamp):
    """One piece of `tx` at the insertion point moving `box` from `dram_idx`, without the clamps
    of `unclamp`; the elements it moves."""
    from mlir.ir import ArrayAttr, DenseI64ArrayAttr, IntegerAttr, IntegerType
    piece = tx.op.clone()
    del piece.attributes["cut"]
    piece.attributes["piece_box"] = ArrayAttr.get(
        [DenseI64ArrayAttr.get(list(box.get(d, ()))) for d in range(max(tx.spad_rank, 1))])
    piece.operands[operand_number(piece, "dram_idx")] = dram_idx
    kept = [(d, lo, hi) for d, lo, hi in tx.masked_pairs if d not in unclamp]
    if len(kept) != len(tx.masked_pairs):
        i64 = IntegerType.get_signless(64)
        fill = piece.opview.dyn_fill
        new = restated(piece, operands={
            "clamp": [v for _, lo, hi in kept for v in (lo, hi)],
            "dyn_fill": [fill] if kept and fill is not None else []}, attributes={
            "masked_axes": ArrayAttr.get([IntegerAttr.get(i64, d) for d, _, _ in kept])
            if kept else None,
            **({} if kept else {"masked_fill": None})})
        piece.erase()
        piece = new
    return math.prod(moved_shape(piece))


def _plus(index, c):
    from mlir.ir import AffineConstantExpr, AffineExpr, AffineMap, AffineMapAttr, IndexType, Operation
    expr = AffineExpr.get_dim(0) + AffineConstantExpr.get(c)
    return Operation.create(
        "affine.apply", results=[IndexType.get()], operands=[index],
        attributes={"map": AffineMapAttr.get(AffineMap.get(1, 0, [expr]))}).results[0]


class DescriptorCannotState(NotImplementedError, Refusal):
    """No fold or peel states the tile in the descriptor's dims."""
    reason = Reason.DESCRIPTOR_CANNOT_STATE


def choose_fit(tx, dims):
    """The fit issuing fewest DMAs -- its loop trips times the pieces its cut splits into -- then
    walking fewest loops, then peeling the outermost axes."""
    fits, why = candidate_fits(tx, dims)
    if not fits:
        raise DescriptorCannotState(
            f"no fold or peel states this rank-{tx.spad_rank} tile {tx.tile_shape} in "
            f"{dims} descriptor dims: {'; '.join(dict.fromkeys(why))}: {tx.op}")
    return min(fits, key=lambda f: (f.dmas, len(f.peeled), f.peeled))


def candidate_fits(tx, dims):
    """Every (peeled axes, piece axes as groups of tile axes) a descriptor states and every per-axis
    attribute survives, and why each other peel is not one. Lane axis 0 is never peeled nor grouped."""
    movable = [d for d in tx.eff if d != 0]
    fits, why = [], []
    for n in range(len(movable) + 1):
        for peeled in itertools.combinations(movable, n):
            try:
                _refuse(tx, peeled)
                groups = _groups(tx, peeled, dims)
                pieces = _pieces_of(tx, groups, renumbered_per_axis(tx.op, groups, tx.tile_shape))
            except NotImplementedError as e:
                why.append(str(e))
                continue
            fits.append(Fit(peeled, groups,
                            math.prod(tx.tile_shape[d] for d in peeled) * pieces))
    return fits, why


def _pieces_of(tx, groups, attrs):
    """How many pieces the cut of the fit's one transfer, restated in `attrs`, splits it into."""
    from mlir.ir import ArrayAttr
    from ...dialects.torchsim import cut_digits
    if attrs.get("cut") is None:
        return 1
    shape = [math.prod(tx.tile_shape[d] for d in g) for g in groups]
    stride = int_array(attrs["dram_stride"])
    cut = {k: cut_digits(int_array(a)) for k, a in enumerate(ArrayAttr(attrs["cut"]))
           if len(int_array(a))}
    return math.prod(len(r) for r in _cut_runs(cut, shape, stride).values())


def _named_axes(tx):
    named = {d for d, _lo, _hi in tx.masked_pairs} | {d for d, _v in tx.dyn_strides}
    named |= set(tx.piece_box or ())
    return named | ({tx.gather_axis} if tx.gather_axis is not None else set())


def _groups(tx, peeled, dims):
    shape, named = tx.tile_shape, _named_axes(tx)
    if peeled:
        kept = [d for d in range(tx.spad_rank)
                if d not in peeled and (d == 0 or shape[d] != 1 or d in named)]
        groups = ([[d] for d in kept] if len(kept) <= dims
                  else _merge(tx, [[d] for d in kept], named))
    else:
        groups = _squeeze(shape, named)
        if len(groups) > dims:
            groups = _merge(tx, groups, named)
    if len(groups) > dims:
        raise NotImplementedError(
            f"{'peeling ' + str(list(peeled)) if peeled else 'with no peel, folding'} leaves "
            f"{len(groups)} axes")
    return groups


def _squeeze(shape, named):
    """Each extent-1 axis no attribute names joins the next axis that moves or is named (trailing
    ones the last); axis 0 stands alone, and no two named axes share a group."""
    groups, pending = [[0]], []
    for d in range(1, len(shape)):
        if shape[d] == 1 and d not in named:
            pending.append(d)
        else:
            groups.append(pending + [d])
            pending = []
    groups[-1] += pending
    return groups


def _merge(tx, groups, named):
    """Adjacent groups whose moving axes are contiguous in both DRAM and the bank, as one."""
    if tx.subtile is not None:
        return groups
    shape, dram, tile = tx.tile_shape, tx.dram_stride, tx.tile_stride

    def inner(g):
        return next((d for d in reversed(g) if shape[d] != 1), g[0])

    out = [list(groups[0])]
    for g in groups[1:]:
        p, i = inner(out[-1]), inner(g)
        if (len(out) > 1 and not (named & {p, i}) and shape[p] != 1 and shape[i] != 1
                and dram[p] == dram[i] * shape[i] and tile[p] == tile[i] * shape[i]):
            out[-1] += g
        else:
            out.append(list(g))
    return out


def _refuse(tx, peeled):
    """The peels a piece cannot state."""
    from mlir.ir import ShapedType
    if not peeled:
        return
    if tx.uniform:
        raise NotImplementedError(
            "a uniform tile's transfer is not cut into pieces here: no kernel reaches such a peel, "
            "so what Spike answers for one is unchecked")
    so = strides_and_offset(tx.sram_ty)
    dyn = ShapedType.get_dynamic_stride_or_offset()
    if so is None or dyn in so[0] or so[1] == dyn:
        raise NotImplementedError(
            f"a peeled tile whose scratchpad type states no static strides: a piece's subview "
            f"has nothing to say where its elements are: {tx.sram_ty}")
    if any(d in peeled for d, _lo, _hi in tx.masked_pairs):
        raise NotImplementedError(f"masked-DMA clamp on peeled axis {list(peeled)}")
    if set(tx.piece_box or ()) & set(peeled):
        raise NotImplementedError(f"piece box on peeled axis {list(peeled)}")
    if any(d in peeled for d, _v in tx.dyn_strides):
        raise NotImplementedError(f"runtime stride on peeled axis {list(peeled)}")
    if tx.cut and set(tx.cut) & set(peeled):
        raise NotImplementedError(
            f"cut axis {sorted(set(tx.cut) & set(peeled))} peeled: a loop steps DRAM by "
            f"dram_stride per trip, and the digits of a cut axis do not step it affinely")
    if tx.gather_axis in peeled:
        raise NotImplementedError(
            f"the index buffer holds one entry per position on axis {tx.gather_axis}, which "
            f"peeling {list(peeled)} walks in a loop, so the coordinate the DMA reads "
            f"restarts")
    if tx.indirect and tx.gather_axis is None:
        raise NotImplementedError(
            f"an indirect transfer with no indirect_dim (masked, or its index computed) reads a "
            f"tile-shaped index buffer from the start, and every piece of a peel would read the "
            f"first piece's indices")


def _apply(tx, fit):
    from mlir.dialects import affine
    from mlir.ir import InsertionPoint
    ip = InsertionPoint(tx.op)
    ivs = []
    for d in fit.peeled:
        loop = affine.AffineForOp(0, tx.tile_shape[d], 1, ip=ip)
        ivs.append(loop.induction_variable)
        with InsertionPoint(loop.body):
            affine.AffineYieldOp([])
        ip = InsertionPoint.at_block_terminator(loop.body)
    with ip:
        piece = _piece(tx, fit, ivs)
    whole = math.prod(moved_shape(tx.op))
    pieces = math.prod(tx.tile_shape[d] for d in fit.peeled) * math.prod(moved_shape(piece))
    if pieces != whole:
        raise NotImplementedError(
            f"the pieces of this fit move {pieces} elements and the transfer {whole}, so the "
            f"wait on its semaphore would count neither: {tx.op}")
    tx.op.erase()
    return piece


def _piece(tx, fit, ivs):
    """One transfer of the fit at the current insertion point: a view of the kept axes (a tile-shaped
    index buffer folded with it), the loops' offsets folded into both indices, and every per-axis
    attribute renumbered to the groups."""
    from mlir.ir import AffineExpr, AffineMap, AffineMapAttr, IndexType, MemRefType, Operation
    idx = IndexType.get()
    kept = [d for g in fit.groups for d in g]
    view = _subview(tx, kept) if fit.peeled else tx.sram
    at = {d: k for k, d in enumerate(kept)}
    moved = {}
    if any(len(g) > 1 for g in fit.groups):
        groups = [[at[d] for d in g] for g in fit.groups]
        view = _collapse(view, groups)
        if (tx.indirect and tx.gather_axis is None and not fit.peeled
                and list(MemRefType(tx.offset_buf.type).shape) == list(tx.tile_shape)):
            moved["index"] = _collapse(tx.offset_buf, groups)

    def offset(base, steps):
        expr = AffineExpr.get_dim(0)
        for k, s in enumerate(steps):
            expr = expr + AffineExpr.get_dim(k + 1) * s
        return Operation.create(
            "affine.apply", results=[idx], operands=[base, *ivs],
            attributes={"map": AffineMapAttr.get(AffineMap.get(len(steps) + 1, 0, [expr]))}
        ).results[0]

    moved["sram"] = view
    if fit.peeled:
        moved["sram_idx"] = offset(tx.sram_idx, [tx.tile_stride[d] for d in fit.peeled])
        moved["dram_idx"] = offset(tx.dram_idx, [tx.dram_stride[d] for d in fit.peeled])
    piece = tx.op.clone()
    for name, v in moved.items():
        piece.operands[operand_number(piece, name)] = v
    renumber_per_axis(piece, fit.groups, tx.tile_shape)
    return piece


def _subview(tx, kept):
    from mlir.ir import (DenseI32ArrayAttr, DenseI64ArrayAttr, MemRefType, Operation,
                         StridedLayoutAttr)
    rank = tx.spad_rank
    strides, off = strides_and_offset(tx.sram_ty)
    ty = MemRefType.get([tx.tile_shape[d] for d in kept], tx.elem,
                        layout=StridedLayoutAttr.get(off, [strides[d] for d in kept]),
                        memory_space=tx.space)
    return Operation.create(
        "memref.subview", results=[ty], operands=[tx.sram],
        attributes={"static_offsets": DenseI64ArrayAttr.get([0] * rank),
                    "static_sizes": DenseI64ArrayAttr.get(
                        [tx.tile_shape[d] if d in kept else 1 for d in range(rank)]),
                    "static_strides": DenseI64ArrayAttr.get([1] * rank),
                    "operandSegmentSizes": DenseI32ArrayAttr.get([1, 0, 0, 0])}).results[0]


def _collapse(view, groups):
    from mlir.ir import (ArrayAttr, IntegerAttr, IntegerType, MemRefType, Operation,
                         StridedLayoutAttr)
    i64 = IntegerType.get_signless(64)
    ty = MemRefType(view.type)
    shape = list(ty.shape)
    target = [math.prod(shape[d] for d in g) for g in groups]
    try:
        StridedLayoutAttr(ty.layout)
        strided = True
    except (TypeError, ValueError):
        strided = False
    if strided:
        strides, off = strides_and_offset(ty)
        layout = StridedLayoutAttr.get(
            off, [strides[next((d for d in reversed(g) if shape[d] != 1), g[0])] for g in groups])
        out = MemRefType.get(target, ty.element_type, layout=layout, memory_space=ty.memory_space)
    else:
        out = MemRefType.get(target, ty.element_type, memory_space=ty.memory_space)
    return Operation.create(
        "memref.collapse_shape", results=[out], operands=[view],
        attributes={"reassociation": ArrayAttr.get(
            [ArrayAttr.get([IntegerAttr.get(i64, d) for d in g]) for g in groups])}).results[0]
