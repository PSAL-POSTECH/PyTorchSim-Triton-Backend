"""torchsim-fit-transfers-to-descriptor: a transfer of more axes than the target's DMA descriptor has
dims (the lower-dma provider's DESCRIPTOR_DIMS) becomes the transfers it states -- folded, peeled into
affine.for loops, or both, whichever issues fewest DMAs -- and a cut transfer becomes its pieces, one
transfer per run of every cut axis, each moving its `piece_box` from the DRAM its first element is at."""

import itertools
import math
from collections import namedtuple

from ...contract.stop import Reason, Refusal

from ...dialects.torchsim import (TransferOp, moved_shape, operand_number, renumber_per_axis,
                                  renumbered_per_axis)
from .lib_descriptor import TransferFacts
from .lib_ir import strides_and_offset, walk

MARKERS = (TransferOp.OPERATION_NAME,)

Fit = namedtuple("Fit", "peeled groups dmas")


def run(module, *, descriptor_dims, **_):
    """Fit every transfer of `module` whose tile has more axes than the descriptor has dims, and
    split every cut transfer into its pieces."""
    for op in [o for o in walk(module.operation) if o.name == TransferOp.OPERATION_NAME]:
        tx = TransferFacts(op)
        if tx.cut:
            _split_cut(tx, descriptor_dims)
        elif len(tx.tile_shape) > descriptor_dims:
            _apply(tx, choose_fit(tx, descriptor_dims))


def runs_of(kind, a, b, st, n, *digits):
    """(start, count, DRAM delta) of each run one cut axis's entry splits it into: the DRAM a
    descriptor walking the whole axis from dram_idx + delta reads at the run's coordinates."""
    from .p01_convert_accesses_to_transfer import CUT_SPLIT, CUT_WRAP
    if kind == CUT_SPLIT:
        sizes, strides = digits[0::2], digits[1::2]
        inner, step = sizes[-1], strides[-1]
        runs = []
        for idx in itertools.product(*(range(s) for s in sizes[:-1])):
            start = 0
            for k, i in enumerate(idx):
                start = start * sizes[k] + i
            start *= inner
            runs.append((start, inner, sum(i * t for i, t in zip(idx, strides)) - start * step))
        return runs
    if kind == CUT_WRAP:
        W, o = a, b
        runs, p0 = [], 0
        for p in range(1, n + 1):
            if p == n or (o + p * st) // W != (o + p0 * st) // W:
                runs.append((p0, p - p0, -((o + p0 * st) // W) * W))
                p0 = p
        return runs
    return [(j, min(a, n - j), j // a * st) for j in range(0, n, a)]


def _split_cut(tx, dims):
    """One transfer per combination of the cut axes' runs, each moving that box of the tile."""
    from mlir.ir import ArrayAttr, DenseI64ArrayAttr, InsertionPoint
    if len(tx.tile_shape) > dims:
        raise NotImplementedError(
            f"a wrapped tile of rank {len(tx.tile_shape)} would be collapsed or "
            f"peeled, and a piece's box does not survive either remap: {tx.op}")
    clamped = {d for d, _lo, _hi in tx.masked_pairs}
    if clamped & set(tx.cut):
        raise NotImplementedError(
            f"axis {sorted(clamped & set(tx.cut))} is both cut and clamped: "
            f"outside the box the clamp wants a fill and the piece wants a "
            f"skip, and one box cannot say both: {tx.op}")
    if tx.subtile is not None:
        raise NotImplementedError(
            f"a cut tile that is also sub-tiled: a piece's box names tile rows and the "
            f"subtile restarts them: {tx.op}")
    axes = sorted(tx.cut)
    runs = [runs_of(*tx.cut[d]) for d in axes]
    rank = max(tx.spad_rank, 1)
    whole, moved = math.prod(moved_shape(tx.op)), 0
    with InsertionPoint(tx.op):
        for combo in itertools.product(*runs):
            box = {d: (start, start + count) for d, (start, count, _) in zip(axes, combo)}
            origin = sum(delta for _, _, delta in combo) + sum(
                lo * tx.dram_stride[d] for d, (lo, _) in box.items())
            dram_idx = _plus(tx.dram_idx, origin) if origin else tx.dram_idx
            piece = tx.op.clone()
            del piece.attributes["cut"]
            piece.attributes["piece_box"] = ArrayAttr.get(
                [DenseI64ArrayAttr.get(list(box.get(d, ()))) for d in range(rank)])
            piece.operands[operand_number(piece, "dram_idx")] = dram_idx
            moved += math.prod(moved_shape(piece))
    if moved != whole:
        raise NotImplementedError(
            f"the pieces of this cut move {moved} elements and the transfer {whole}, so the "
            f"wait on its semaphore would count neither: {tx.op}")
    tx.op.erase()


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
    """The fit issuing fewest DMAs, then walking fewest loops, then peeling the outermost axes."""
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
                renumbered_per_axis(tx.op, groups, tx.tile_shape)
            except NotImplementedError as e:
                why.append(str(e))
                continue
            fits.append(Fit(peeled, groups, math.prod(tx.tile_shape[d] for d in peeled)))
    return fits, why


def _named_axes(tx):
    named = {d for d, _lo, _hi in tx.masked_pairs} | {d for d, _v in tx.dyn_strides}
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
    if any(d in peeled for d, _v in tx.dyn_strides):
        raise NotImplementedError(f"runtime stride on peeled axis {list(peeled)}")
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
    pieces = fit.dmas * math.prod(moved_shape(piece))
    if pieces != whole:
        raise NotImplementedError(
            f"the pieces of this fit move {pieces} elements and the transfer {whole}, so the "
            f"wait on its semaphore would count neither: {tx.op}")
    tx.op.erase()


def _piece(tx, fit, ivs):
    """One transfer of the fit at the current insertion point: a view of the kept axes, the loops'
    offsets folded into both indices, and every per-axis attribute renumbered to the groups."""
    from mlir.ir import AffineExpr, AffineMap, AffineMapAttr, IndexType, Operation
    idx = IndexType.get()
    kept = [d for g in fit.groups for d in g]
    view = _subview(tx, kept) if fit.peeled else tx.sram
    at = {d: k for k, d in enumerate(kept)}
    if any(len(g) > 1 for g in fit.groups):
        view = _collapse(view, [[at[d] for d in g] for g in fit.groups])

    def offset(base, steps):
        expr = AffineExpr.get_dim(0)
        for k, s in enumerate(steps):
            expr = expr + AffineExpr.get_dim(k + 1) * s
        return Operation.create(
            "affine.apply", results=[idx], operands=[base, *ivs],
            attributes={"map": AffineMapAttr.get(AffineMap.get(len(steps) + 1, 0, [expr]))}
        ).results[0]

    moved = {"sram": view}
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
