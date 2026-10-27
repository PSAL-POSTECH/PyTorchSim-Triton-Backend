"""indexing_maps: which iteration dim each axis of each operand IS, read off the IR."""

from ...dialect.pytorchsim import TransferOp, operand_number
from .ir_utils import CommonFactors, array_kind, indexing_maps, int_array, name_of, reassoc_groups, rank_of


_NO_EDGE = ("tts.load", "tts.store", "tts.gather", "tts.scatter",
           "tts.make_tptr", "tts.make_gather_scatter_tptr",
           TransferOp.OPERATION_NAME, "func.return")


def _stated_maps(op):
    """Per operand of `op`'s indexing maps, the iteration dim each result IS (an AffineDimExpr's
    position), None for a result that is any other expression."""
    from mlir.ir import AffineDimExpr
    maps = indexing_maps(op)
    if maps is None:
        return None
    return [[AffineDimExpr(r).position if isinstance(r, AffineDimExpr) else None for r in m.results]
            for m in maps]


MATMUL_M, MATMUL_N, MATMUL_K = 0, 1, 2


def matmul_axis(op, slot, dim):
    """Which axis of pytorchsim.dot operand `slot` carries iteration `dim`."""
    maps = maps_of(op)
    m = maps[slot] if maps is not None and slot < len(maps) else None
    if m is not None and dim in m:
        return m.index(dim)
    raise ValueError(
        f"matmul operand {slot} does not name iteration dim d{dim} in its "
        f"indexing map; there is no axis for the array to walk: {op}")


def maps_of(op):
    """Per-operand axis -> iteration dim, or None when the op relates no axes."""
    nm = name_of(op)

    if nm == TransferOp.OPERATION_NAME and op.operation.opview.index is not None:
        tile, index = op.operation.opview.sram, op.operation.opview.index
        n = rank_of(tile) or 0
        if n and rank_of(index) == n:
            maps = [None] * (operand_number(op, "index") + 1)
            maps[operand_number(op, "sram")] = list(range(n))
            maps[operand_number(op, "index")] = list(range(n))
            return maps
        return None

    if nm in _NO_EDGE:
        return None

    if nm == "linalg.generic":
        return _stated_maps(op)

    kind = array_kind(op)

    if kind == "matmul":
        stated = _stated_maps(op)
        if stated is not None:
            return _with_result(op, stated, stated[len(stated) - 1])
        return _with_result(op, [[0, 2], [2, 1], [0, 1]], [0, 1])

    if kind == "batch_matmul":
        stated = _stated_maps(op)
        if stated is not None:
            ours = {0: 3, 1: MATMUL_M, 2: MATMUL_N, 3: MATMUL_K}
            stated = [[None if d is None else ours.get(d, d) for d in dims]
                      for dims in stated]
            return _with_result(op, stated, stated[len(stated) - 1])
        return _with_result(op, [[3, MATMUL_M, MATMUL_K],
                                 [3, MATMUL_K, MATMUL_N],
                                 [3, MATMUL_M, MATMUL_N]],
                            [3, MATMUL_M, MATMUL_N])

    if nm == "linalg.transpose":
        perm = int_array(op.attributes["permutation"])
        return _with_result(op, [list(range(len(perm))), list(perm)], list(perm))

    if nm == "linalg.reduce":
        n = rank_of(op.operands[0]) or 0
        dropped = set(int_array(op.attributes["dimensions"]))
        kept = [d for d in range(n) if d not in dropped]
        k = len(op.operands) // 2 or 1
        return _with_result(op, [list(range(n))] * k + [kept] * (
            len(op.operands) - k), kept)

    if nm == "linalg.broadcast":
        n = rank_of(op.operands[1]) or 0
        added = set(int_array(op.attributes["dimensions"]))
        kept = [d for d in range(n) if d not in added]
        allx = list(range(n))
        return _with_result(op, [kept, allx], allx)

    if nm == "linalg.fill":
        n = rank_of(op.operands[1]) or 0
        allx = list(range(n))
        return _with_result(op, [[], allx], allx)

    if nm in ("tensor.expand_shape", "memref.expand_shape",
              "tensor.collapse_shape", "memref.collapse_shape"):
        return _reassociation_maps(op)

    if nm in ("tensor.reshape", "memref.reshape"):
        return _reshape_maps(op)

    if nm in ("memref.subview", "tensor.extract_slice"):
        return _slice_maps(op)

    if nm == "memref.cast":
        allx = list(range(rank_of(op.results[0]) or 0))
        return [allx, allx]

    if nm in ("tensor.insert_slice", "tensor.insert"):
        return _insert_maps(op)

    return None


def _with_result(op, operand_maps, result_map):
    return list(operand_maps) + [result_map] * len(op.results)


def get_kept_axes(sizes, small):
    """Which axes of a `sizes`-sized window the `small` tile keeps, in order, or None."""
    from mlir.ir import ShapedType
    dyn = ShapedType.get_dynamic_size()
    kept, j = [], 0
    for i, s in enumerate(sizes):
        if j < len(small) and (s == small[j] or s == dyn or small[j] == dyn):
            kept.append(i)
            j += 1
        elif s != 1:
            return None
    return kept if j == len(small) else None


def _slice_maps(op):
    from mlir.ir import ShapedType
    sizes = int_array(op.attributes["static_sizes"])
    kept = get_kept_axes(sizes, list(ShapedType(op.results[0].type).shape))
    if kept is None:
        return None

    src = [None] * len(sizes)
    for slot, axis in enumerate(kept):
        src[axis] = slot
    return _with_result(op, [src] + [None] * (len(op.operands) - 1),
                        list(range(len(kept))))


def _insert_maps(op):
    from mlir.ir import ShapedType
    n = rank_of(op.operands[1]) or 0
    allx = list(range(n))
    if name_of(op) == "tensor.insert":
        upd = []
    else:
        upd = get_kept_axes(int_array(op.attributes["static_sizes"]),
                         list(ShapedType(op.operands[0].type).shape))
        if upd is None:
            return None
    return _with_result(op, [upd, allx] + [None] * (len(op.operands) - 2), allx)


def _reassociation_maps(op):
    from mlir.ir import ShapedType
    groups = reassoc_groups(op)
    small = list(ShapedType(op.operands[0].type).shape)
    big = list(ShapedType(op.results[0].type).shape)
    expanding = len(big) > len(small)
    grouped = big if expanding else small

    src_dims, grouped_dims = [], [None] * len(grouped)
    for i, g in enumerate(groups):
        real = [a for a in g if grouped[a] != 1]
        src_dims.append(i)
        if len(real) == 1:
            grouped_dims[real[0]] = i
    return ([src_dims, grouped_dims] if expanding
            else [grouped_dims, src_dims])


def _reshape_maps(op):
    from math import prod
    from mlir.ir import ShapedType
    src = list(ShapedType(op.operands[0].type).shape)
    dst = list(ShapedType(op.results[0].type).shape)
    if ShapedType.get_dynamic_size() in src + dst:
        return None
    if not src or not dst or prod(src) != prod(dst):
        return None
    bounds = CommonFactors(src, dst)
    ga = [list(range(i, k)) for (i, _), (k, _) in zip(bounds, bounds[1:])]
    gb = [list(range(j, k)) for (_, j), (_, k) in zip(bounds, bounds[1:])]
    mid = [prod(src[x] for x in g) for g in ga]

    def related(shape, groups):
        if shape == mid:
            return list(range(len(shape)))
        dims = [None] * len(shape)
        for j, g in enumerate(groups):
            real = [x for x in g if shape[x] != 1]
            if len(real) == 1:
                dims[real[0]] = j
        return dims

    return [related(src, ga), None, related(dst, gb)]


