"""layout_convert: what a `pytorchsim.convert_layout` moves, for whichever unit a target carries it on --
its kind (broadcast, window, reshape, relayout), each element's
`(source (bank, row), destination (bank, row))` pair on the machine's banks, and the rows a copy from each
bank's own store would take."""

import itertools
import math
from collections.abc import Sequence

from .bank_layout import is_uniform
from .ir_utils import int_array


class PairTable(Sequence):
    """Pairs held as four integer arrays -- source lane, source row, destination lane, destination
    row -- read as the `((lane, row), (lane, row))` list only when something walks them: a convert
    moves hundreds of thousands of elements, and the folds and rounds below take the arrays whole."""

    def __init__(self, s_lane, s_row, d_lane, d_row):
        self.cols = (s_lane, s_row, d_lane, d_row)
        self._list = None

    def _items(self):
        if self._list is None:
            s_lane, s_row, d_lane, d_row = (c.tolist() for c in self.cols)
            self._list = list(zip(zip(s_lane, s_row), zip(d_lane, d_row)))
        return self._list

    def __len__(self):
        return int(self.cols[0].size)

    def __getitem__(self, i):
        return self._items()[i]

    def __iter__(self):
        return iter(self._items())


def fold_onto_lanes(pairs, s_shape, d_shape, lanes):
    """`pairs` on the lanes the machine HAS: lane-full t of an axis 0 longer than `lanes` sits t
    times the other axes' rows deeper, as the DMA stages it and need_per_lane reserves it; a last,
    partial lane-full (a scan pads to one and a half) on the lanes it reaches, as deep as a whole."""
    if pairs is None:
        return None
    fs, fd = _folds(s_shape, lanes), _folds(d_shape, lanes)
    if isinstance(pairs, PairTable):
        if not fs and not fd:
            return pairs
        import numpy as np
        s_lane, s_row, d_lane, d_row = pairs.cols
        if fs:
            q, s_lane = np.divmod(s_lane, lanes)
            s_row = q * _rows_of(s_shape) + s_row
        if fd:
            q, d_lane = np.divmod(d_lane, lanes)
            d_row = q * _rows_of(d_shape) + d_row
        return PairTable(s_lane, s_row, d_lane, d_row)
    if not fs and not fd:
        return list(pairs)
    n = lanes
    if fs and fd:
        rs, rd = _rows_of(s_shape), _rows_of(d_shape)
        return [((a % n, a // n * rs + r), (b % n, b // n * rd + q)) for (a, r), (b, q) in pairs]
    if fs:
        rs = _rows_of(s_shape)
        return [((a % n, a // n * rs + r), t) for (a, r), t in pairs]
    rd = _rows_of(d_shape)
    return [(f, (b % n, b // n * rd + q)) for f, (b, q) in pairs]


def _folds(shape, lanes):
    return bool(shape) and shape[0] > lanes


def kind_of(op):
    if "added_dims" in op.attributes:
        return "broadcast"
    if "static_offsets" in op.attributes:
        return "window"
    src, dst = op.operands[0], op.operands[1]
    if (logical_shape(list(src.type.shape), perm_of(op, "src_perm"))
            != logical_shape(list(dst.type.shape), perm_of(op, "dst_perm"))):
        return "reshape"
    return "relayout"


def pairs_of(op, kind, lanes):
    """Each element's `(source (lane, row), destination (lane, row))`; a UNIFORM destination is
    held by every lane, so each of its elements is listed once for every lane of the machine."""
    pairs = _pairs_by_strides(op, kind, lanes)
    return _pairs_of_slow(op, kind, lanes) if pairs is None else pairs


def _pairs_by_strides(op, kind, lanes):
    """pairs_of in array arithmetic: flat_of is linear in an element's coordinates, so each side's
    flat index over the row-major walk elements_of takes is a sum of per-axis weights; positions_of
    is that index's (index // the other axes' rows, index % them). None where an index could leave
    its buffer (the element walk's own errors and wraps are _pairs_of_slow's to keep)."""
    import numpy as np
    src, dst = op.operands[0], op.operands[1]
    s_phys, d_phys = list(src.type.shape), list(dst.type.shape)
    if any(n < 0 for n in s_phys + d_phys):
        return None
    ps, pd = perm_of(op, "src_perm"), perm_of(op, "dst_perm")
    ss, ds = logical_shape(s_phys, ps), logical_shape(d_phys, pd)
    ws, wd = _flat_weights(ss, ps), _flat_weights(ds, pd)
    if kind == "broadcast":
        added = set(int_array(op.attributes["added_dims"]))
        kept = [d for d in range(len(ds)) if d not in added]
        if len(kept) != len(ss):
            return None
        on_d = [0] * len(ds)
        for i, d in enumerate(kept):
            on_d[d] = ws[i]
        walk, s_w, d_w, s_0, d_0 = ds, on_d, wd, 0, 0
    elif kind == "window":
        from .indexing_maps import get_kept_axes
        into = "into_window" in op.attributes
        offs, sizes, steps = window_of(op)
        small, big_w = (ss, wd) if into else (ds, ws)
        kept = get_kept_axes(sizes, small)
        if kept is None or len(offs) != len(big_w) or len(steps) != len(big_w):
            return None
        on_small = [steps[a] * big_w[a] for a in kept]
        base = sum(o * w for o, w in zip(offs, big_w))
        small_w = ws if into else wd
        walk = small
        s_w, d_w, s_0, d_0 = ((small_w, on_small, 0, base) if into
                              else (on_small, small_w, base, 0))
    elif kind == "reshape":
        if math.prod(ss) != math.prod(ds):
            return None
        s_flat = _flat_over(ss, ws, 0, np)
        d_flat = _flat_over(ds, wd, 0, np)
        walk = None
    else:
        walk, s_w, d_w, s_0, d_0 = ds, ws, wd, 0, 0
    if walk is not None:
        s_flat = _flat_over(walk, s_w, s_0, np)
        d_flat = _flat_over(walk, d_w, d_0, np)
    s_n, d_n = math.prod(s_phys), math.prod(d_phys)
    if s_flat.size and (s_flat.min() < 0 or s_flat.max() >= s_n
                        or d_flat.min() < 0 or d_flat.max() >= d_n):
        return None
    s_lane, s_row = np.divmod(s_flat, _rows_of(s_phys))
    d_lane, d_row = np.divmod(d_flat, _rows_of(d_phys))
    if is_uniform(dst):
        s_lane, s_row = np.repeat(s_lane, lanes), np.repeat(s_row, lanes)
        d_lane, d_row = np.tile(np.arange(lanes, dtype=np.int64), d_flat.size), np.repeat(d_row, lanes)
    return PairTable(s_lane, s_row, d_lane, d_row)


def _flat_weights(shape, perm):
    """Each logical axis's weight in flat_of(shape, perm, .)."""
    w, acc = [0] * len(shape), 1
    for a in reversed(perm):
        w[a] = acc
        acc *= shape[a]
    return w


def _flat_over(walk, weights, base, np):
    """`base + sum(x[i] * weights[i])` for every x of `walk`'s row-major walk, as one int64 array."""
    out = np.full((1,) * len(walk), base, dtype=np.int64)
    for i, (n, w) in enumerate(zip(walk, weights)):
        ax = [1] * len(walk)
        ax[i] = n
        out = out + (np.arange(n, dtype=np.int64) * w).reshape(ax)
    return np.broadcast_to(out, tuple(walk)).ravel()


def _rows_of(shape):
    """The rows per lane positions_of gives a buffer of `shape`: its axes after the first."""
    return math.prod(shape[1:]) if shape else 1


def _pairs_of_slow(op, kind, lanes):
    src, dst = op.operands[0], op.operands[1]
    ps, pd = perm_of(op, "src_perm"), perm_of(op, "dst_perm")
    ss, ds = logical_shape(list(src.type.shape), ps), logical_shape(list(dst.type.shape), pd)
    at_s, at_d = positions_of(list(src.type.shape)), positions_of(list(dst.type.shape))
    pairs = [(at_s[flat_of(ss, ps, a)], at_d[flat_of(ds, pd, b)])
             for a, b in elements_of(op, kind, ss, ds)]
    if not is_uniform(dst):
        return pairs
    return [(f, (lane, t[1])) for f, t in pairs for lane in range(lanes)]


def held_by(dst, lanes):
    """Every `(lane, row)` the destination holds, on the machine's lanes: a UNIFORM one in every
    lane."""
    shape = list(dst.type.shape)
    at = positions_of(shape)
    if is_uniform(dst):
        return [(lane, t[1]) for t in at for lane in range(lanes)]
    return [t for _f, t in fold_onto_lanes([((0, 0), t) for t in at], [], shape, lanes)]


def elements_of(op, kind, ss, ds):
    from .indexing_maps import get_kept_axes
    if kind == "broadcast":
        added = set(int_array(op.attributes["added_dims"]))
        kept = [d for d in range(len(ds)) if d not in added]
        return [([y[d] for d in kept], y) for y in all_coords(ds)]
    if kind == "window":
        into = "into_window" in op.attributes
        offs, sizes, steps = window_of(op)
        small, big = (ss, ds) if into else (ds, ss)
        kept = get_kept_axes(sizes, small)
        out = []
        for y in all_coords(small):
            x = list(offs)
            for j, a in enumerate(kept):
                x[a] += steps[a] * y[j]
            out.append((y, x) if into else (x, y))
        return out
    if kind == "reshape":
        return [(coords_of(f, ss), coords_of(f, ds)) for f in range(math.prod(ss))]
    return [(y, y) for y in all_coords(ds)]


def positions_of(shape):
    """Each element's (index on the first axis, its row-major row over the rest), in row-major order."""
    if not shape:
        return [(0, 0)]
    rows = _rows_of(shape)
    return [(a, r) for a in range(shape[0]) for r in range(rows)]


def perm_of(op, name):
    return list(int_array(op.attributes[name]))


def logical_shape(shape, perm):
    out = [0] * len(shape)
    for k, a in enumerate(perm):
        out[a] = shape[k]
    return out


def window_of(op):
    return tuple(list(int_array(op.attributes[n]))
                 for n in ("static_offsets", "static_sizes", "static_strides"))


def coords_of(f, shape):
    x = [0] * len(shape)
    for i in reversed(range(len(shape))):
        x[i], f = f % shape[i], f // shape[i]
    return x


def all_coords(shape):
    return [list(x) for x in itertools.product(*[range(n) for n in shape])]


def flat_of(shape, perm, x):
    n = 0
    for a in perm:
        n = n * shape[a] + x[a]
    return n


def physical_groups(op):
    """For a reshape copy into a destination laid as it reads, the source's physical dims each
    destination dim gathers, outer first: its logical dims' row-major collapse; else None."""
    src, dst = op.operands[0], op.operands[1]
    ps, pd = perm_of(op, "src_perm"), perm_of(op, "dst_perm")
    if kind_of(op) != "reshape" or pd != list(range(len(pd))):
        return None
    groups = collapse_groups(logical_shape(list(src.type.shape), ps), list(dst.type.shape))
    return None if groups is None else [[ps.index(a) for a in g] for g in groups]


def collapse_groups(s_shape, d_shape):
    """The groups of adjacent source dims each destination dim is the product of, or None."""
    groups, k = [], 0
    for n in d_shape:
        g, prod = [], 1
        while k < len(s_shape) and (prod < n or (not g and s_shape[k] == n)):
            prod *= s_shape[k]
            g.append(k)
            k += 1
            if prod == n:
                break
        if prod != n or not g:
            return None
        groups.append(g)
    return groups if k == len(s_shape) else None


def in_bank_rows(pairs, held, s_shape, d_shape, lanes):
    """Each destination row (its offset in a lane's bank) and the source row it takes, or None when
    a row does not take the same source row in every lane, or the copy leaves a lane's part of a row
    it writes unwritten: a copy from each lane's own bank writes the row in every lane."""
    rows, written = {}, set()
    for (_sl, sd), at in fold_onto_lanes(pairs, s_shape, d_shape, lanes):
        if rows.setdefault(at[1], sd) != sd:
            return None
        written.add(at)
    if any(at[1] in rows and at not in written for at in held):
        return None
    return rows
