"""layout_convert: what a `pytorchsim.convert_layout` moves, for whichever unit a target carries it on --
its kind (broadcast, window, reshape, relayout), each element's
`(source (bank, row), destination (bank, row))` pair on the machine's banks, and the rows a copy from each
bank's own store would take."""

import itertools
import math

from .bank_layout import is_uniform, row_major
from .ir_utils import int_array


def fold_onto_lanes(pairs, s_shape, d_shape, lanes):
    """`pairs` on the lanes the machine HAS: lane-full t of an axis 0 longer than `lanes` sits t
    times the other axes' rows deeper, as the DMA stages it and need_per_lane reserves it; a last,
    partial lane-full (a scan pads to one and a half) on the lanes it reaches, as deep as a whole."""
    def fold(shape):
        if not shape or shape[0] <= lanes:
            return lambda at: at
        rows = 1
        for d in shape:
            rows *= d
        rows //= shape[0]
        return lambda at: (at[0] % lanes, at[0] // lanes * rows + at[1])
    if pairs is None:
        return None
    fs, fd = fold(s_shape), fold(d_shape)
    return [(fs(f), fd(t)) for f, t in pairs]


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
    if not shape:
        return [(0, 0)]
    rows = row_major(shape)
    return [(x[0], sum(x[i] * rows[i] for i in range(1, len(shape))))
            for x in all_coords(shape)]


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
