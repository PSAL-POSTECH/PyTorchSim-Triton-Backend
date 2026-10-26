"""pytorchsim-drop-unread-padding: every scratchpad tile is cut, by its type, to the extent its readers need
(768 of a numel-768 block of 1024), worked backwards from the stores, on tensors, so bufferize allocates the
cut buffers; a mask that only cuts past the new extent folds to true. It runs after fuse-elementwise-ops,
whose casts re-made at each reader show who reads a tile; run before it, it cuts less."""

import logging
import math

from .lib_bank_layout import is_one_bank
from .lib_ir import (CONDITION_VALUES, FOR_ARGS, FOR_INITS, attrs_of, const_int, funcs, indexing_maps,
                     int_array, lanes_of_module, op_of, parallel_iterator, producer, reduction_iterator, walk)
from .lib_live import interval, program_ids
from .a04_fuse_elementwise_ops import CASTS, folds_lanes
from . import pass_name

NAME = pass_name(__name__)
LOG = logging.getLogger(__name__)

_LINALG = ("linalg.generic", "linalg.reduce", "linalg.fill")
_DOT = "pytorchsim.dot"

# a view axis that is part of a tile's lane axis split into [lanes, laps]: (LANE, axis) or (LAP, axis);
# a view axis a collapse merged from several tile axes, in order: (MERGED, (axis, ...))
LANE, LAP, MERGED = "lane", "lap", "merged"
_VIEWS = ("tensor.expand_shape", "tensor.collapse_shape")
_TRANSFER = "pytorchsim.transfer"
_REGIONS = ("scf.for", "scf.while", "scf.if", "scf.yield", "scf.condition")
_INTO_DEST = ("tensor.insert_slice", "tensor.insert")
_FOLD_LT = (2, 6)            # arith.cmpi predicates slt, ult

# combiner -> the padding value that leaves it unchanged
_IDENTITY = {
    "arith.addf": lambda v: v == 0.0, "arith.addi": lambda v: v == 0,
    "arith.mulf": lambda v: v == 1.0, "arith.muli": lambda v: v == 1,
    "arith.maxnumf": lambda v: v == -math.inf, "arith.maximumf": lambda v: v == -math.inf,
    "arith.minnumf": lambda v: v == math.inf, "arith.minimumf": lambda v: v == math.inf,
}

# on an elementwise copy whose mask folded and that stays: why its readers do not read its input
COPY_KEPT = "pytorchsim.copy_kept_because"

# a last axis is cut only to a multiple of this
_VECTOR_GRAIN = 8


def run(module, grid=(), **_opts):
    lanes = lanes_of_module(module)
    for func in funcs(module):
        _Live(func, grid, lanes).fit()
    return module


class _Live:
    """A tile is a use of a fresh tensor (tensor.empty, bufferization.alloc_tensor) -- bufferize gives each
    such use a buffer -- and every value written in place into it, through views and loops: its key."""

    def __init__(self, func, grid, lanes):
        self.func = func
        self.lanes = lanes
        self.known = program_ids(func, grid)
        self.ops = [o for o in walk(func.operation)
                    if _walked(o) or o.name == _TRANSFER or (
                        o.name not in _VIEWS + _REGIONS and any(_is_tensor(v) for v in o.operands))]
        self.refs = {}
        self.shape = {}          # key -> shape
        self.fresh = {}          # fresh tensor -> (op, operand) of each use; key (tensor, n) the n-th
        for o in walk(func.operation):
            if _is_fresh(o):
                v = o.results[0]
                self.fresh[v] = [(op_of(u.owner), u.operand_number) for u in v.uses]
                for n in range(len(self.fresh[v])):
                    self.shape[(v, n)] = list(_shape(v))
        self.tied = _Union()     # (key, axis) -> the axes region-carried values make one
        self.group = _Union()    # key -> the keys region-carried values make one buffer
        self.pinned = {}         # key -> why it keeps its extents
        self._tie_regions()
        self.need = {}           # tied (key, axis) -> extent its readers need
        self.readers = {}        # key group -> ops that read it
        for o in self.ops:
            for k, v in enumerate(o.operands):
                ref = self.ref_at(o, k)
                if ref is not None and self._reads(o, k):
                    self.readers.setdefault(self.group.find(ref[0]), []).append(o)

    # ---- tiles and views -----------------------------------------------------------------------

    def ref_at(self, o, k):
        """(key, axis map) of the tile `o` takes at operand `k`: `axes[i]` is the key's axis that
        axis i is, None for a unit axis a reshape added, (MERGED, axes) for several a collapse
        merged. None when the operand is not a tile this follows."""
        v = o.operands[k]
        if _is_fresh(producer(v)):
            n = next((n for n, (p, j) in enumerate(self.fresh.get(v, ())) if j == k and p == o), None)
            return None if n is None else ((v, n), list(range(len(self.shape[(v, n)]))))
        return self.root(v)

    def root(self, v):
        """ref_at of a value that is no fresh tensor itself."""
        if v not in self.refs:
            self.refs[v] = None
            self.refs[v] = self._root(v) if _is_tensor(v) else None
        return self.refs[v]

    def _root(self, v):
        o = producer(v)
        if o is None:
            host = op_of(v.owner.owner)
            if host.name == "scf.for" and v.arg_number >= FOR_ARGS:
                return self.ref_at(host, FOR_INITS + v.arg_number - FOR_ARGS)
            if host.name == "scf.while" and v.owner == host.regions[0].blocks[0]:
                return self.ref_at(host, v.arg_number)
            if host.name == "scf.while":
                cond = _terminator(host.regions[0].blocks[0])
                return self.ref_at(cond, CONDITION_VALUES + v.arg_number)
            return None
        i = v.result_number
        if o.name in _VIEWS:
            return self._view(o)
        if o.name == "scf.for":
            return self.ref_at(o, FOR_INITS + i)
        if o.name == "scf.while":
            return self.ref_at(_terminator(o.regions[0].blocks[0]), CONDITION_VALUES + i)
        if o.name == "scf.if":
            return self.ref_at(_terminator(o.regions[0].blocks[0]), i)
        k = _tied_operand(o, i)
        return None if k is None else self.ref_at(o, k)

    def _view(self, o):
        up = self.ref_at(o, 0)
        if up is None:
            return None
        src, groups = _shape(o.operands[0]), _groups(o)
        out = list(_shape(o.results[0]))
        axes = [None] * len(out)
        expand = o.name == "tensor.expand_shape"
        for s, g in enumerate(groups):
            big = [i for i in g if (out[i] if expand else src[i]) != 1]
            if len(big) > 1 and expand and len(big) == 2 and isinstance(up[1][s], int) \
                    and up[1][s] == 0 and out[big[0]] == self.lanes \
                    and not is_one_bank(self.shape[up[0]]):
                for i in g:
                    axes[i] = None
                axes[big[0]], axes[big[1]] = (LANE, 0), (LAP, 0)
                continue
            if len(big) > 1 and not expand and all(isinstance(up[1][i], int) for i in big):
                axes[s] = (MERGED, tuple(up[1][i] for i in big))
                continue
            if len(big) > 1:
                return None
            if expand:
                for i in g:
                    axes[i] = up[1][s] if i in big else None
                if not big and src[s] != 1:
                    return None
            elif big:
                axes[s] = up[1][big[0]]
        return up[0], axes

    def _tie_regions(self):
        """A loop's carried value and what a trip yields for it, and a branch's result and what each arm
        yields, are one buffer: their axes one extent. A side this does not follow pins the other."""
        for o in walk(self.func.operation):
            pairs = []
            if o.name == "scf.for":
                body = o.regions[0].blocks[0]
                pairs = [(a, _terminator(body), k) for k, a in enumerate(list(body.arguments)[FOR_ARGS:])]
            elif o.name == "scf.while":
                before = o.regions[0].blocks[0]
                pairs = [(a, _terminator(o.regions[1].blocks[0]), k) for k, a in enumerate(before.arguments)]
            elif o.name == "scf.if":
                pairs = [(r, _terminator(arm.blocks[0]), r.result_number) for r in o.results
                         for arm in list(o.regions)[1:] if len(arm.blocks)]
            for v, yld, k in pairs:
                if _is_tensor(v):
                    self._tie(self.root(v), self.ref_at(yld, k))

    def _tie(self, x, y):
        if x is None or y is None:
            for ref in (x, y):
                if ref is not None:
                    self.pinned.setdefault(ref[0], "it is carried with a tile this does not follow")
            return
        self.group.join(x[0], y[0])
        aligned = len(x[1]) == len(y[1])
        for a, b in zip(x[1], y[1]) if aligned else ():
            if (a is None or self._unit(x[0], a)) and (b is None or self._unit(y[0], b)):
                continue
            if isinstance(a, int) and isinstance(b, int):
                pairs = [(a, b)]
            elif isinstance(a, tuple) and isinstance(b, tuple) and a[0] == b[0] and a[0] != MERGED:
                pairs = [(a[1], b[1])]
            elif isinstance(a, tuple) and isinstance(b, tuple) and a[0] == b[0] and len(a[1]) == len(b[1]):
                pairs = list(zip(a[1], b[1]))
            else:
                aligned = False
                break
            for p, q in pairs:
                if self.shape[x[0]][p] != self.shape[y[0]][q]:
                    aligned = False
                self.tied.join((x[0], p), (y[0], q))
        if not aligned:
            for key in (x[0], y[0]):
                self.pinned.setdefault(key, "it is carried with a tile through views that do not align")

    def _unit(self, b, a):
        return isinstance(a, int) and self.shape[b][a] == 1

    def _need(self, b):
        return [self.need.get(self.tied.find((b, a)), 0) for a in range(len(self.shape[b]))]

    def _reads(self, o, k):
        if _is_broadcast_copy(o) or _copy_maps(o) is not None:
            return k == 0
        if o.name in _LINALG:
            return k < _n_ins(o) or (o.name != "linalg.fill" and _reads_init(o, k))
        if o.name == _TRANSFER:
            return (not _is_load(o)) and k == _sram_slot(o)
        return True

    def _has_readers(self, b):
        return bool(self.readers.get(self.group.find(b)))

    # ---- the analysis ------------------------------------------------------------------------

    def fit(self):
        for _ in range(64):
            if not self._sweep():
                break
        self._cut(self._extents())

    def _sweep(self):
        changed = False
        for o in reversed(self.ops):
            if _walked(o):
                it = self._iteration(o)
                for k, m in enumerate(_maps(o)):
                    if self._reads(o, k):
                        changed |= self._ask(o, k, [None if d is None else it[d] for d in m])
            elif o.name == _TRANSFER:
                if not _is_load(o):
                    changed |= self._ask(o, _sram_slot(o), self._stored(o))
            elif _is_broadcast_copy(o):
                changed |= self._ask(o, 0, None)
            else:
                for k, v in enumerate(o.operands):
                    if _is_tensor(v):
                        changed |= self._ask(o, k, None)
        return changed

    def _ask(self, o, k, extents):
        """The tile `o` takes at operand `k` is read for `extents` of it (None: all of it); True if that
        is news."""
        ref = self.ref_at(o, k)
        if ref is None:
            for b in self._roots_under(o.operands[k]):
                if self._raise(b, self.shape[b]):
                    return True
            return False
        b, axes = ref
        want = self._need(b)
        for i, a in enumerate(axes):
            if a is None:
                continue
            if isinstance(a, tuple):
                if a[0] == MERGED:
                    box = self._box(b, a[1])
                    for x in a[1]:
                        e = None if box is None or x != box[0] or extents is None else extents[i]
                        want[x] = self.shape[b][x] if e is None else max(
                            want[x], min(-(-e // box[1]), self.shape[b][x]))
                if a[0] == LAP:
                    laps = self.shape[b][a[1]] // self.lanes
                    e = laps if extents is None or extents[i] is None else extents[i]
                    want[a[1]] = max(want[a[1]], min(e, laps) * self.lanes)
                continue
            e = self.shape[b][a] if extents is None or extents[i] is None else extents[i]
            want[a] = max(want[a], min(e, self.shape[b][a]))
        return self._raise(b, want)

    def _raise(self, b, want):
        changed = False
        for a, w in enumerate(want):
            t = self.tied.find((b, a))
            if w > self.need.get(t, 0):
                self.need[t] = w
                changed = True
        return changed

    def _box(self, b, merged):
        """(axis, scale) by which a view merging `b`'s axes `merged` (row-major in a lane's bank) is
        cut: its outermost axis but the lane axis, to an extent `scale` times smaller than the view's;
        the other merged axes stay whole. None when they are out of order or the lane axis wraps."""
        lane = 0 if not is_one_bank(self.shape[b]) else None
        if list(merged) != sorted(merged) or lane in merged and self.shape[b][0] > self.lanes:
            return None
        axis = next((x for x in merged if x != lane), None)
        if axis is None:
            return None
        return axis, math.prod(self.shape[b][x] for x in merged if x != axis)

    def _roots_under(self, v):
        """The tiles a value this does not follow is made from: the one each op making it takes in
        place, or its first operand."""
        seen = []
        while _is_tensor(v):
            o = producer(v)
            if o is None or not len(o.operands):
                break
            k = _tied_operand(o, v.result_number)
            k = 0 if k is None else k
            ref = self.ref_at(o, k)
            if ref is not None:
                seen.append(ref[0])
                break
            v = o.operands[k]
        return seen

    def _stored(self, tx):
        """What a store out of the scratchpad takes of its tile, per tile axis."""
        shape = list(_shape(tx.operands[_sram_slot(tx)]))
        if not _whole_tile(tx, self.known):
            return None
        out = list(shape)
        for a, (lo, hi) in _clamps(tx).items():
            l, h = interval(lo, self.known), interval(hi, self.known)
            if l == (0, 0) and h is not None:
                out[a] = max(0, min(shape[a], h[1]))
        return out

    def _iteration(self, o):
        """How far `o` has to run along each of its dims for its readers."""
        maps = _maps(o)
        n_in = _n_ins(o)
        full = _iter_shape(o, maps)
        outs = list(range(n_in, len(o.operands)))
        refs = {k: self.ref_at(o, k) for k in outs}
        if not any(r is not None and self._has_readers(r[0]) for r in refs.values()):
            return full
        it = [0] * len(full)
        for k in outs:
            if refs[k] is None:
                return full
            b, axes = refs[k]
            need = self._need(b)
            for i, d in enumerate(maps[k]):
                if d is not None:
                    a = axes[i]
                    box = self._box(b, a[1]) if isinstance(a, tuple) and a[0] == MERGED else None
                    it[d] = max(it[d], need[box[0]] * box[1] if box
                                else full[d] if a is None or (isinstance(a, tuple) and a[0] != LAP)
                                else -(-need[a[1]] // self.lanes) if isinstance(a, tuple)
                                else need[a])
        for d in self._reduced(o):
            it[d] = self._reduce_reach(o, d, maps, full[d])
        lane = maps[-1][0] if maps[-1] and not self._uniform(o, len(o.operands) - 1) else None
        vec = max((d for m in maps for d in m if d is not None), default=None)
        for d in range(len(it)):
            if d == lane:
                it[d] = self._lane_cut(it[d], full[d])
            elif it[d] <= 0 or it[d] >= full[d]:
                it[d] = full[d]
            elif d == vec and it[d] % _VECTOR_GRAIN:
                it[d] = min(full[d], -(-it[d] // _VECTOR_GRAIN) * _VECTOR_GRAIN)
        return it

    def _uniform(self, o, k):
        """Is the tile `o` takes at `k` held by one lane: its key's shape one bank, whatever a view says."""
        ref = self.ref_at(o, k)
        under = [ref[0]] if ref is not None else self._roots_under(o.operands[k])
        return is_one_bank(self.shape[under[0]] if under else list(_shape(o.operands[k])))

    def _lane_cut(self, want, full):
        """The extent a lane axis is cut to: `want` itself when it is a whole number of rows a
        lane, every lane full and the same depth; else the axis whole. A cut that leaves masked
        rows (327 of 384, 3 a lane) moves its partial lanes differently from the tile it was
        (327 of 512, 4 a lane): the masked load's rows then land in other lanes than the ops read
        them from."""
        n = self.lanes
        if not n or full <= n or want <= 0 or want % n or want >= full:
            return full
        return want

    def _reduced(self, o):
        if o.name == _DOT:
            maps = _maps(o)
            return sorted({d for m in maps[:-1] for d in m if d is not None} - set(maps[-1]))
        if o.name == "linalg.reduce":
            return list(int_array(o.attributes["dimensions"]))
        if o.name == "linalg.generic":
            from mlir.ir import ArrayAttr
            return [d for d, t in enumerate(ArrayAttr(o.attributes["iterator_types"]))
                    if t == reduction_iterator()]
        return []

    def _reduce_reach(self, o, d, maps, full):
        """Where the inputs of a reduce over `d` stop holding data: past it they hold the
        combiner's identity, so the reduce need not read them; `full` when that is not known."""
        if o.name == _DOT:
            return self._dot_reach(o, d, maps, full)
        if o.name == "linalg.generic":
            return self._generic_reach(o, d, full)
        if o.name != "linalg.reduce" or len(o.operands) != 2:
            return full
        body = [x.operation for x in o.regions[0].blocks[0].operations]
        if len(body) != 2 or body[0].name not in _IDENTITY:
            return full
        ref = self.ref_at(o, 0)
        if ref is None:
            return full
        a = ref[1][maps[0].index(d)]
        if isinstance(a, tuple) and a[0] != LAP:
            return full
        pad = self._padding(o.operands[0], ref[0], a[1] if isinstance(a, tuple) else a)
        if pad is None or not _IDENTITY[body[0].name](pad[1]):
            return full
        if isinstance(a, tuple):
            return pad[0] // self.lanes if pad[0] % self.lanes == 0 else full
        return pad[0]

    def _dot_reach(self, o, d, maps, full):
        """Where a matmul's product over `d` is 0: past where either input's padding along `d` holds 0."""
        reach = full
        for k in range(_n_ins(o)):
            ref = self.ref_at(o, k)
            if ref is None or d not in maps[k]:
                continue
            a = ref[1][maps[k].index(d)]
            if a is None or isinstance(a, tuple):
                continue
            pad = self._padding(o.operands[k], ref[0], a)
            if pad is not None and pad[1] == 0:
                reach = min(reach, pad[0])
        return reach

    def _generic_reach(self, o, d, full):
        """A reducing generic whose body combines `select(index d < C, x, identity)` into its init
        reads nothing past C along `d`."""
        body = o.regions[0].blocks[0]
        steps = [x.operation for x in body.operations]
        if len(steps) < 2 or steps[-1].name != "linalg.yield":
            return full
        comb = steps[-2]
        acc = body.arguments[_n_ins(o)]
        if comb.name not in _IDENTITY or len(comb.operands) != 2 or acc not in list(comb.operands):
            return full
        x = comb.operands[1] if comb.operands[0] == acc else comb.operands[0]
        sel = producer(x)
        if sel is None or sel.name != "arith.select":
            return full
        value, cut = _float_const(sel.operands[2]), _mask_cut(sel.operands[0], d)
        if value is None or cut is None or not _IDENTITY[comb.name](value):
            return full
        return min(cut, full)

    def _padding(self, v, b, axis):
        """(extent, value): past `extent` along axis `axis` of tile `b` the op that wrote `v` (through
        views) leaves `value`: a masked load that fills, or a generic that selects."""
        while producer(v) is not None and producer(v).name in _VIEWS:
            v = producer(v).operands[0]
        w = producer(v)
        ref = self.root(v) if w is not None else None
        if ref is None or ref[0] != b or axis not in ref[1]:
            return None
        j = ref[1].index(axis)
        if w.name == _TRANSFER:
            if not _whole_tile(w, self.known) or "masked_fill" not in w.attributes:
                return None
            if "dyn_fill" in w.attributes:
                return None
            got = _clamps(w).get(j)
            if got is None:
                return None
            lo, hi = interval(got[0], self.known), interval(got[1], self.known)
            if lo != (0, 0) or hi is None:
                return None
            from mlir.ir import IntegerAttr, ShapedType
            bits = IntegerAttr(w.attributes["masked_fill"]).value
            value = _fill_value(bits, ShapedType(v.type).element_type)
            if value is None:
                return None
            return min(hi[1], self.shape[b][axis]), value
        if w.name == "linalg.generic":
            d = _maps(w)[_n_ins(w) + v.result_number][j]
            ys = [x.operation for x in w.regions[0].blocks[0].operations][-1]
            sel = producer(ys.operands[v.result_number])
            if d is None or sel is None or sel.name != "arith.select":
                return None
            value = _float_const(sel.operands[2])
            cut = _mask_cut(sel.operands[0], d)
            if value is None or cut is None:
                return None
            return min(cut, self.shape[b][axis]), value
        return None

    # ---- cutting -----------------------------------------------------------------------------

    def _extents(self):
        """{key: new shape}: axes one op walks with one dim, or a region carries as one, form a class cut to
        the most any needs (a merged view axis joins by its _box). Kept, and logged: walked with a view not
        followed, pinned (_tie), disagreeing on extent or scale, or with no box; a lane axis: _lane_cut."""
        from fractions import Fraction
        parent, rel, why, clash = {}, {}, {}, []

        def find(x):
            """(root, f): one element of `x` is f of `root`'s."""
            f = Fraction(1)
            while parent.setdefault(x, x) != x:
                f *= rel[x]
                x = parent[x]
            return x, f

        def join(x, sx, y, sy):
            (rx, fx), (ry, fy) = find(x), find(y)
            r = Fraction(sx * fy) / (sy * fx)
            if rx == ry:
                if r != 1:
                    clash.append(x)
            else:
                parent[rx], rel[rx] = ry, r

        for b, shape in self.shape.items():
            for a in range(len(shape)):
                find((b, a))
                if self.tied.find((b, a)) != (b, a):
                    join((b, a), 1, self.tied.find((b, a)), 1)
                if b in self.pinned:
                    why.setdefault((b, a), []).append(self.pinned[b])
        for o in self.ops:
            if not _walked(o):
                continue
            by_dim, kept = {}, {}
            for k, m in enumerate(_maps(o)):
                v = o.operands[k]
                ref = self.ref_at(o, k)
                for i, d in enumerate(m):
                    if d is None or not _is_tensor(v):
                        continue
                    a = None if ref is None else ref[1][i]
                    box = (self._box(ref[0], a[1]) if isinstance(a, tuple) and a[0] == MERGED
                           else None)
                    if box is not None:
                        by_dim.setdefault(d, []).append(((ref[0], box[0]), box[1]))
                    elif ref is None or isinstance(a, tuple) and a[0] == MERGED:
                        if _shape(v)[i] != 1:
                            kept.setdefault(d, []).append(
                                f"{o.name} walks dim {d} through {v.type}, a view this does not cut")
                    elif a is not None and not isinstance(a, tuple):
                        by_dim.setdefault(d, []).append(((ref[0], a), 1))
            for d, keys in by_dim.items():
                for k, sc in keys[1:]:
                    join(k, sc, *keys[0])
                for w in kept.get(d, []):
                    why.setdefault(keys[0][0], []).append(w)
        classes = {}
        for key in parent:
            root, f = find(key)
            classes.setdefault(root, []).append((key, f))
        out = {b: list(s) for b, s in self.shape.items()}
        for root, keys in classes.items():
            unit = math.lcm(*(f.denominator for _, f in keys))
            members = [(b, a, int(f * unit)) for (b, a), f in keys]
            reasons = [w for k in why if find(k)[0] == root for w in why[k]]
            if reasons:
                self._keep(members, "; ".join(dict.fromkeys(reasons)))
                continue
            if any(find(k)[0] == root for k in clash):
                self._keep(members, "its axes are walked at different scales")
                continue
            sizes = {self.shape[b][a] * sc for b, a, sc in members}
            if len(sizes) != 1:
                if unit != 1 or any(sc != 1 for _, _, sc in members):
                    self._keep(members, f"its axes walk different extents {sorted(sizes)}")
                continue
            full = sizes.pop()
            need = max(self._need(b)[a] * sc for b, a, sc in members)
            step = math.lcm(*(sc * (_VECTOR_GRAIN if a == len(self.shape[b]) - 1 else 1)
                              for b, a, sc in members))
            want = -(-need // step) * step
            if any(a == 0 and not is_one_bank(self.shape[b]) for b, a, _ in members):
                want = self._lane_cut(want, full)
            if 0 < want < full:
                for b, a, sc in members:
                    out[b][a] = want // sc
            elif any(sc != 1 for _, _, sc in members):
                least = min(self._need(b)[a] * sc for b, a, sc in members)
                if 0 < least < full:
                    self._keep(members, f"{least} of {full} has no box in steps of {step}")
        return out

    def _keep(self, members, reason):
        """Log that the class of `members` keeps its extent, and why, when a member needs less."""
        if any(self._need(b)[a] < self.shape[b][a] for b, a, _ in members):
            LOG.warning("%s keeps %s whole: %s", NAME, ", ".join(
                f"axis {a} of {b[0].type}" for b, a, _ in members), reason)

    def _cut(self, shapes):
        """Every value of a cut tile retyped, its views' output shapes with it; a fresh tensor whose uses
        are cut apart made again for each extent but its first use's."""
        from mlir.ir import DenseI64ArrayAttr, RankedTensorType
        cut = {b: s for b, s in shapes.items() if s != self.shape[b]}
        if cut:
            plan = []
            for v in [r for o in walk(self.func.operation) for r in o.results] + [
                    a for o in walk(self.func.operation) if o.name in _REGIONS
                    for r in o.regions for blk in r.blocks for a in blk.arguments]:
                ref = self.root(v)
                if ref is None or ref[0] not in cut:
                    continue
                s = _view_shape(list(_shape(v)), ref[1], cut[ref[0]], self.lanes)
                if s != list(_shape(v)):
                    plan.append((v, s))
            for e, uses in list(self.fresh.items()):
                by_shape = {}
                for n, use in enumerate(uses):
                    by_shape.setdefault(tuple(cut.get((e, n), self.shape[(e, n)])), []).append(n)
                for i, (s, ns) in enumerate(by_shape.items()):
                    plan.append((self._split(e, ns) if i else e, list(s)))
            for v, s in plan:
                t = RankedTensorType(v.type)
                v.set_type(RankedTensorType.get(s, t.element_type, t.encoding))
                o = producer(v)
                if o is not None and "static_output_shape" in o.attributes:
                    o.attributes["static_output_shape"] = DenseI64ArrayAttr.get(s)
        self.refs = {}
        self._fold_bodies()
        self.func.operation.verify()

    def _split(self, e, ns):
        """A copy of fresh tensor `e` for its uses `ns`, each now the copy's, its tile the same."""
        from mlir.ir import InsertionPoint
        made = producer(e).clone(ip=InsertionPoint(producer(e))).results[0]
        self.fresh[made] = []
        for n in ns:
            o, k = self.fresh[e][n]
            o.operands[k] = made
            new = (made, len(self.fresh[made]))
            self.fresh[made].append((o, k))
            self.shape[new] = list(self.shape[(e, n)])
            self.group.join(new, (e, n))
            for a in range(len(self.shape[new])):
                self.tied.join((new, a), ((e, n), a))
        return made

    def _fold_bodies(self):
        """Every generic's masks that only cut past its (new) extents folded, its body simplified."""
        from mlir.ir import IntegerAttr, IntegerType, InsertionPoint
        from mlir.dialects import arith
        folded = []
        for o in self.ops:
            if o.name != "linalg.generic":
                continue
            it = _iter_shape(o, _maps(o))
            steps = [x.operation for x in o.regions[0].blocks[0].operations]
            known = list(self.known) + [
                (x.results[0], (0, it[_index_dim(x.results[0])] - 1))
                for x in steps if x.name == "linalg.index"]
            for x in steps:
                if x.name != "arith.cmpi":
                    continue
                pred = IntegerAttr(x.attributes["predicate"]).value
                c = _int_const(x.operands[1])
                lhs = interval(x.operands[0], known)
                if pred in _FOLD_LT and c is not None and lhs is not None and lhs[1] < c:
                    with InsertionPoint(o):
                        t = arith.ConstantOp(IntegerType.get_signless(1), 1).result
                    x.results[0].replace_all_uses_with(t)
                    folded.append(o)
            _simplify(o)
        for o in dict.fromkeys(folded):
            chain = _copy_chain(o)
            if chain is not None:
                self._forward(o, chain)

    def _forward(self, cp, casts):
        """An elementwise generic whose mask folded only copies its input, or casts it (`casts`, in
        order): the ops that read what it made read its input through the same views instead, each cast
        made in their body as they read it, and the generic goes. A generic kept carries COPY_KEPT, why."""
        from mlir.ir import StringAttr

        def keep(why):
            cp.attributes[COPY_KEPT] = StringAttr.get(why)
        self.refs = {}
        src, made = cp.operands[0], cp.results[0]
        mine, theirs = self.ref_at(cp, 1), self.root(src)
        if mine is None or theirs is None:
            return keep("it writes or reads a view this pass does not follow")
        if cp.parent.name != "func.func":
            return keep("it is inside a loop or a branch")
        order = walk(self.func.operation)
        at = {o: k for k, o in enumerate(order)}
        writers = {}
        for o in order:
            for k in range(len(o.operands)):
                ref = self.ref_at(o, k) if _carries(o, k) else None
                if ref is not None:
                    writers.setdefault(self.group.find(ref[0]), []).append(o)
        if any(w.name not in _LINALG and not _is_broadcast_copy(w)
               for w in writers.get(self.group.find(mine[0]), [])):
            return keep("its buffer is also written in part (a transfer or a window)")
        uses = _uses_through_views(made)
        ours = {self.group.find(mine[0]), self.group.find(theirs[0])}
        if any(r.parent != cp.parent or any(_carries(r, j) and self.group.find(
                (self.ref_at(r, j) or (None,))[0]) in ours for j in range(len(r.operands))) for r, _ in uses):
            return keep("a reader is in a loop or a branch, or writes its input or its buffer")
        last = max((at[r] for r, _ in uses), default=at[cp])
        if any(at[cp] < at[w] < last for w in writers.get(self.group.find(theirs[0]), [])):
            return keep("its input is written again before a reader")
        if casts:
            for r, k in uses:
                if r.name not in ("linalg.generic", "linalg.reduce") or k >= _n_ins(r):
                    return keep(f"a cast it makes would go into a {r.name} that does not walk it")
                if r.name == "linalg.reduce" and folds_lanes(r):
                    return keep("a cast it makes would go into a fold across the lanes")
        for r, k in uses:
            r.operands[k] = _view_again(r.operands[k], made, src, r)
            if casts:
                _cast_in_body(r, k, casts)
        _erase_unused_views(made)
        self.ops.remove(cp)
        cp.erase()
        self.refs = {}


# ---- small facts ------------------------------------------------------------------------------

class _Union:
    """A union-find over hashable keys."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        while self.parent.setdefault(x, x) != x:
            x = self.parent[x]
        return x

    def join(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def _is_tensor(v):
    from mlir.ir import RankedTensorType
    return v is not None and isinstance(v.type, RankedTensorType)


def _is_fresh(o):
    """A fresh tensor: one whose elements nothing has written (or are undefined), so bufferize allocates
    for each writer."""
    from mlir.dialects import bufferization, tensor, ub
    fresh = (tensor.EmptyOp, bufferization.AllocTensorOp, ub.PoisonOp)
    return o is not None and isinstance(op_of(o).opview, fresh) and not len(o.operands)


def _terminator(block):
    return block.operations[len(block.operations) - 1].operation


def _tied_operand(o, i):
    """The operand `o` writes in place for its tensor result `i`, or None."""
    if o.name in _LINALG:
        return _n_ins(o) + i
    if o.name in (_DOT, "pytorchsim.gather", "pytorchsim.convert_layout"):
        from ...dialects.pytorchsim import operand_number
        return operand_number(o, {"pytorchsim.convert_layout": "dst"}.get(o.name, "init"))
    if o.name == _TRANSFER:
        return _sram_slot(o)
    if o.name in _INTO_DEST:
        return 1
    return None


def _carries(o, k):
    """Does `o` write the tile it takes at operand `k` in place, or carry it on as a loop's or a branch's?"""
    if o.name in _REGIONS:
        return o.name != "scf.for" or k >= FOR_INITS
    return any(_tied_operand(o, i) == k for i in range(len(o.results)))


def _view_shape(old, axes, cut, lanes):
    """The shape of a value whose axes are `axes` of a tile cut to `cut`."""
    return [old[i] if a is None else lanes if isinstance(a, tuple) and a[0] == LANE
            else math.prod(cut[x] for x in a[1]) if isinstance(a, tuple) and a[0] == MERGED
            else cut[a[1]] // lanes if isinstance(a, tuple) else cut[a]
            for i, a in enumerate(axes)]


def _uses_through_views(v):
    """(op, operand) of each use of `v` but by a view, and of each use of its views'."""
    out = []
    for u in v.uses:
        o = op_of(u.owner)
        if o.name in _VIEWS:
            out += _uses_through_views(o.results[0])
        else:
            out.append((o, u.operand_number))
    return out


def _erase_unused_views(v):
    for u in list(v.uses):
        o = op_of(u.owner)
        if o.name in _VIEWS:
            _erase_unused_views(o.results[0])
            if not list(o.results[0].uses):
                o.erase()


def _is_broadcast_copy(o):
    """A pytorchsim.convert_layout that broadcasts (added_dims) into the whole of its destination."""
    return (o.name == "pytorchsim.convert_layout" and "added_dims" in o.attributes
            and "into_window" not in o.attributes and "static_offsets" not in o.attributes)


def _shape(v):
    from mlir.ir import ShapedType
    return ShapedType(v.type).shape


def _groups(o):
    from mlir.ir import ArrayAttr
    return [list(int_array(g)) for g in ArrayAttr(o.attributes["reassociation"])]


def _walked(o):
    """Whether the analysis walks `o` by its maps: a linalg op, a matmul whose maps cover every axis of its
    operands (a batched one is left to the lowering's refusal), a copy that is no window."""
    if o.name == _DOT:
        maps = indexing_maps(o)
        return maps is not None and all(len(m.results) == len(_shape(v)) for m, v in zip(maps, o.operands))
    return o.name in _LINALG or _copy_maps(o) is not None


def _copy_maps(o):
    """[source dims, destination dims] of a pytorchsim.convert_layout that is no window, its dims the
    destination's logical axes: a broadcast's source axes land on the axes it does not add, a copy's on its
    equal logical axes (unit axes left out); None for a window or a reshape that splits an axis."""
    if o.name != "pytorchsim.convert_layout" or "into_window" in o.attributes or "static_offsets" in o.attributes:
        return None
    sp, dp = list(int_array(o.attributes["src_perm"])), list(int_array(o.attributes["dst_perm"]))
    ss, ds = _shape(o.operands[0]), _shape(o.operands[1])
    if len(sp) != len(ss) or len(dp) != len(ds):
        return None
    src_l, dst_l = [0] * len(ss), [0] * len(ds)
    for s_, l in enumerate(sp):
        src_l[l] = ss[s_]
    for t, l in enumerate(dp):
        dst_l[l] = ds[t]
    if "added_dims" in o.attributes:
        added = set(int_array(o.attributes["added_dims"]))
        kept = [j for j in range(len(dst_l)) if j not in added]
        if [dst_l[j] for j in kept] != src_l:
            return None
        to = dict(enumerate(kept))
    else:
        big_s = [i for i, e in enumerate(src_l) if e != 1]
        big_d = [j for j, e in enumerate(dst_l) if e != 1]
        if [src_l[i] for i in big_s] != [dst_l[j] for j in big_d]:
            return None
        to = dict(zip(big_s, big_d))
    return [[to.get(l) for l in sp], list(dp)]


def _n_ins(o):
    if o.name == "pytorchsim.convert_layout":
        return 1
    if o.name == _DOT:
        return 2
    if o.name == "linalg.fill":
        return 1
    if o.name == "linalg.reduce":
        return len(o.operands) // 2
    return o.attributes["operandSegmentSizes"][0]


def _reads_init(o, k):
    """Whether linalg op `o`'s body reads its `outs` operand `k`."""
    body = o.regions[0].blocks[0]
    return k < len(body.arguments) and bool(list(body.arguments[k].uses))


def _maps(o):
    """Per operand, the iteration dim each of its axes walks (None for a constant 0)."""
    from mlir.ir import AffineConstantExpr, AffineDimExpr
    if o.name == "pytorchsim.convert_layout":
        return _copy_maps(o)
    if o.name in ("linalg.generic", _DOT):
        out = []
        for m in indexing_maps(o):
            out.append([None if isinstance(r, AffineConstantExpr) else
                        r.position if isinstance(r, AffineDimExpr) else -1 for r in m.results])
        return out
    if o.name == "linalg.reduce":
        rank = len(_shape(o.operands[0]))
        red = list(int_array(o.attributes["dimensions"]))
        n = len(o.operands) // 2
        return [list(range(rank))] * n + [[d for d in range(rank) if d not in red]] * n
    return [[], list(range(len(_shape(o.operands[1]))))]


def _iter_shape(o, maps):
    nd = 1 + max((d for m in maps for d in m if d is not None and d >= 0), default=-1)
    ext = [1] * nd
    for v, m in zip(o.operands, maps):
        if not _is_tensor(v):
            continue
        for ax, d in enumerate(m):
            if d is not None and d >= 0 and _shape(v)[ax] != 1:
                ext[d] = _shape(v)[ax]
    return ext


def _is_load(tx):
    from ...dialects.pytorchsim import is_load
    return is_load(tx)


def _sram_slot(tx):
    from ...dialects.pytorchsim import operand_number
    return operand_number(tx, "sram")


def _whole_tile(tx, known):
    """A transfer of its whole scratchpad tile, at its start: no gather, subtile or piece. A cut may stay:
    it says only where in DRAM each element is, and a fit keeps dram_idx and every tile coordinate."""
    from ...dialects.pytorchsim import operand_number
    for name in ("subtile_size", "indirect_dim", "piece_box"):
        if name in tx.attributes:
            return False
    idx = tx.operands[operand_number(tx, "sram_idx")]
    return interval(idx, known) == (0, 0)


def _clamps(tx):
    """{tile axis: (low, high)} of a masked transfer."""
    if "masked_axes" not in tx.attributes:
        return {}
    axes = list(int_array(tx.attributes["masked_axes"]))
    clamp = list(tx.opview.clamp)
    return {a: (clamp[2 * i], clamp[2 * i + 1]) for i, a in enumerate(axes)}


def _copy_chain(o):
    """The casts, in order, by which a generic writes its one input into its one output of the
    same shape, walked alike ([] when it copies it as it is); None for any other generic."""
    from mlir.ir import ArrayAttr
    body = o.regions[0].blocks[0]
    ops_ = [x.operation for x in body.operations]
    maps = _maps(o)
    if not (_n_ins(o) == 1 and len(o.operands) == 2 and len(o.results) == 1
            and ops_[-1].name == "linalg.yield" and _shape(o.operands[0]) == _shape(o.operands[1])
            and maps[0] == maps[1] and sorted(maps[1]) == list(range(len(maps[1])))
            and all(t == parallel_iterator() for t in ArrayAttr(o.attributes["iterator_types"]))):
        return None
    cur = body.arguments[0]
    for x in ops_[:-1]:
        if x.name not in CASTS or len(x.operands) != 1 or x.operands[0] != cur:
            return None
        cur = x.results[0]
    if ops_[-1].operands[0] != cur:
        return None
    if not ops_[:-1] and o.operands[0].type != o.operands[1].type:
        return None
    return ops_[:-1]


def _view_again(v, made, src, at):
    """`v`, a view of `made` (or `made` itself), made again on `src` before `at`: the same reshapes,
    of `src`'s element type."""
    from mlir.ir import InsertionPoint, Operation, RankedTensorType
    chain = []
    while v != made:
        chain.append(producer(v))
        v = chain[-1].operands[0]
    y = src
    elt = RankedTensorType(src.type).element_type
    with InsertionPoint(at):
        for p in reversed(chain):
            old = RankedTensorType(p.results[0].type)
            ty = RankedTensorType.get(list(old.shape), elt, old.encoding)
            y = Operation.create(p.name, results=[ty], operands=[y],
                                 attributes=attrs_of(p)).results[0]
    return y


def _cast_in_body(r, k, casts):
    """Reader `r`'s body takes its operand `k` as the casts' input and makes the casts first."""
    from mlir.ir import InsertionPoint, Operation, ShapedType
    body = r.regions[0].blocks[0]
    arg = body.arguments[k]
    arg.set_type(ShapedType(r.operands[k].type).element_type)
    with InsertionPoint.at_block_begin(body):
        cur, made = arg, []
        for c in casts:
            made.append(Operation.create(c.name, results=[c.results[0].type], operands=[cur]))
            cur = made[-1].results[0]
    arg.replace_all_uses_with(cur)
    made[0].operands[0] = arg


def _simplify(gen):
    """A generic's body after its masks folded: a select on true is its value, an add of +0.0 its
    other operand (a zero-filled accumulator fusion folded into a constant), unused arith ops and
    indices erased."""
    body = gen.regions[0].blocks[0]
    for x in [x.operation for x in body.operations]:
        if x.name == "arith.select" and _int_const(x.operands[0]) in (1, -1):
            x.results[0].replace_all_uses_with(x.operands[1])
        elif x.name == "arith.addf":
            for i in (0, 1):
                z = _float_const(x.operands[i])
                if z == 0.0 and math.copysign(1.0, z) > 0:
                    x.results[0].replace_all_uses_with(x.operands[1 - i])
                    break
    _erase_dead(body)


def _erase_dead(block):
    """The body's arith ops and linalg.index left with no use (a folded mask's compare and what fed
    it), erased until none is."""
    while True:
        dead = [x.operation for x in block.operations
                if (x.operation.name.startswith("arith.") or x.operation.name == "linalg.index")
                and x.operation.results and not any(list(r.uses) for r in x.operation.results)]
        if not dead:
            return
        for x in dead:
            x.erase()


def _fill_value(bits, elem):
    """A transfer's `masked_fill` -- the element's bit pattern -- as the number it is."""
    import struct
    from mlir.ir import BF16Type, F16Type, F32Type, IntegerType
    if isinstance(elem, F32Type):
        return struct.unpack("<f", (bits & 0xFFFFFFFF).to_bytes(4, "little"))[0]
    if isinstance(elem, BF16Type):
        return struct.unpack("<f", ((bits & 0xFFFF) << 16).to_bytes(4, "little"))[0]
    if isinstance(elem, F16Type):
        return struct.unpack("<e", (bits & 0xFFFF).to_bytes(2, "little"))[0]
    if isinstance(elem, IntegerType):
        w = IntegerType(elem).width
        v = bits & ((1 << w) - 1)
        return float(v - (1 << w) if v >> (w - 1) else v)
    return None


def _int_const(v):
    c = const_int(v)
    if c is not None:
        return c
    o = producer(v)
    if o is not None and o.name in ("arith.index_cast", "arith.extsi", "arith.extui"):
        return _int_const(o.operands[0])
    return None


def _float_const(v):
    from mlir.ir import FloatAttr, IntegerAttr
    o = producer(v)
    if o is None or o.name != "arith.constant":
        return None
    a = o.attributes["value"]
    try:
        return FloatAttr(a).value
    except ValueError:
        try:
            return float(IntegerAttr(a).value)
        except ValueError:
            return None


def _index_dim(v):
    """The iteration dim `v` is, when it is `linalg.index d` (through an index cast)."""
    o = producer(v)
    while o is not None and o.name in ("arith.index_cast", "arith.index_castui"):
        o = producer(o.operands[0])
    if o is None or o.name != "linalg.index":
        return None
    from mlir.ir import IntegerAttr
    return IntegerAttr(o.attributes["dim"]).value


def _mask_cut(cond, d):
    """C when `cond` is false for every index past C along dim `d`: `linalg.index d < C`, or an
    `and` with such a compare on one side."""
    from mlir.ir import IntegerAttr
    o = producer(cond)
    if o is None:
        return None
    if o.name == "arith.andi":
        cuts = [c for c in (_mask_cut(o.operands[0], d), _mask_cut(o.operands[1], d))
                if c is not None]
        return min(cuts) if cuts else None
    if o.name != "arith.cmpi":
        return None
    pred = IntegerAttr(o.attributes["predicate"]).value
    if pred not in _FOLD_LT or _index_dim(o.operands[0]) != d:
        return None
    return _int_const(o.operands[1])
