"""pytorchsim-fit-to-numel: every scratchpad buffer is cut, by its type, to the extent its readers need (768
of a numel-768 block of 1024), worked backwards from the stores; a mask that only cuts past the new extent
folds to true."""

import logging
import math

from .lib_bank_layout import is_a_scratchpad_buffer, is_one_bank, is_uniform
from .lib_ir import (attrs_of, const_int, ends_lifetime, funcs, indexing_maps, int_array,
                     is_memref, lanes_of_module, op_of, parallel_iterator, producer, walk)
from .lib_live import interval, program_ids
from .p03_fuse import CASTS, folds_lanes

NAME = "pytorchsim-fit-to-numel"
LOG = logging.getLogger(__name__)

_LINALG = ("linalg.generic", "linalg.reduce", "linalg.fill")
_DOT = "torchsim.dot"

# a view axis that is part of a buffer's lane axis split into [lanes, laps]: (LANE, axis) or (LAP, axis);
# a view axis a collapse merged from several buffer axes, in order: (MERGED, (axis, ...))
LANE, LAP, MERGED = "lane", "lap", "merged"
_VIEWS = ("memref.expand_shape", "memref.collapse_shape")
_TRANSFER = "torchsim.transfer"
_FOLD_LT = (2, 6)            # arith.cmpi predicates slt, ult

# combiner -> the padding value that leaves it unchanged
_IDENTITY = {
    "arith.addf": lambda v: v == 0.0, "arith.addi": lambda v: v == 0,
    "arith.mulf": lambda v: v == 1.0, "arith.muli": lambda v: v == 1,
    "arith.maxnumf": lambda v: v == -math.inf, "arith.maximumf": lambda v: v == -math.inf,
    "arith.minnumf": lambda v: v == math.inf, "arith.minimumf": lambda v: v == math.inf,
}

# on an elementwise copy whose mask folded and that stays: why its readers do not read its input
COPY_KEPT = "torchsim.copy_kept_because"

# a last axis is cut only to a multiple of this
_VECTOR_GRAIN = 8


def run(module, grid=(), **_opts):
    lanes = lanes_of_module(module)
    for func in funcs(module):
        _Live(func, grid, lanes).fit()
    return module


class _Live:

    def __init__(self, func, grid, lanes):
        self.func = func
        self.lanes = lanes
        self.known = program_ids(func, grid)
        self.ops = [o for o in walk(func.operation)
                    if _walked(o) or o.name == _TRANSFER or any(
                        is_memref(v) and not ends_lifetime(o, v) for v in o.operands)]
        self.ops = [o for o in self.ops if o.name not in _VIEWS and o.name != "memref.get_global"]
        self.shape = {}          # root buffer -> shape
        self.need = {}           # root buffer -> per-axis extent its readers need
        self.writers = {}        # root buffer -> ops that write it
        self.readers = {}        # root buffer -> ops that read it
        self.iters = {}          # linalg op -> per-iteration-dim extent
        for b in [r for o in walk(func.operation) for r in o.results if is_a_scratchpad_buffer(r)]:
            self.shape[b] = list(_shape(b))
            self.need[b] = [0] * len(self.shape[b])
            self.writers[b], self.readers[b] = [], []
        for o in self.ops:
            for k, v in enumerate(o.operands):
                ref = self.root(v)
                if ref is None:
                    continue
                b = ref[0]
                if self._writes(o, k):
                    self.writers[b].append(o)
                if self._reads(o, k):
                    self.readers[b].append(o)

    # ---- buffers and views -------------------------------------------------------------------

    def root(self, v):
        """(buffer, axis map) of a scratchpad value: `axes[i]` is the buffer axis `v`'s axis i
        is, None for a unit axis a reshape added, (MERGED, axes) for several a collapse merged. None
        when `v` is not a buffer this follows."""
        if not is_memref(v):
            return None
        o = producer(v)
        if o is None:
            return None
        if is_a_scratchpad_buffer(v):
            return (v, list(range(len(_shape(v))))) if v in self.shape else None
        if o.name in _VIEWS:
            up = self.root(o.operands[0])
            if up is None:
                return None
            src, groups = _shape(o.operands[0]), _groups(o)
            out = list(_shape(v))
            axes = [None] * len(out)
            expand = o.name == "memref.expand_shape"
            for s, g in enumerate(groups):
                big = [i for i in g if (out[i] if expand else src[i]) != 1]
                if len(big) > 1 and expand and len(big) == 2 and isinstance(up[1][s], int) \
                        and up[1][s] == 0 and out[big[0]] == self.lanes \
                        and not is_one_bank(self.shape[up[0]]):
                    # a lane axis deeper than the lanes viewed as [lanes, laps]: the lanes are
                    # all of the machine's, the laps what of the axis a lane holds
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
                else:
                    if big:
                        axes[s] = up[1][big[0]]
            return up[0], axes
        return None

    def _writes(self, o, k):
        if _is_broadcast_copy(o) or _copy_maps(o) is not None:
            return k == 1
        if o.name in _LINALG:
            return k >= _n_ins(o)
        if o.name == _DOT and _walked(o):
            return k == _n_ins(o)
        if o.name == _TRANSFER:
            return _is_load(o) and k == _sram_slot(o)
        return True

    def _reads(self, o, k):
        if _is_broadcast_copy(o) or _copy_maps(o) is not None:
            return k == 0
        if o.name in _LINALG:
            return k < _n_ins(o) or (o.name != "linalg.fill" and _reads_init(o, k))
        if o.name == _TRANSFER:
            return (not _is_load(o)) and k == _sram_slot(o)
        return True

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
                self.iters[o] = it
                for k, (v, m) in enumerate(zip(o.operands, _maps(o))):
                    if self._reads(o, k):
                        changed |= self._ask(v, [None if d is None else it[d] for d in m])
            elif o.name == _TRANSFER:
                if not _is_load(o):
                    v = o.operands[_sram_slot(o)]
                    changed |= self._ask(v, self._stored(o))
            elif _is_broadcast_copy(o):
                # a broadcast writes its whole destination and reads all of its source: the
                # destination's extent is what its own readers need
                changed |= self._ask(o.operands[0], None)
            else:
                for v in o.operands:
                    changed |= self._ask(v, None)
        return changed

    def _ask(self, v, extents):
        """`v`'s readers need `extents` of it (None: all of it); True if that is news."""
        ref = self.root(v)
        if ref is None:
            if is_memref(v):
                for b in self._roots_under(v):
                    changed = self._raise(b, self.shape[b])
                    if changed:
                        return True
            return False
        b, axes = ref
        want = list(self.need[b])
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
        got = [max(x, y) for x, y in zip(self.need[b], want)]
        if got != self.need[b]:
            self.need[b] = got
            return True
        return False

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
        """The buffers a value this does not follow may alias: all of them that reach it."""
        o = producer(v)
        seen = []
        while o is not None and is_memref(o.operands[0] if o.operands else None):
            src = o.operands[0]
            if is_a_scratchpad_buffer(src):
                if src in self.shape:
                    seen.append(src)
                break
            o = producer(src)
        return seen

    def _stored(self, tx):
        """What a store out of the scratchpad takes of its tile, per tile axis."""
        view = tx.operands[_sram_slot(tx)]
        shape = list(_shape(view))
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
        if not any(self.readers.get(self.root(o.operands[k])[0] if self.root(o.operands[k])
                                    else None) for k in outs):
            return full
        it = [0] * len(full)
        for k in outs:
            ref = self.root(o.operands[k])
            if ref is None:
                return full
            b, axes = ref
            for i, d in enumerate(maps[k]):
                if d is not None:
                    a = axes[i]
                    box = self._box(b, a[1]) if isinstance(a, tuple) and a[0] == MERGED else None
                    it[d] = max(it[d], self.need[b][box[0]] * box[1] if box
                                else full[d] if a is None or (isinstance(a, tuple) and a[0] != LAP)
                                else -(-self.need[b][a[1]] // self.lanes) if isinstance(a, tuple)
                                else self.need[b][a])
        for d in self._reduced(o):
            it[d] = self._reduce_reach(o, d, maps, full[d])
        lane = maps[-1][0] if maps[-1] and not is_uniform(o.operands[-1]) else None
        vec = max((d for m in maps for d in m if d is not None), default=None)
        for d in range(len(it)):
            if d == lane:
                it[d] = self._lane_cut(it[d], full[d])
            elif it[d] <= 0 or it[d] >= full[d]:
                it[d] = full[d]
            elif d == vec and it[d] % _VECTOR_GRAIN:
                it[d] = min(full[d], -(-it[d] // _VECTOR_GRAIN) * _VECTOR_GRAIN)
        return it

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
                    if "reduction" in str(t)]
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
        ref = self.root(o.operands[0])
        if ref is None:
            return full
        b, axes = ref
        a = axes[maps[0].index(d)]
        if isinstance(a, tuple) and a[0] != LAP:
            return full
        pad = self._padding(b, a[1] if isinstance(a, tuple) else a, o)
        if pad is None or not _IDENTITY[body[0].name](pad[1]):
            return full
        if isinstance(a, tuple):
            return pad[0] // self.lanes if pad[0] % self.lanes == 0 else full
        return pad[0]

    def _dot_reach(self, o, d, maps, full):
        """Where a matmul's product over `d` is 0: past where either input's padding along `d` holds 0."""
        reach = full
        for k in range(_n_ins(o)):
            ref = self.root(o.operands[k])
            if ref is None or d not in maps[k]:
                continue
            a = ref[1][maps[k].index(d)]
            if a is None or isinstance(a, tuple):
                continue
            pad = self._padding(ref[0], a, o)
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

    def _padding(self, b, axis, reader=None):
        """(extent, value): past `extent` along `axis` the writer of `b` leaves `value` -- its one
        writer, or the last one before `reader` in its block when all of them stand there."""
        ws = self.writers[b]
        if len(ws) != 1:
            blk = None if reader is None else reader.operation.block
            if blk is None or any(w.operation.block != blk for w in ws):
                return None
            before = [w for w in ws if w.operation.is_before_in_block(reader.operation)]
            if not before:
                return None
            ws = [max(before, key=lambda w: [x.operation for x in blk.operations].index(w.operation))]
        w = ws[0]
        if w.name == _TRANSFER:
            if not _whole_tile(w, self.known) or "masked_fill" not in w.attributes:
                return None
            if "dyn_fill" in w.attributes:
                return None
            got = _clamps(w).get(axis)
            if got is None:
                return None
            lo, hi = interval(got[0], self.known), interval(got[1], self.known)
            if lo != (0, 0) or hi is None:
                return None
            from mlir.ir import IntegerAttr, MemRefType
            bits = IntegerAttr(w.attributes["masked_fill"]).value
            value = _fill_value(bits, MemRefType(b.type).element_type)
            if value is None:
                return None
            return min(hi[1], self.shape[b][axis]), value
        if w.name == "linalg.generic":
            maps = _maps(w)
            k = list(w.operands).index(b) if b in list(w.operands) else None
            if k is None or k < _n_ins(w):
                return None
            d = maps[k][axis]
            body = w.regions[0].blocks[0]
            ys = [x.operation for x in body.operations][-1]
            y = ys.operands[k - _n_ins(w)]
            sel = producer(y)
            if sel is None or sel.name != "arith.select":
                return None
            value = _float_const(sel.operands[2])
            cut = _mask_cut(sel.operands[0], d)
            if value is None or cut is None:
                return None
            return min(cut, self.shape[b][axis]), value
        return None

    # ---- cutting -----------------------------------------------------------------------------

    def _extents(self):
        """{buffer: new shape}: the axes one op walks with one dim form a class, cut to the most any
        of them needs. A merged view axis joins by its box's axis, each element of which is `scale`
        of the class's (_box). A class is kept, and logged, when walked with a view this does not
        follow, when its axes disagree on their extent or scale, or when it has no box below its
        extent; one with a lane axis is cut only by whole lanes (_lane_cut)."""
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
        for o in self.ops:
            if not _walked(o):
                continue
            by_dim, kept = {}, {}
            for v, m in zip(o.operands, _maps(o)):
                ref = self.root(v)
                for i, d in enumerate(m):
                    if d is None or not is_memref(v):
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
            need = max(self.need[b][a] * sc for b, a, sc in members)
            step = math.lcm(*(sc * (_VECTOR_GRAIN if a == len(self.shape[b]) - 1 else 1)
                              for b, a, sc in members))
            want = -(-need // step) * step
            if any(a == 0 and not is_one_bank(self.shape[b]) for b, a, _ in members):
                want = self._lane_cut(want, full)
            if 0 < want < full:
                for b, a, sc in members:
                    out[b][a] = want // sc
            elif any(sc != 1 for _, _, sc in members):
                least = min(self.need[b][a] * sc for b, a, sc in members)
                if 0 < least < full:
                    self._keep(members, f"{least} of {full} has no box in steps of {step}")
        return out

    def _keep(self, members, reason):
        """Log that the class of `members` keeps its extent, and why, when a member needs less."""
        if any(self.need[b][a] < self.shape[b][a] for b, a, _ in members):
            LOG.warning("%s keeps %s whole: %s", NAME, ", ".join(
                f"axis {a} of {b.type}" for b, a, _ in members), reason)

    def _cut(self, shapes):
        from mlir.ir import (DenseI64ArrayAttr, FlatSymbolRefAttr, IntegerType, InsertionPoint,
                             MemRefType, TypeAttr)
        from mlir.dialects import arith
        cut = {b: s for b, s in shapes.items() if s != self.shape[b]}
        if not cut:
            self._fold_bodies()
            return
        module = self.func.operation
        while module.name != "builtin.module":
            module = op_of(module.parent)
        globals_ = {str(o.attributes["sym_name"]).strip('"'): o for o in walk(module)
                    if o.name == "memref.global"}
        for b, s in cut.items():
            ty = MemRefType(b.type)
            new = MemRefType.get(s, ty.element_type, memory_space=ty.memory_space)
            if producer(b).name != "memref.get_global":
                b.set_type(new)
                continue
            name = FlatSymbolRefAttr(producer(b).attributes["name"]).value
            g = globals_[name]
            g.attributes["type"] = TypeAttr.get(new)
            b.set_type(new)
        for o in walk(self.func.operation):
            if o.name not in _VIEWS:
                continue
            ref = self.root(o.results[0])
            if ref is None or ref[0] not in cut:
                continue
            b, axes = ref
            old = MemRefType(o.results[0].type)
            s = [old.shape[i] if a is None else self.lanes if isinstance(a, tuple) and a[0] == LANE
                 else math.prod(cut[b][x] for x in a[1]) if isinstance(a, tuple) and a[0] == MERGED
                 else cut[b][a[1]] // self.lanes if isinstance(a, tuple) else cut[b][a]
                 for i, a in enumerate(axes)]
            o.results[0].set_type(MemRefType.get(s, old.element_type,
                                                 memory_space=old.memory_space))
            if "static_output_shape" in o.attributes:
                o.attributes["static_output_shape"] = DenseI64ArrayAttr.get(s)
        self._fold_bodies()
        self.func.operation.verify()

    def _fold_bodies(self):
        """Every generic's masks that only cut past its (new) extents folded, its body simplified."""
        from mlir.ir import IntegerType, InsertionPoint
        from mlir.dialects import arith
        folded = []
        for o in self.ops:
            if o.name != "linalg.generic":
                continue
            it = _iter_shape(o, _maps(o))
            steps = [x.operation for x in o.regions[0].blocks[0].operations]
            # what each index of the body can be now: below its dim's extent; program ids by grid
            known = list(self.known) + [
                (x.results[0], (0, it[_index_dim(x.results[0])] - 1))
                for x in steps if x.name == "linalg.index"]
            for x in steps:
                if x.name != "arith.cmpi":
                    continue
                pred = int(str(x.attributes["predicate"]).split(":")[0])
                c = _int_const(x.operands[1])
                lhs = interval(x.operands[0], known)
                if pred in _FOLD_LT and c is not None and lhs is not None and lhs[1] < c:
                    # made ahead of the op, not in its body: vectorize widens a body's constants
                    # by their type, and an i1 widened so is no constant it can spell
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
        order): the ops that read what it wrote (after it in its block, before the buffer is
        written again) read its input through the same views instead, each cast made in their body
        as they read it, and the generic goes. A generic kept carries COPY_KEPT, why."""
        from mlir.ir import StringAttr

        def keep(why):
            cp.attributes[COPY_KEPT] = StringAttr.get(why)
        src, dst = cp.operands[0], cp.operands[1]
        if not is_a_scratchpad_buffer(dst) or dst not in self.writers or self.root(src) is None:
            return keep("it writes or reads a view this pass does not follow")
        whole = ("linalg.generic", "linalg.reduce", "linalg.fill")
        if any(w.name not in whole and not _is_broadcast_copy(w) for w in self.writers[dst]):
            return keep("its buffer is also written in part (a transfer or a window)")
        if cp.parent.name != "func.func":
            return keep("it is inside a loop or a branch")
        at = {o: k for k, o in enumerate(self.ops)}
        k0 = at[cp]
        nxt = min((at[w] for w in self.writers[dst] if at[w] > k0), default=len(self.ops))
        src_root = self.root(src)[0]
        readers = [r for r in dict.fromkeys(self.readers[dst]) if k0 < at[r] < nxt]
        if any(r.parent != cp.parent or r in self.writers[dst] or r in self.writers[src_root]
               for r in readers):
            return keep("a reader is in a loop or a branch, or writes its input or its buffer")
        last = max((at[r] for r in readers), default=k0)
        if any(k0 < at[w] < last for w in self.writers[src_root]):
            return keep("its input is written again before a reader")
        if self._freed_between(src_root, cp, self.ops[last] if readers else cp):
            return keep("its input's life ends before a reader")
        uses = [(r, k) for r in readers for k, v in enumerate(r.operands)
                if (self.root(v) or (None,))[0] == dst]
        if casts:
            for r, k in uses:
                if r.name not in ("linalg.generic", "linalg.reduce") or k >= _n_ins(r):
                    return keep(f"a cast it makes would go into a {r.name} that does not walk it")
                if r.name == "linalg.reduce" and folds_lanes(r):
                    return keep("a cast it makes would go into a fold across the lanes")
        for r, k in uses:
            r.operands[k] = _view_again(r.operands[k], dst, src, r)
            if casts:
                _cast_in_body(r, k, casts)
        for r in readers:
            self.readers[dst] = [x for x in self.readers[dst] if x != r]
            self.readers[src_root].append(r)
        for book in (self.writers, self.readers):
            for b in book:
                book[b] = [x for x in book[b] if x != cp]
        self.ops.remove(cp)
        cp.erase()

    def _freed_between(self, b, first, last):
        """Does an op after `first` and up to `last` (both in the function's body) end buffer
        `b`'s life -- free it or a view of it?"""
        body = [o.operation for o in first.parent.regions[0].blocks[0].operations]
        span = body[body.index(first) + 1:body.index(last) + 1]
        return any(self.root(v) is not None and self.root(v)[0] == b and ends_lifetime(o, v)
                   for top in span for o in [top] + walk(top) for v in o.operands)



# ---- small facts ------------------------------------------------------------------------------

def _is_broadcast_copy(o):
    """A torchsim.copy that broadcasts (added_dims) into the whole of its destination."""
    return (o.name == "torchsim.copy" and "added_dims" in o.attributes
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
    """[source dims, destination dims] of a torchsim.copy that is no window, its dims the destination's
    logical axes: a broadcast's source axes land on the axes it does not add, a copy's on its equal logical
    axes (unit axes left out); None for a window or a reshape that splits an axis."""
    if o.name != "torchsim.copy" or "into_window" in o.attributes or "static_offsets" in o.attributes:
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
    if o.name == "torchsim.copy":
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
    if o.name == "torchsim.copy":
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
        if not is_memref(v):
            continue
        for ax, d in enumerate(m):
            if d is not None and d >= 0 and _shape(v)[ax] != 1:
                ext[d] = _shape(v)[ax]
    return ext


def _is_load(tx):
    from ...dialects.torchsim import is_load
    return is_load(tx)


def _sram_slot(tx):
    from ...dialects.torchsim import operand_number
    return operand_number(tx, "sram")


def _whole_tile(tx, known):
    """A transfer of its whole scratchpad tile, at its start: no gather, subtile or piece. A cut may stay:
    it says only where in DRAM each element is, and a fit keeps dram_idx and every tile coordinate."""
    from ...dialects.torchsim import operand_number
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
    from mlir.ir import ArrayAttr, MemRefType
    body = o.regions[0].blocks[0]
    ops_ = [x.operation for x in body.operations]
    maps = _maps(o)
    if not (_n_ins(o) == 1 and len(o.operands) == 2 and ops_[-1].name == "linalg.yield"
            and MemRefType(o.operands[0].type).shape == MemRefType(o.operands[1].type).shape
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


def _view_again(v, dst, src, at):
    """`v`, a view of `dst` (or `dst` itself), made again on `src` before `at`: the same reshapes,
    of `src`'s element type."""
    from mlir.ir import InsertionPoint, MemRefType, Operation
    chain = []
    while v != dst:
        chain.append(producer(v))
        v = chain[-1].operands[0]
    y = src
    elt = MemRefType(src.type).element_type
    with InsertionPoint(at):
        for p in reversed(chain):
            old = MemRefType(p.results[0].type)
            ty = MemRefType.get(list(old.shape), elt, memory_space=old.memory_space)
            y = Operation.create(p.name, results=[ty], operands=[y],
                                 attributes=attrs_of(p)).results[0]
    return y


def _cast_in_body(r, k, casts):
    """Reader `r`'s body takes its operand `k` as the casts' input and makes the casts first."""
    from mlir.ir import InsertionPoint, MemRefType, Operation
    body = r.regions[0].blocks[0]
    arg = body.arguments[k]
    arg.set_type(MemRefType(r.operands[k].type).element_type)
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
                z = producer(x.operands[i])
                if z is not None and z.name == "arith.constant" and _float_const(x.operands[i]) == 0.0 \
                        and not str(z.attributes["value"]).lstrip().startswith("-"):
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
    t = str(elem)
    if t == "f32":
        return struct.unpack("<f", (bits & 0xFFFFFFFF).to_bytes(4, "little"))[0]
    if t == "bf16":
        return struct.unpack("<f", ((bits & 0xFFFF) << 16).to_bytes(4, "little"))[0]
    if t == "f16":
        return struct.unpack("<e", (bits & 0xFFFF).to_bytes(2, "little"))[0]
    if t.startswith("i") and t[1:].isdigit():
        w = int(t[1:])
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
    o = producer(cond)
    if o is None:
        return None
    if o.name == "arith.andi":
        cuts = [c for c in (_mask_cut(o.operands[0], d), _mask_cut(o.operands[1], d))
                if c is not None]
        return min(cuts) if cuts else None
    if o.name != "arith.cmpi":
        return None
    pred = int(str(o.attributes["predicate"]).split(":")[0])
    if pred not in _FOLD_LT or _index_dim(o.operands[0]) != d:
        return None
    return _int_const(o.operands[1])
