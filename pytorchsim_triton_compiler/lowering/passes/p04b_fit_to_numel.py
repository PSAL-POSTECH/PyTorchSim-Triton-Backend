"""torchsim-fit-to-numel: every scratchpad buffer is cut to the part of it the kernel's data
needs, by changing its type -- once, here, so nothing after this works masks out again.

A Triton block is a power of two, so a numel of 768 rides a block of 1024. The layouts need that
(p02 cuts a dim by the bits of its linear layout); once bufferize has made them memref shapes,
nothing does: legalize widens any width, a DMA moves any tile, an array takes any K. So a buffer
axis padded past its numel is given the extent its readers need, worked backwards from the stores:
  - a masked store needs the scratchpad up to its clamp's high end, a plain one all of it;
  - a reduce needs its reduced dim up to where its input's padding starts, when that padding
    holds the combiner's identity (a masked load's fill, or a `select(index < C, x, identity)`);
  - an elementwise op needs what its outputs' readers need, and asks it of its inputs.
Anything else that touches a buffer (an array op, a view this does not follow) needs all of it.

The axes one linalg op walks with one dim keep one extent (they are cut together, to the most any
of them needs); a lane axis deeper than the lanes is cut to a whole number of rows a lane (every
consumer reads the lane split off the type, so the rows stay where each op looks for them), and a
cut of the last axis keeps a multiple of _VECTOR_GRAIN. Every op then walks the smaller buffers as they
are; a mask in a body that only cuts past the new extent is folded to true, and a generic that
then only copies its input is gone, its readers reading the input.
"""

import math

from .lib_bank_layout import is_one_bank
from .lib_ir import (const_int, funcs, indexing_maps, int_array, is_memref, op_of,
                     parallel_iterator, producer, walk)
from .lib_live import interval, program_ids

NAME = "torchsim-fit-to-numel"

_LINALG = ("linalg.generic", "linalg.reduce", "linalg.fill")

# a view axis that is part of a buffer's lane axis split into [lanes, laps]: (LANE, axis) or (LAP, axis)
LANE, LAP = "lane", "lap"
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


def run(module, grid=(), lanes=None, **_opts):
    for func in funcs(module):
        _Live(func, grid, lanes).fit()
    return module


class _Live:

    def __init__(self, func, grid, lanes=None):
        self.func = func
        self.lanes = lanes
        self.known = program_ids(func, grid)
        self.ops = [o for o in walk(func.operation)
                    if o.name in _LINALG or o.name == _TRANSFER or any(
                        is_memref(v) for v in o.operands)]
        self.ops = [o for o in self.ops if o.name not in _VIEWS and o.name != "memref.get_global"]
        self.shape = {}          # root buffer -> shape
        self.need = {}           # root buffer -> per-axis extent its readers need
        self.writers = {}        # root buffer -> ops that write it
        self.readers = {}        # root buffer -> ops that read it
        self.iters = {}          # linalg op -> per-iteration-dim extent
        for o in walk(func.operation):
            if o.name == "memref.get_global":
                b = o.results[0]
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
        is, or None for a unit axis a reshape added. None when `v` is not a buffer this follows."""
        if not is_memref(v):
            return None
        o = producer(v)
        if o is None:
            return None
        if o.name == "memref.get_global":
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
                        and up[1][s] == 0 and self.lanes and out[big[0]] == self.lanes \
                        and not is_one_bank(self.shape[up[0]]):
                    # a lane axis deeper than the lanes viewed as [lanes, laps]: the lanes are
                    # all of the machine's, the laps what of the axis a lane holds
                    for i in g:
                        axes[i] = None
                    axes[big[0]], axes[big[1]] = (LANE, 0), (LAP, 0)
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
        if _is_broadcast_copy(o):
            return k == 1
        if o.name in _LINALG:
            return k >= _n_ins(o)
        if o.name == _TRANSFER:
            return _is_load(o) and k == _sram_slot(o)
        return True

    def _reads(self, o, k):
        if _is_broadcast_copy(o):
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
            if o.name in _LINALG:
                it = self._iteration(o)
                self.iters[o] = it
                for k, (v, m) in enumerate(zip(o.operands, _maps(o))):
                    if k < _n_ins(o) or (o.name != "linalg.fill" and _reads_init(o, k)):
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

    def _roots_under(self, v):
        """The buffers a value this does not follow may alias: all of them that reach it."""
        o = producer(v)
        seen = []
        while o is not None and is_memref(o.operands[0] if o.operands else None):
            src = o.operands[0]
            if producer(src) is not None and producer(src).name == "memref.get_global":
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
                    it[d] = max(it[d], full[d] if a is None or (isinstance(a, tuple) and a[0] == LANE)
                                else -(-self.need[b][a[1]] // self.lanes) if isinstance(a, tuple)
                                else self.need[b][a])
        for d in self._reduced(o):
            it[d] = self._reduce_reach(o, d, maps, full[d])
        lane = maps[-1][0] if maps[-1] and not is_one_bank(list(_shape(o.operands[-1]))) else None
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
        if isinstance(a, tuple) and a[0] == LANE:
            return full
        pad = self._padding(b, a[1] if isinstance(a, tuple) else a, o)
        if pad is None or not _IDENTITY[body[0].name](pad[1]):
            return full
        if isinstance(a, tuple):
            return pad[0] // self.lanes if pad[0] % self.lanes == 0 else full
        return pad[0]

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
        """{buffer: new shape}: the axes one op walks with one dim form a class, cut to the most
        any of them needs; a class with a lane axis, or with axes of different extents, is kept."""
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for b, shape in self.shape.items():
            for a in range(len(shape)):
                find((b, a))
        for o in self.ops:
            if o.name not in _LINALG:
                continue
            by_dim = {}
            for v, m in zip(o.operands, _maps(o)):
                ref = self.root(v)
                if ref is None:
                    continue
                for i, d in enumerate(m):
                    if d is not None and ref[1][i] is not None and not isinstance(ref[1][i], tuple):
                        by_dim.setdefault(d, []).append((ref[0], ref[1][i]))
            for keys in by_dim.values():
                for k in keys[1:]:
                    parent[find(k)] = find(keys[0])
        classes = {}
        for key in parent:
            classes.setdefault(find(key), []).append(key)
        out = {b: list(s) for b, s in self.shape.items()}
        for keys in classes.values():
            sizes = {self.shape[b][a] for b, a in keys}
            if len(sizes) != 1:
                continue
            full = sizes.pop()
            want = max(self.need[b][a] for b, a in keys)
            if any(a == 0 and not is_one_bank(self.shape[b]) for b, a in keys):
                want = self._lane_cut(want, full)
            if any(a == len(self.shape[b]) - 1 for b, a in keys):
                want = -(-want // _VECTOR_GRAIN) * _VECTOR_GRAIN
            if 0 < want < full:
                for b, a in keys:
                    out[b][a] = want
        return out

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
        import re
        from mlir.ir import StringAttr
        for b, s in cut.items():
            ty = MemRefType(b.type)
            new = MemRefType.get(s, ty.element_type, memory_space=ty.memory_space)
            name = FlatSymbolRefAttr(producer(b).attributes["name"]).value
            g = globals_[name]
            g.attributes["type"] = TypeAttr.get(new)
            b.set_type(new)
            if s[0] != ty.shape[0] and re.search(r"_\d+lane$", name):
                # a scratchpad global's name states its lane spread, read back by later passes
                renamed = re.sub(r"_\d+lane$", f"_{s[0]}lane", name)
                g.attributes["sym_name"] = StringAttr.get(renamed)
                for o in walk(module):
                    if o.name == "memref.get_global" and \
                            FlatSymbolRefAttr(o.attributes["name"]).value == name:
                        o.attributes["name"] = FlatSymbolRefAttr.get(renamed)
        for o in walk(self.func.operation):
            if o.name not in _VIEWS:
                continue
            ref = self.root(o.results[0])
            if ref is None or ref[0] not in cut:
                continue
            b, axes = ref
            old = MemRefType(o.results[0].type)
            s = [old.shape[i] if a is None else self.lanes if isinstance(a, tuple) and a[0] == LANE
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
            if _is_identity_copy(o):
                self._forward(o)

    def _forward(self, cp):
        """An elementwise copy whose mask folded copies its input as it is: the ops that read what
        it wrote (after it in its block, before the buffer is written again) read its input through
        the same views instead, and the copy goes. A copy kept carries COPY_KEPT, why."""
        from mlir.ir import InsertionPoint, StringAttr

        def keep(why):
            cp.attributes[COPY_KEPT] = StringAttr.get(why)
        src, dst = cp.operands[0], cp.operands[1]
        if producer(dst) is None or producer(dst).name != "memref.get_global" \
                or dst not in self.writers or self.root(src) is None:
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
        for r in readers:
            for k, v in enumerate(r.operands):
                ref = self.root(v)
                if ref is None or ref[0] != dst:
                    continue
                chain, x = [], v
                while x != dst:
                    p = producer(x)
                    chain.append(p)
                    x = p.operands[0]
                y = src
                with InsertionPoint(r):
                    for p in reversed(chain):
                        c = p.clone()
                        c.operands[0] = y
                        y = c.results[0]
                r.operands[k] = y
            self.readers[dst] = [x for x in self.readers[dst] if x != r]
            self.readers[src_root].append(r)
        for book in (self.writers, self.readers):
            for b in book:
                book[b] = [x for x in book[b] if x != cp]
        self.ops.remove(cp)
        cp.erase()



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


def _n_ins(o):
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
    if o.name == "linalg.generic":
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
    """A transfer of its whole scratchpad tile, at its start: no gather, cut or subtile."""
    from ...dialects.torchsim import operand_number
    for name in ("subtile_size", "cut", "indirect_dim"):
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


def _is_identity_copy(o):
    """A generic that writes its one input, unchanged, into its one output of the same type."""
    from mlir.ir import ArrayAttr
    body = o.regions[0].blocks[0]
    ops_ = [x.operation for x in body.operations]
    maps = _maps(o)
    return (_n_ins(o) == 1 and len(o.operands) == 2 and len(ops_) == 1
            and ops_[0].name == "linalg.yield" and ops_[0].operands[0] == body.arguments[0]
            and o.operands[0].type == o.operands[1].type
            and maps[0] == maps[1] and sorted(maps[1]) == list(range(len(maps[1])))
            and all(t == parallel_iterator() for t in ArrayAttr(o.attributes["iterator_types"])))


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
