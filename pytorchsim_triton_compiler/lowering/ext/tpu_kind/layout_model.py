"""The TPU's answers on the layout port (slots.LayoutModel): what a conversion costs is the move
lower-convert makes of it -- the XLU op lib_xlu_plan chooses for the element pairs the two layouts imply, in
the cycles the machine file's keys give that op; a conversion that keeps every element in its lane is a
copy in each lane's bank. A join, split or cat lowers in any layout; an op made again costs its DMA
hold or its vector registers."""
import math

from .... import machine
from .lib_xlu_plan import (XLU_CROSSING, XLU_FOLDS, XLU_PERMUTE, choose_xlu_op, crossings_of,
                           emitter_of, permute_runs)

lanes = machine.VECTORLANE_SIZE


def layout_free(op):
    """A join, split or cat lowers in any layout: TO_LINALG makes it slices of tensors, whose layouts a03
    lays out as it lays out any; for every other op upstream's rules stand."""
    return op in ("join", "split", "cat")


def layout_legal(op, operand, result):
    """Whether `op` takes `operand` into `result` in each lane: the axis a join adds, a split takes away or
    a cat doubles must sit in each lane's registers, since a03 holds a tile replicated across lanes in its
    first lanes only -- on a lane bit, a part would have to cross the lanes to its place."""
    if op == "join":
        return not any(b[-1] for b in result.lane)
    if op == "split":
        return not any(b[-1] for b in operand.lane)
    if op == "cat":
        half = operand.shape[0]
        return not any(b[0] & half for b in result.lane)
    return True


def lane_folds(combiner, elem_bits):
    """Whether the cross-lane unit folds a reduce combining by `combiner`: the f32 folds of
    xlu_push_reduce (lib_xlu_plan.XLU_FOLDS)."""
    return combiner in XLU_FOLDS and elem_bits == 32


def remat_cost(op, at):
    """The cycles of making an op of `op`'s kind in layout `at`: an address or a shape op nothing (an address
    becomes the DMA descriptor's strides, not per-element work), a load one DMA piece's hold
    (dma_load_latency_cycles), an elementwise op or a reduce the vector registers each lane holds of `at`
    (one a cycle; a reduce's `at` is its input; a layout every lane holds whole is one lane's work), a math
    function those plus tpu_sfu_latency_cycles; -1 where the machine file says no cost."""
    t = machine.target()
    if op in ("reshape", "trans", "address"):
        return 0.0
    if op == "load":
        return -1.0 if t.dma_load_latency_cycles is None else float(t.dma_load_latency_cycles)
    regs = -(-(1 << sum(1 for b in at.reg if any(b))) * at.elem_bits // t.vlen_bits)
    if op == "math":
        return float(regs + int(t.vcix.get("tpu_sfu_latency_cycles", 0)))
    return float(regs)


def convert_cost(src, dst):
    """The cycles of the move from `src` into `dst`: a copy in the bank (one vector register a cycle
    over each lane's rows) when no element leaves its lane, else the XLU op lib_xlu_plan chooses, through
    _move_cycles; inf when no XLU op carries it."""
    pairs = pairs_of_layouts(src, dst)
    if all(s[0] == d[0] for s, d in pairs):
        rows = 1 << len(dst.reg)
        return float(-(-rows * dst.elem_bits // machine.target().vlen_bits))
    s_shape, d_shape = [1 << len(src.lane), 1 << len(src.reg)], [1 << len(dst.lane), 1 << len(dst.reg)]
    src_lanes = 1 << sum(1 for b in src.lane if any(b))
    chosen = choose_xlu_op(pairs, s_shape, d_shape, lanes, src_lanes == 1, src_lanes)
    if chosen is None:
        return math.inf
    cycles = _move_cycles(chosen, src.elem_bits, 1 << sum(1 for b in dst.lane if any(b)), len(pairs))
    return -1.0 if cycles is None else float(cycles)


def pairs_of_layouts(src, dst):
    """Each destination slot's `(source (lane, row), destination (lane, row))`, an element its row-major
    place in its tile (so a reshape pairs as a conversion does): the row is the register index, the source
    slot one in the destination's own lane where that lane holds the element, else the first that does."""
    where = {}
    for lane, row, at in _slots(src):
        where.setdefault(_flat(at, src.shape), {}).setdefault(lane, row)
    return [((lane, held[lane]) if lane in held else next(iter(held.items())), (lane, row))
            for lane, row, at in _slots(dst) for held in [where[_flat(at, dst.shape)]]]


def _flat(at, shape):
    """The row-major place of coordinate `at` in a tile of `shape`."""
    place = 0
    for x, n in zip(at, shape):
        place = place * n + x
    return place


def _slots(layout):
    """(lane, row, coordinate) of every register of every lane a tile of `layout` is held in: a lane bit
    that moves no element (a replica across lanes) is no lane a03 holds the tile in, so lanes count over
    the bits that move one, as a03 lays the tile on its first lanes."""
    lanes = [b for b in layout.lane if any(b)]
    for lane in range(1 << len(lanes)):
        for row in range(1 << len(layout.reg)):
            at = [0] * len(layout.shape)
            for bits, bases in ((lane, lanes), (row, layout.reg)):
                for i, b in enumerate(bases):
                    if bits >> i & 1:
                        at = [x ^ y for x, y in zip(at, b)]
            yield lane, row, tuple(at)


def _move_cycles(chosen, elem_bits, dst_lanes, moved):
    """The unit's cycles for one move, from the target's vcix keys: a permute's runs through the move
    stages, a crossing's passes one after another, a gather through the stages then its elements over
    the lanes a cycle; None when a key is unsaid."""
    xlu_op, plan = chosen
    t = machine.target()
    if "tpu_xlu_latency_cycles" not in t.vcix:
        return None
    crossings, _ = crossings_of(elem_bits)
    wide = max(elem_bits, 32)
    if emitter_of(xlu_op) != XLU_PERMUTE:
        stages = int(t.vcix["tpu_xlu_latency_cycles"]) + 1
        if emitter_of(xlu_op) == XLU_CROSSING:
            per_reg = t.vlen_bits // wide
            return None if not crossings else crossings * sum(
                stages + rows // (math.gcd(rows, per_reg) if s_step == 1 else 1)
                + width // (math.gcd(width, per_reg) if d_step == 1 else 1)
                for _s, s_step, _d, d_step, rows, width in plan)
        return stages + -(-moved // lanes)
    if "tpu_xlu_elements_per_cycle" not in t.vcix or not crossings:
        return None
    unit = int(t.vcix["tpu_xlu_elements_per_cycle"])
    latency = int(t.vcix.get("tpu_xlu_move_latency_cycles", t.vcix["tpu_xlu_latency_cycles"]))
    runs = permute_runs(plan, min(dst_lanes, lanes), t.vlen_bits // wide)
    return latency + 1 + crossings * sum(-(-n // math.gcd(w, unit)) + -(-n // w) for _, (*_, n), w, _ in runs)
