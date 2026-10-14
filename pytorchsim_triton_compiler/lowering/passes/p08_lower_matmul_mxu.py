"""torchsim-lower-matmul-mxu: every torchsim.dot onto the matrix unit of a target whose
`matmul_unit` is "mxu" (vcix-accelerator's libtpu_mxu.so): an array twice as wide and as deep as
the machine has lanes, so an operand is two registers -- a vmatprep with the upper lane tile, then
the vmatpush (weights) or the vmatmul (inputs) with the lower. A word of 32 bits is one f32 or two
rows of 16 bits; a result leaves as f32, the lower lane tile's columns and the upper's in turn,
and each pop is added into C -- written there as it is at the first K pair when C is proven zero,
and run through the dot's epilogue at the last, by torchsim-lower-matmul's own rules. The unit
has `matmul_units` arrays (the v6e's mxu0 and mxu1), named in the field its encodings leave free
(sf.vc.iv's rd, sf.vc.v.i's rs2): the K pairs go to them by turns, a group of up to that many
pairs holding a panel on each array, and an input goes to every array of the group before the
group's rows are popped from each and added in registers, so C is read and written once a group.
The dot's inputs are one stream, across its K groups and its N pairs alike: as many are pushed
ahead of their pops as the unit's delay line holds at the rate it takes them, and from then on
one is popped for each one pushed. Runs in place of torchsim-lower-matmul, which it leaves as it
is."""
import sys
from collections import namedtuple

from ... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from . import p08_lower_matmul as p08  # noqa: E402
from . import lib_pack  # noqa: E402
from .lib_bank_layout import bank_view, index_bank_casts  # noqa: E402
from .lib_ir import (build_trip_loop, const_index, elem_bits, transfer_read,  # noqa: E402
                     transfer_write, zero_of)
from .lib_vcix import bitcast, i64 as a64, vcix  # noqa: E402

MATMUL_OPS = p08.MATMUL_OPS

FORBIDS = MATMUL_OPS

MARKERS = MATMUL_OPS

# The vcix.iv / vcix.v.i opcodes of the unit's four instructions, and the bits of their imm.
VMATMUL, VMATPUSH, VMATPREP, VPOP = 0, 1, 2, 2
FORMAT_F32, FORMAT_F16, FORMAT_BF16 = 0, 1, 3
SECOND = 4      # a vmatprep of an input, not of weights; a vpop of the upper columns
LATCH = 4       # a vmatmul that first makes the staged weights the active
FIRST_HALF, SECOND_HALF = 0x08, 0x10    # which half of the depth a vmatpush stages, at [4:3]


def _simm5(field):
    """The field at 19:15 as the signed 5-bit immediate the op carries: bit 4 set reads negative."""
    return field - 32 if field & 0x10 else field


def run(module, *, vectorlane, vlen, queue_entries, matmul_units=1, **_):
    """Every matmul in `module` lowered in place. A C proven zero takes the first K group's pops as
    they are, and its zero fill goes when nothing else reads it."""
    from .lib_ir import walk
    index_bank_casts(module)
    dots = [o for o in walk(module.operation) if o.name in MATMUL_OPS]
    inits = [_zero_init(o, vectorlane, vlen) for o in dots]
    dead = p08._dead_initializers([i for i in inits if i is not None])
    follows = [_follows(dots[i - 1], o) if i else False for i, o in enumerate(dots)]
    held = None
    for o, init, after in zip(dots, inits, follows):
        held = _rewrite_matmul(o, vectorlane, vlen, queue_entries, matmul_units, init is not None,
                               held if after else None)
    for nest in dead:
        nest.erase()
    return module


def _follows(before, op):
    """Whether `op` runs straight after the dot `before` with the same weights: the next matmul of
    its block, B the same buffer, and nothing between that writes it. The arrays then still hold
    what `before` staged last, and a first panel of `op` that is that one is not staged again."""
    if before.operation.block != op.operation.block:
        return False
    b = op.operands[1]
    if p08._root(before.operands[1]) != p08._root(b) or before.operands[1].type != b.type:
        return False
    return p08._used_between(b, before.operation, op.operation, writes=True) is None


def _zero_init(op, lanes, vlen):
    """torchsim-lower-matmul's proof that C holds zero over every element the pops touch -- here
    all of C: its last writer in the block is a zero-splat store nest over them, or a nest copying
    them from a buffer whose own last writer is one. None otherwise."""
    C = op.operands[2]
    if any(p08._root(v) == p08._root(C) for v in op.operands[:2]) or not p08._plain(C):
        return None
    elt = ir.MemRefType(op.operands[0].type).element_type
    if elem_bits(elt) not in (16, 32):
        return None
    words = vlen // 32
    rows = words * (32 // elem_bits(elt))
    M, N, K = p08._get_mnk(op)
    if M % rows:
        return None
    runs = p08._runs([n * M + m * rows + i for n in range(max(1, N // lanes))
                      for m in range(M // rows) for i in range(0, rows, words)], words)
    if runs is None:
        return None
    writer = p08._last_writer(op.operation, C)
    if writer is None:
        return None
    filled = p08._store_nest(writer, C, None)
    if filled is not None and p08._covers(filled, runs):
        return p08._ZeroInit(op.operation, C, runs, True, None, (writer, filled))
    copied = p08._store_nest(writer, C, True)
    if copied is None or not p08._covers(copied[1], runs):
        return None
    Z = copied[0]
    source = p08._last_writer(writer, Z)
    filled = None if source is None else p08._store_nest(source, Z, None)
    if filled is None or not p08._covers(filled, copied[1]):
        return None
    return p08._ZeroInit(op.operation, C, runs, True, (writer, copied[1]), (source, filled))


def _format(elt):
    if isinstance(elt, ir.BF16Type):
        return FORMAT_BF16
    return {16: FORMAT_F16, 32: FORMAT_F32}[elem_bits(elt)]


def _refuse(op, M, N, K, lanes, rows, queue, elt, acc_elt):
    if elem_bits(elt) not in (16, 32):
        raise p08.ArrayRefusal(
            f"the matrix unit takes words of one f32 or two rows of 16 bits, not {elt}: {op}")
    if elem_bits(acc_elt) != 32:
        raise p08.ArrayRefusal(f"the matrix unit pops f32, and C is {acc_elt}: {op}")
    for d, name in ((N, "N"), (K, "K")):
        if d % lanes and (name == "K" or d > lanes):
            raise p08.ArrayRefusal(
                f"matmul {name}={d} is not whole lane tiles of {lanes}; the matrix unit's "
                f"lowering masks no lanes yet: {op}")
    if M % rows:
        raise p08.ArrayRefusal(
            f"matmul M={M} is not whole registers of {rows} rows; the matrix unit's lowering "
            f"has no shorter last push yet: {op}")
    if queue < rows:
        raise p08.ArrayRefusal(
            f"the matrix unit's queues hold {queue} rows, less than one input of {rows}: {op}")


def _rewrite_matmul(op, lanes, vlen, queue, arrays, zero, held=None):
    from mlir.dialects import arith

    A, B, C = op.operands[0], op.operands[1], op.operands[2]
    mtA, mtB, mtC = (ir.MemRefType(x.type) for x in (A, B, C))
    elt, acc_elt = mtA.element_type, mtC.element_type
    M, N, K = p08._get_mnk(op)
    p08._refuse_unlowerable(op, elt, acc_elt, mtB.element_type)
    words = vlen // 32
    pack = 32 // elem_bits(elt) if elem_bits(elt) in (16, 32) else 0
    rows = words * pack
    _refuse(op, M, N, K, lanes, rows, queue, elt, acc_elt)

    i32 = ir.IntegerType.get_signless(32)
    word_ty = ir.VectorType.get([words], i32)
    u = _Unit(lanes=lanes, words=words, rows=rows, pack=pack, fmt=_format(elt), M=M, K=K,
              arrays=arrays, vec=ir.VectorType.get([rows], elt), word=word_ty,
              acc=ir.VectorType.get([words], acc_elt))
    ip = ir.InsertionPoint(op)
    with ip:
        u.rvl = arith.ConstantOp(ir.IntegerType.get_signless(64), a64(words)).result
        u.zero = arith.ConstantOp(word_ty, ir.DenseElementsAttr.get_splat(
            word_ty, ir.IntegerAttr.get(i32, 0))).result
        u.a, u.b, u.c = (bank_view(x, lanes) for x in (A, B, C))
        # A and B of 16 bits are read as the words the unit takes, through torchsim.pack: read
        # as 16 bits and cast, a load would change the element width and back.
        if pack > 1:
            u.a, u.b = (lib_pack.pack(x, op) for x in (u.a, u.b))
        u.word_pad = zero_of(i32)
        u.pad, u.acc_pad = zero_of(elt), zero_of(acc_elt)

    # Lane tiles go to the arrays in pairs; an odd last N tile goes with zeros for its upper columns,
    # an odd last K tile stages the first half of the depth alone. The K pairs go to the arrays by
    # turns, a group of up to `arrays` pairs at a time. The dot's inputs are one stream over every
    # (N pair, K group) in turn: the input `lag` behind the one pushed is popped before the push,
    # from every array of its group, weights change while results are on their way, and what is
    # left is popped at the end -- the unit's delay is waited out once a dot. Which group a pop's
    # rows are of, so which columns of C they go to, whether C is read under them and whether the
    # epilogue runs on them, is settled here, group by group, and the loops are cut where it
    # changes: no input tests it at run time.
    n_tiles, k_tiles = max(1, N // lanes), K // lanes
    inputs = M // rows
    n_parts, k_parts = _pairs(n_tiles), _pairs(k_tiles)
    pairs_total = sum(pairs for _, pairs, _ in k_parts)
    u.zero_c = zero
    g = p08._Gemm(None, a64, None, None, elem_bits(elt), rows, words, None, None, rows, False,
                  ((n_tiles, pairs_total, inputs), K, M, rows, lanes), None, zero)
    u.epi = p08._find_epilogue(op, g, p08._GemmOperand(u.c, u.acc, u.word, u.acc_pad), ip)
    if u.epi is not None:
        p08._build_waits(ip, u.epi)
    held = _build_dot(u, ip, n_parts, k_parts, inputs, _lag(lanes, words, rows, queue), held)
    if u.epi is not None:
        u.epi.nest.erase()
    op.erase()
    return held


def _lag(lanes, words, rows, queue):
    """How many inputs are pushed before the first is popped: what the unit's delay line holds,
    one word a cycle, plus the input entering and one more -- but no more than the unit's queues
    hold, `queue` rows in and `queue` rows out of each half of the columns. The line is the
    array's width + depth - 1 slots, 4 x lanes - 1, the model's default for `tpu_mxu_delay_cycles`
    (vcix-accelerator, examples/tpu/mxu.hpp); a pop that comes sooner waits for its rows. The
    inputs between a pop and the push after it, `lag` of them, are all in the unit: when the
    line has stopped because the output queues are full, they are in the queues alone, and a
    push with no room in the input queue waits for a pop that is behind it in program order --
    the core goes no further (seen at a lag of 66 with queues of 256 rows: 32 inputs).
    Measured on 1024^3 bf16: a lag of 32 (the systolic lowering's 2 x queue / rows) left the pops
    waiting 26% of the tile; 64 took the matmul tile from 18711 to 14297 cycles."""
    delay = 4 * lanes - 1
    by_delay = -(-(delay + words) // words) + 1
    return max(1, min(by_delay, 2 * queue // rows))


# A pop before an input: the rows of input `row` of the N pair `back` before this one, from each of
# the `arrays` arrays of its K group, `first` when that is the first K group of its N pair (C holds
# nothing yet), `last` when the last (the sum is done).
_Pop = namedtuple("_Pop", "row back first last arrays")


def _build_dot(u, ip, n_parts, k_parts, inputs, lag, held=None):
    """The stream: every (N pair, K group) in turn -- a K group is up to `u.arrays` K pairs, a
    panel on each array -- its panels and its inputs, then the drain. Position p of the stream is
    input p % inputs of group p // inputs; what is popped before it is position p - lag, from
    every array of that group. Each N pair is emitted on its own (see n0); within one, K groups
    with the same plan are one loop. `held` is what the dot before left staged on each array, when
    this one follows it with the same weights (_follows): a first group whose panels are those is
    not staged again, its first input latching what is there. What this dot leaves is returned."""
    # K pair q: its first K tile, and whether it has an upper tile; the groups of them.
    k_pairs = [(first + 2 * t, upper) for first, pairs, upper in k_parts for t in range(pairs)]
    groups = [k_pairs[g:g + u.arrays] for g in range(0, len(k_pairs), u.arrays)]
    n_groups = len(groups)
    last_g = n_groups - 1
    n_pairs = n_parts[0][1] if n_parts[0][2] else 0
    odd_n = None if n_parts[-1][2] else n_parts[-1]
    n_count = n_pairs + (1 if odd_n else 0)
    total = n_count * n_groups * inputs
    lag = min(total, lag)
    # Every N pair is emitted on its own. From the N pair where every position is past `lag` on,
    # the pairs could be one loop (which N pair a pop is of is then a distance, which K group a
    # remainder), but with that loop's body LLVM runs short of registers and rebuilds each access's
    # address from a base beside the load, which the in-order core waits out: the K pairs that read
    # C went from 14 to 17 cycles an input, and the dot gained nothing from its one stream. Emitted
    # apart, every such loop keeps its incremented addresses (1024^3 bf16: 14763 -> 14081 cycles).
    n0 = n_pairs
    # The groups that are `u.arrays` pairs with an upper K tile each: a loop over them puts array
    # j's panel at 2 * j tiles from the loop's. They come first; the rest are emitted on their own.
    regular = sum(1 for g in groups if len(g) == u.arrays and all(upper for _, upper in g))
    assert all(len(g) == u.arrays and all(upper for _, upper in g) for g in groups[:regular])

    def n_tile_of(i):
        return 2 * i if i < n_pairs else odd_n[0]

    def staged(i, g):
        """What N pair `i`'s K group `g` stages, per array: the panel's tiles and its format."""
        return {array: (n_tile_of(i), i < n_pairs, k_tile, k_upper, u.fmt, u.K)
                for array, (k_tile, k_upper) in enumerate(groups[g])}
    reuse = held is not None and all(held.get(a) == p for a, p in staged(0, 0).items())

    def plan(i, g):
        """The inputs of K group `g` of N pair `i` as segments (m_lo, m_hi, pop): before input m,
        nothing, or the rows of input `pop.row + m - m_lo` of the group the pop is of."""
        segments = []
        for m in range(inputs):
            p = (i * n_groups + g) * inputs + m
            if p < lag:
                key = pop = None
            else:
                b = p - lag
                n_b, g_b = divmod(b // inputs, n_groups)
                key = (i - n_b, g_b)
                pop = _Pop(b % inputs, i - n_b, g_b == 0, g_b == last_g, len(groups[g_b]))
            if segments and segments[-1][2] == key:
                segments[-1][1] = m + 1
            else:
                segments.append([m, m + 1, key, pop])
        return tuple((lo, hi, pop) for lo, hi, _, pop in segments)

    def emit_n(n_ip, i, n_tile, n_upper, n_index):
        """The K groups of one N pair. `i` stands for the pair in the plans: the pair itself, or
        the first of the looped ones. `n_index` is the pair's number when the code is its own,
        None in the loop, where the pair a pop is of is `back` pairs below the loop's."""
        from mlir.dialects import arith

        def target(dn):
            if n_index is not None:
                return const_index(n_tile_of(n_index - dn)), n_index - dn < n_pairs
            return (n_tile if dn == 0 else arith.SubIOp(n_tile, const_index(2 * dn)).result), True
        g = 0
        while g < regular:
            pl, hi = plan(i, g), g + 1
            kept = reuse and n_index == 0 and g == 0    # its panels are in the arrays already
            while not kept and hi < regular and plan(i, hi) == pl:
                hi += 1
            g_ip, base = _trip_loop(n_ip, 2 * u.arrays * g, hi - g, 2 * u.arrays)
            tiles = [(_tiles_on(g_ip, base, 2 * j), True) for j in range(u.arrays)]
            _build_group(u, g_ip, n_tile, n_upper, tiles, pl, target, not kept)
            g = hi
        for g in range(regular, n_groups):
            tiles = [(const_index_at(n_ip, tile), upper) for tile, upper in groups[g]]
            _build_group(u, n_ip, n_tile, n_upper, tiles, plan(i, g), target,
                         not (reuse and n_index == 0 and g == 0))
    for i in range(n0):
        emit_n(ip, i, const_index_at(ip, n_tile_of(i)), True, i)
    if n0 < n_pairs:
        l_ip, n_tile = _trip_loop(ip, 2 * n0, n_pairs - n0, 2)
        emit_n(l_ip, n0, n_tile, True, None)
    if odd_n:
        emit_n(ip, n_pairs, const_index_at(ip, odd_n[0]), False, n_pairs)
    for b_lo, b_hi, g_b in _plan_drain(total, lag, inputs):
        n_b, g_b = divmod(g_b, n_groups)
        d_ip, row = _trip_loop(ip, b_lo % inputs, b_hi - b_lo, 1)
        with d_ip:
            _build_pops(u, const_index(n_tile_of(n_b)), row, n_b < n_pairs, g_b == 0, g_b == last_g,
                        len(groups[g_b]))
    # An array the last group does not use keeps an older panel; it is not counted on.
    return staged(n_count - 1, last_g)


def _plan_drain(total, lag, inputs):
    """What the stream leaves to pop, as runs (b_lo, b_hi, group) over the stream's positions."""
    out = []
    for b in range(total - lag, total):
        if out and out[-1][2] == b // inputs:
            out[-1][1] = b + 1
        else:
            out.append([b, b + 1, b // inputs])
    return [tuple(x) for x in out]


def _tiles_on(ip, base, offset):
    """The K tile `offset` tiles from `base`: `base` itself at 0."""
    from mlir.dialects import arith
    if offset == 0:
        return base
    with ip:
        return arith.AddIOp(base, const_index(offset)).result


def _build_group(u, ip, n_tile, n_upper, tiles, segments, target, weights=True):
    """A panel of weights on each array of the group -- `tiles` is (K tile, whether it has an
    upper tile) per array; none when `weights` is False, the arrays holding them from the dot
    before -- then the group's inputs segment by segment, each to every array,
    behind the pops its segment says; `target(back)` is the N tile and whether it has an upper
    tile of the pair a pop is of. The first input is emitted on its own, as it latches the panels."""
    from mlir.dialects import arith

    def pops(at, row, pop):
        with at:
            tile, upper = target(pop.back)
            _build_pops(u, tile, row, upper, pop.first, pop.last, pop.arrays)

    def inputs_of(m_idx, latch):
        for array, (k_tile, k_upper) in enumerate(tiles):
            _build_input(u, k_tile, m_idx, k_upper, latch, array)
    if weights:
        with ip:
            for array, (k_tile, k_upper) in enumerate(tiles):
                _build_weights(u, n_tile, k_tile, n_upper, k_upper, array)
    for m_lo, m_hi, pop in segments:
        if m_lo == 0:
            if pop is not None:
                pops(ip, const_index_at(ip, pop.row), pop)
            with ip:
                inputs_of(const_index(0), True)
            m_lo = 1
            if pop is not None:
                pop = pop._replace(row=pop.row + 1)
        if m_lo >= m_hi:
            continue
        s_ip, m = _trip_loop(ip, m_lo, m_hi - m_lo, 1)
        if pop is not None:
            with s_ip:
                if m_hi - m_lo == 1:
                    row = const_index(pop.row)
                elif pop.row >= m_lo:
                    row = arith.AddIOp(m, const_index(pop.row - m_lo)).result
                else:
                    row = arith.SubIOp(m, const_index(m_lo - pop.row)).result
            pops(s_ip, row, pop)
        with s_ip:
            inputs_of(m, False)


class _Unit:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def const_index_at(ip, v):
    with ip:
        return const_index(v)


def _pairs(tiles):
    """(first tile, trips, whether each trip has an upper tile): the pairs, then an odd last tile."""
    out = []
    if tiles // 2:
        out.append((0, tiles // 2, True))
    if tiles % 2:
        out.append((tiles - 1, 1, False))
    return out


def _trip_loop(ip, first, trips, step):
    """A loop over the values first, first + step, ... `trips` times: (the point inside, the value
    as an index). One trip is no loop and the value a constant."""
    from mlir.dialects import arith
    if trips == 1:
        return ip, const_index_at(ip, first)
    iv, inner = build_trip_loop(ip, trips)
    with inner:
        v = iv if step == 1 else arith.MulIOp(iv, const_index(step)).result
        return inner, (v if first == 0 else arith.AddIOp(v, const_index(first)).result)


def _at(tile, upper, extent, inner, span, at):
    """[0, (tile + upper) * extent + inner * span + at], the in-bank offset of one register."""
    from mlir.dialects import arith
    if upper:
        tile = arith.AddIOp(tile, const_index(1)).result
    flat = arith.MulIOp(tile, const_index(extent)).result
    if inner is not None:
        flat = arith.AddIOp(flat, arith.MulIOp(inner, const_index(span)).result).result
    return [const_index(0), arith.AddIOp(flat, const_index(at)).result]


def _words(u, bank, at):
    """One register of the operand at `at` -- an offset in words where the bank is packed -- as
    the words the unit takes."""
    if u.pack == 1:
        return bitcast(transfer_read(u.vec, bank, at, u.pad, True), u.word)
    return transfer_read(u.word, bank, at, u.word_pad, True)


def _build_weights(u, n_tile, k_tile, n_upper, k_upper, array):
    """A panel of weights staged on `array` as the v6e stages it: the first half of the depth (K
    rows 0 to lanes - 1 of the pair), then the second, each a register of rows at a time with the
    deepest first, so that the last push is lowest. A pair with no upper K tile stages the first
    half alone, as XLA does for K = 128: the second half keeps what it held and meets the zeros
    under the inputs' vmatprep. The vmatprep carries the upper N tile's columns, zeros where the
    pair has none."""
    from mlir.dialects import arith
    # Offsets are in the bank's elements, words where a word is `pack` of B's.
    k_at = arith.MulIOp(k_tile, const_index(u.lanes // u.pack)).result
    extent = u.K // u.pack
    for half, base in ((FIRST_HALF, 0), (SECOND_HALF, u.lanes)):
        if base and not k_upper:
            continue
        for row in range(base + u.lanes - u.rows, base - 1, -u.rows):
            at = row // u.pack
            upper = _words(u, u.b, _at(n_tile, True, extent, k_at, 1, at)) if n_upper else u.zero
            vcix("vcix.iv", [upper, u.rvl], [],
                 {"opcode": a64(VMATPREP), "imm": a64(u.fmt), "rd": a64(array)})
            lower = _words(u, u.b, _at(n_tile, False, extent, k_at, 1, at))
            vcix("vcix.iv", [lower, u.rvl], [],
                 {"opcode": a64(VMATPUSH), "imm": a64(_simm5(u.fmt | half)), "rd": a64(array)})


def _build_input(u, k_tile, m_idx, k_upper, latch, array):
    """One register of rows of A to `array`: the vmatprep carries the upper K tile's lanes, zeros
    where the pair has none, and the vmatmul the lower; one that latches makes the staged weights
    active."""
    extent, span = u.M // u.pack, u.rows // u.pack
    upper = _words(u, u.a, _at(k_tile, True, extent, m_idx, span, 0)) if k_upper else u.zero
    vcix("vcix.iv", [upper, u.rvl], [],
         {"opcode": a64(VMATPREP), "imm": a64(u.fmt | SECOND), "rd": a64(array)})
    lower = _words(u, u.a, _at(k_tile, False, extent, m_idx, span, 0))
    vcix("vcix.iv", [lower, u.rvl], [],
         {"opcode": a64(VMATMUL), "imm": a64(u.fmt | (LATCH if latch else 0)), "rd": a64(array)})


def _build_pops(u, n_tile, m_idx, n_upper, first, last, arrays):
    """The rows one input's vmatmuls computed, a register of f32 at a time from each of the
    group's `arrays` arrays, added in registers: the lower N tile's columns, then the upper's.
    Each sum is added into C -- written there as it is, at the stream's first K group, when C is
    proven zero -- and at the last K group the epilogue runs on the sum. The upper half of an odd
    last N tile is popped and dropped."""
    from mlir.dialects import arith
    for row in range(0, u.rows, u.words):
        for upper in (False, True):
            got = [bitcast(vcix("vcix.v.i", [u.rvl], [u.word],
                                {"opcode": a64(VPOP), "imm": a64(SECOND if upper else 0),
                                 "rs2": a64(array)}).results[0], u.acc)
                   for array in range(arrays)]
            if upper and not n_upper:
                continue
            summed = got[0]
            for more in got[1:]:
                summed = arith.AddFOp(summed, more).result
            at = _at(n_tile, upper, u.M, m_idx, u.rows, row)
            if first and u.zero_c:
                total = summed
            else:
                total = arith.AddFOp(transfer_read(u.acc, u.c, at, u.acc_pad, True), summed).result
            transfer_write(total, u.c, at, True)
            if last and u.epi is not None:
                p08._build_epilogue(u.epi, total, at, True)
