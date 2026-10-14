"""tritonpytorchsim-lower-matmul-tpuv6e: every torchsim.dot onto vcix_tpuv6e's matrix unit (libtpuv6e.so),
an array 2 x lanes wide and deep fed by vmatprep/vmatpush/vmatmul and drained by vpop, the dot's
inputs one stream kept as far ahead of their pops as the unit's delay line and queues allow."""
import sys
from collections import namedtuple

from .... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from ...passes import lib_matmul as mm  # noqa: E402
from ...passes import lib_pack  # noqa: E402
from ...passes.lower import LOWER_MATMUL  # noqa: E402
from . import lib_mxu as mxu  # noqa: E402
from ...passes.lib_bank_layout import MIN_VEC, bank_view, index_bank_casts  # noqa: E402
from ...passes.lib_ir import (build_trip_loop, const_index, elem_bits, transfer_read,  # noqa: E402
                     transfer_write, zero_of)
from ...passes.lib_vcix import bitcast, i64 as a64, vcix  # noqa: E402

NAME = "tritonpytorchsim-lower-matmul-tpuv6e"

REQUIRES = ("vcix_tpuv6e",)

SLOT = LOWER_MATMUL.name

FORBIDS = LOWER_MATMUL.lowers

MARKERS = FORBIDS

# The vcix.iv / vcix.v.i opcodes of the unit's four instructions, and the bits of their imm.
VMATMUL, VMATPUSH, VMATPREP, VPOP = 0, 1, 2, 2
FORMAT_F32, FORMAT_F16, FORMAT_BF16 = 0, 1, 3
SECOND = 4      # a vmatprep of an input, not of weights; a vpop of the upper columns
LATCH = 4       # a vmatmul that first makes the staged weights the active
FIRST_HALF, SECOND_HALF = 0x08, 0x10    # which half of the depth a vmatpush stages, at [4:3]


def _simm5(field):
    """The field at 19:15 as the signed 5-bit immediate the op carries: bit 4 set reads negative."""
    return field - 32 if field & 0x10 else field


# The model's keys the lowering reads from the target's vcix block: each array's queue size, which
# the target must state, and its delay in cycles, which it may.
QUEUE_KEY, DELAY_KEY = "tpu_mxu_queue_entries", "tpu_mxu_delay_cycles"


def _stated(target):
    """(queue entries, delay cycles or None) from the target's vcix block; refused, by the key's
    name, when it does not state the queue size."""
    from ....contract.target import TargetError
    if QUEUE_KEY not in target.vcix:
        raise TargetError(f"{target.name}: the vcix block does not state {QUEUE_KEY}, which "
                          f"{NAME} schedules the matrix unit's stream by")
    return int(target.vcix[QUEUE_KEY]), (int(target.vcix[DELAY_KEY]) if DELAY_KEY in target.vcix
                                         else None)


def run(module, *, target, grid=(), **_):
    """Every matmul in `module` lowered in place, in three steps over the whole function.

    1. Shape: a loop around a dot (Inductor's mm template walks K in blocks) gives its first trip
       and its last apart (_peel_k_loops), the only trips whose C is known before or read after.
    2. What each dot needs, decided for every dot before any is rewritten (_facts), on two rules:
       nothing is computed that nobody reads or that is known -- the lane tiles past the
       operands' data, a C known to be zero, a zero fill every later write covers -- and a value
       is used where it is made -- the epilogue at the last pops, a panel the unit still holds.
    3. Each dot rewritten with its facts (_rewrite_matmul); the fills step 2 found dead go.
    `grid` is the launch grid, which bounds the program ids a masked load's clamp is made of."""
    from ...passes.lib_ir import walk
    vectorlane, vlen = target.vector_lanes, target.vlen_bits
    queue_entries, delay = _stated(target)
    index_bank_casts(module)
    _peel_k_loops(module)
    dots = [o for o in walk(module.operation) if o.name in mm.MATMUL_OPS]
    facts = [_facts(o, dots[i - 1] if i else None, vectorlane, vlen, grid)
             for i, o in enumerate(dots)]
    dead = mxu.dead_initializers([f.init for f in facts if f.init is not None])
    held = None
    for o, f in zip(dots, facts):
        held = _rewrite_matmul(o, vectorlane, vlen, queue_entries, delay, f,
                               held if f.follows else None)
    for nest in dead:
        nest.erase()
    return module


# What step 2 decides of a dot: its _zero_init proof (or None), the (N, K) lane tiles it computes,
# and whether it follows the dot before with the same weights (_follows).
_Facts = namedtuple("_Facts", "init live follows")


def _facts(op, before, lanes, vlen, grid):
    """_Facts of the dot `op`, `before` the matmul before it in program order, or None."""
    from ...passes.lib_indexing_maps import MATMUL_K, MATMUL_N, matmul_axis
    A, B = op.operands[0], op.operands[1]
    known = mxu.program_ids(op, grid)
    reach = (mxu.data_reach(op, A, matmul_axis(op, 0, MATMUL_K), known),
             mxu.data_reach(op, B, matmul_axis(op, 1, MATMUL_K), known),
             mxu.data_reach(op, B, matmul_axis(op, 1, MATMUL_N), known))
    return _Facts(_zero_init(op, lanes, vlen), _live_tiles(op, lanes, reach, known),
                  before is not None and _follows(before, op))


def _peel_k_loops(module):
    """Every scf.for whose body holds a matmul that carries C from trip to trip (_carries), of
    constant bounds and no loop-carried values, with its first trip and its last taken out of it:
    a kernel that walks K in blocks (Inductor's mm template, acc += dot over K) then has a dot
    before the loop and one after it. The first takes a C its fill proves zero as it is, and the
    fill goes; the last has the cast after the loop as its epilogue, run at its pops; and the K
    offset of each is a constant, so how far its operands' data reaches is known (_live_tiles).
    The trips between stay a loop."""
    from mlir.dialects import arith
    from ...passes.lib_ir import loop_bounds, trip_count, walk
    loops = [o for o in walk(module.operation) if o.name == "scf.for"
             and any(_carries(x.operation) for x in o.regions[0].blocks[0].operations)]
    for loop in loops:
        trips = trip_count(loop)
        if trips is None or trips < 2 or len(loop.results) or len(loop.operands) > 3:
            continue
        lb, _, step = loop_bounds(loop)
        ops = [o.operation for o in loop.block.operations]
        after = ops[next(k for k, o in enumerate(ops) if o == loop.operation) + 1]
        _clone_trip(loop, lb, ir.InsertionPoint(loop))
        _clone_trip(loop, lb + (trips - 1) * step, ir.InsertionPoint(after))
        if trips == 2:
            loop.erase()
            continue
        # The trips between, as a loop from 0 (the TOG's sample takes a loop's first trip to start
        # at 0) that adds the first of them to its induction variable.
        ty = loop.operands[0].type
        with ir.InsertionPoint(loop), loop.location:
            zero = arith.ConstantOp(ty, ir.IntegerAttr.get(ty, 0)).result
            hi = arith.ConstantOp(ty, ir.IntegerAttr.get(ty, (trips - 2) * step)).result
        loop.operands[0] = zero
        loop.operands[1] = hi
        block = loop.regions[0].blocks[0]
        iv = block.arguments[0]
        with ir.InsertionPoint(block.operations[0]), loop.location:
            off = arith.ConstantOp(ty, ir.IntegerAttr.get(ty, lb + step)).result
            at = arith.AddIOp(iv, off)
        for x in [o.operation for o in block.operations]:
            for y in [x] + walk(x):
                if y == at.operation:
                    continue
                for k, v in enumerate(list(y.operands)):
                    if v == iv:
                        y.operands[k] = at.result


def _carries(op):
    """Whether `op` is a matmul whose C one trip of its loop hands to the next: nothing in the loop
    body before it writes C (a loop that fills its C each trip, as one over N windows does, has
    nothing to take apart)."""
    return op.name in mm.MATMUL_OPS and mxu.last_writer(op, op.operands[2]) is None


def _clone_trip(loop, value, ip):
    """The body of `loop` once at `ip`, its induction variable the constant `value`."""
    from mlir.dialects import arith
    from ...passes.lib_ir import clone_ops
    block = loop.regions[0].blocks[0]
    iv = block.arguments[0]
    with ip, loop.location:
        at = arith.ConstantOp(iv.type, ir.IntegerAttr.get(iv.type, value)).result
    clone_ops([x.operation for x in block.operations][:-1], {iv: at}, ip)   # not the scf.yield


def _live_tiles(op, lanes, reach, known):
    """(N tiles, K tiles) the dot has to compute, of `reach` = how far A's and B's data reach
    along K and B's along N (mxu.data_reach). A K tile where A or B holds only its load's zeros
    adds nothing to any sum. An N tile where B holds only zeros computes zeros, and it is left out
    when nothing reads C there: C, and whatever an elementwise loop moves it to (the epilogue
    among them), is read past the dot only by stores that clamp to the same columns."""
    from ...passes.lib_indexing_maps import MATMUL_N, matmul_axis
    _, N, K = mm.get_mnk(op)
    n_tiles, k_tiles = -(-N // lanes), -(-K // lanes)
    k_live = min([k_tiles] + [max(1, -(-d // lanes)) for d in reach[:2] if d is not None])
    n_live = n_tiles
    if reach[2] is not None and max(1, -(-reach[2] // lanes)) < n_tiles:
        want = max(1, -(-reach[2] // lanes))
        if mxu.read_only_within(op, op.operands[2], matmul_axis(op, 2, MATMUL_N), want * lanes,
                                known):
            n_live = want
    return n_live, k_live


def _follows(before, op):
    """Whether `op` runs straight after the dot `before` with the same weights: the next matmul of
    its block, B the same buffer, and nothing between that writes it. The unit then still holds
    the panel `before` staged last, and a first panel of `op` that is that one is not staged
    again."""
    if before.operation.block != op.operation.block:
        return False
    a, b = before.operands[1], op.operands[1]
    if mxu.buffer_root(a) != mxu.buffer_root(b) or a.type != b.type:
        return False
    return mxu.used_between(b, before.operation, op.operation, writes=True) is None


def _zero_init(op, lanes, vlen):
    """lib_mxu's proof that C holds zero over every element the pops touch -- here
    all of C: its last writer in the block is a zero-splat store nest over them, or a nest copying
    them from a buffer whose own last writer is one. None otherwise."""
    C = op.operands[2]
    if (any(mxu.buffer_root(v) == mxu.buffer_root(C) for v in op.operands[:2])
            or not mxu.is_plain(C)):
        return None
    elt = ir.MemRefType(op.operands[0].type).element_type
    if elem_bits(elt) not in (16, 32):
        return None
    words = vlen // 32
    rows = words * (32 // elem_bits(elt))
    M, N, K = mm.get_mnk(op)
    if M % rows:
        return None
    runs = mxu.runs_of([n * M + m * rows + i for n in range(-(-N // lanes))
                      for m in range(M // rows) for i in range(0, rows, words)], words)
    if runs is None:
        return None
    writer = mxu.last_writer(op.operation, C)
    if writer is None:
        return None
    filled = mxu.store_nest(writer, C, None)
    if filled is not None and mxu.covers(filled, runs):
        return mxu.ZeroInit(op.operation, C, runs, None, (writer, filled))
    copied = mxu.store_nest(writer, C, True)
    if copied is None or not mxu.covers(copied[1], runs):
        return None
    Z = copied[0]
    source = mxu.last_writer(writer, Z)
    filled = None if source is None else mxu.store_nest(source, Z, None)
    if filled is None or not mxu.covers(filled, copied[1]):
        return None
    return mxu.ZeroInit(op.operation, C, runs, (writer, copied[1]), (source, filled))


def _format(elt):
    if isinstance(elt, ir.BF16Type):
        return FORMAT_BF16
    return {16: FORMAT_F16, 32: FORMAT_F32}[elem_bits(elt)]


def _refuse(op, M, K, lanes, rows, pack, queue, elt, acc_elt):
    """A dot the unit cannot take, whose short last register is narrower than a bank moves, or
    whose queues of `queue` rows hold less than one input."""
    if elem_bits(elt) not in (16, 32):
        raise mm.ArrayRefusal(
            f"the matrix unit takes words of one f32 or two rows of 16 bits, not {elt}: {op}")
    if elem_bits(acc_elt) != 32:
        raise mm.ArrayRefusal(f"the matrix unit pops f32, and C is {acc_elt}: {op}")
    for d, name in ((M, "M"), (K, "K")):
        if d % pack:
            raise mm.ArrayRefusal(
                f"matmul {name}={d} rows of {elt} end inside a word of {pack}, so its tiles "
                f"start inside one: {op}")
    words = rows // pack
    short = (("weights", _tail(K, lanes) % rows // pack), ("inputs", M % rows // pack),
             ("pops", M % rows % words))
    for what, n in short:
        if 0 < n < MIN_VEC:
            raise mm.ArrayRefusal(
                f"matmul M={M} K={K}: the last register of {what} holds {n} word(s), narrower "
                f"than the {MIN_VEC} a bank moves at the least (an M of 1 is a vector op): {op}")
    if queue < rows:
        raise mm.ArrayRefusal(
            f"the matrix unit's queues hold {queue} rows, less than one input of {rows}: {op}")


def _tail(d, tile):
    """The rows of the last tile of `d`: `tile` when `d` is whole tiles."""
    return (d - 1) % tile + 1


def _refuse_formats(elt, acc_elt, b_elt):
    """Element types the unit has no format for: its words are f32, two f16 or two bf16 (formats 0,
    1 and 3), one per push, and its pops are added into C."""
    if b_elt != elt:
        raise NotImplementedError(
            f"matmul operands A ({elt}) and B ({b_elt}) have "
            f"different element types; every push is typed from A's element type at one sew, "
            f"so B's elements would be read as A's")
    if isinstance(acc_elt, ir.BF16Type):
        raise NotImplementedError(
            f"a bf16 accumulator ({acc_elt}): the pop is added into C, and bf16 arithmetic "
            f"has no instruction on this machine; accumulate in f32")
    for t in (elt, acc_elt):
        if isinstance(t, ir.IntegerType):
            raise NotImplementedError(
                f"integer matmul ({t}) has no path to the matrix unit: its words "
                f"are f32, two f16 or two bf16 (vcix-accelerator examples/tpuv6e, "
                f"formats 0, 1 and 3), so there is no integer MAC to lower to")


def _rewrite_matmul(op, lanes, vlen, queue, delay, facts, held=None):
    """`op` on the unit with its _Facts, and the panel it leaves staged (see _build_dot). A
    dimension short of its tile takes the whole last tile, as the v6e pads: K past it is zero
    weights and zero lanes of A, N past it lands in lanes of C no other value holds, M past it is
    zero words of the last input, whose pops are dropped. Lane tiles past the operands' data
    (facts.live) are neither staged, fed nor popped."""
    from mlir.dialects import arith

    A, B, C = op.operands[0], op.operands[1], op.operands[2]
    mtA, mtB, mtC = (ir.MemRefType(x.type) for x in (A, B, C))
    elt, acc_elt = mtA.element_type, mtC.element_type
    M, N, K = mm.get_mnk(op)
    mm.refuse_batched(op)
    _refuse_formats(elt, acc_elt, mtB.element_type)
    mm.refuse_accumulator_in_its_loop(op)
    words = vlen // 32
    pack = 32 // elem_bits(elt) if elem_bits(elt) in (16, 32) else 0
    rows = words * pack
    _refuse(op, M, K, lanes, rows, pack, queue, elt, acc_elt)

    i32 = ir.IntegerType.get_signless(32)
    word_ty = ir.VectorType.get([words], i32)
    inputs = -(-M // rows)
    n_tiles, k_tiles = -(-N // lanes), -(-K // lanes)
    n_live, k_live = facts.live
    # The last K tile computed is short only when it is K's own last tile.
    k_rows = _tail(K, lanes) if k_live == k_tiles else lanes
    u = _Unit(lanes=lanes, words=words, rows=rows, pack=pack, fmt=_format(elt), M=M, K=K,
              vec=ir.VectorType.get([rows], elt), word=word_ty,
              acc=ir.VectorType.get([words], acc_elt), inputs=inputs,
              last_rows=_tail(M, rows), k_rows=k_rows, live=None)
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
        if u.k_rows < lanes:
            u.live = _live_lanes(word_ty, u.k_rows)

    # Lane tiles go to the array in pairs; an odd last N tile goes with zeros for its upper columns,
    # an odd last K tile stages the first half of the depth alone. The dot's inputs are one stream
    # over every (N pair, K pair) in turn: the input `lag` behind the one pushed is popped before
    # the push, weights change while results are on their way, and what is left is popped at the
    # end -- the unit's delay is waited out once a dot. Which pair a pop's rows are of, so which
    # columns of C they go to, whether C is read under them and whether the epilogue runs on them,
    # is settled here, pair by pair, and the loops are cut where it changes: no input tests it at
    # run time.
    n_parts, k_parts = _pairs(n_live, lanes, lanes), _pairs(k_live, u.k_rows, lanes)
    pairs_total = sum(part[1] for part in k_parts)
    u.zero_c = facts.init is not None
    # The epilogue is matched against all of C: past the live N tiles it reads what only stores
    # that clamp those columns away read on (_live_tiles).
    g = mxu.Pops(rows, words, ((n_tiles, pairs_total, inputs), K, M, rows, lanes))
    u.epi = None
    if u.last_rows == rows:
        u.epi = mxu.find_epilogue(op, g, mxu.GemmOperand(u.c, u.acc, u.acc_pad), ip)
    if u.epi is not None:
        mxu.build_waits(ip, u.epi)
    held = _build_dot(u, ip, n_parts, k_parts, inputs, _lag(lanes, words, rows, queue, delay),
                      held)
    if u.epi is not None:
        u.epi.nest.erase()
    op.erase()
    return held


def _lag(lanes, words, rows, queue, delay):
    """Inputs pushed before the first pop: the delay line at a word a cycle plus two, at most what
    the queues hold. `delay` is the target's; unstated, the array's width + depth - 1, the slots
    the model itself gives an unstated tpu_mxu_delay_cycles (examples/tpu/mxu.hpp)."""
    delay = 4 * lanes - 1 if delay is None else delay
    by_delay = -(-(delay + words) // words) + 1
    return max(1, min(by_delay, 2 * queue // rows))


# A pop before an input: the rows of input `row` of the N pair `back` before this one, `first` when
# that is the first K pair of its N pair (C holds nothing yet), `last` when the last (the sum is done).
_Pop = namedtuple("_Pop", "row back first last")


def _build_dot(u, ip, n_parts, k_parts, inputs, lag, held=None):
    """The stream: every (N pair, K pair) in turn, a panel and its inputs each, then the drain.
    Position p of the stream is input p % inputs of pair p // inputs; what is popped before it is
    position p - lag. Each N pair is emitted on its own (see n0); within one, K pairs with the same
    plan are one loop. `held` is the panel the dot before left staged, when this one follows it
    with the same weights (_follows): a first panel that is that one is not staged again, its first
    input latching what is there. The panel this dot leaves staged is returned."""
    pt = sum(part[1] for part in k_parts)
    last_k = pt - 1
    n_pairs = n_parts[0][1] if n_parts[0][2] else 0
    odd_n = None if n_parts[-1][2] else n_parts[-1]
    n_count = n_pairs + (1 if odd_n else 0)
    total = n_count * pt * inputs
    lag = min(total, lag)
    # Every N pair is emitted on its own. From the N pair where every position is past `lag` on,
    # the pairs could be one loop (which N pair a pop is of is then a distance, which K pair a
    # remainder), but with that loop's body LLVM runs short of registers and rebuilds each access's
    # address from a base beside the load, which the in-order core waits out: the K pairs that read
    # C went from 14 to 17 cycles an input, and the dot gained nothing from its one stream. Emitted
    # apart, every such loop keeps its incremented addresses (1024^3 bf16: 14763 -> 14081 cycles).
    n0 = n_pairs

    def n_tile_of(i):
        return 2 * i if i < n_pairs else odd_n[0]

    k_pairs = [(first + 2 * t, upper, rows) for first, pairs, upper, rows in k_parts
               for t in range(pairs)]

    def staged(i, q):
        """What K pair `q` of N pair `i` stages: its tiles, their rows and its format."""
        return (n_tile_of(i), i < n_pairs) + k_pairs[q] + (u.fmt, u.K)
    reuse = held is not None and held == staged(0, 0)

    def plan(i, q):
        """The inputs of K pair `q` of N pair `i` as segments (m_lo, m_hi, pop): before input m,
        nothing, or the rows of input `pop.row + m - m_lo` of the pair the pop is of. A short last
        input, and the pops of one, are segments of their own."""
        segments = []
        for m in range(inputs):
            p = (i * pt + q) * inputs + m
            if p < lag:
                key, pop = (None, u.short(m)), None
            else:
                b = p - lag
                n_b, k_b = divmod(b // inputs, pt)
                key = (i - n_b, k_b, u.short(m), u.short(b % inputs))
                pop = _Pop(b % inputs, i - n_b, k_b == 0, k_b == last_k)
            if segments and segments[-1][2] == key:
                segments[-1][1] = m + 1
            else:
                segments.append([m, m + 1, key, pop])
        return tuple((lo, hi, pop) for lo, hi, _, pop in segments)

    # With every N pair's code its own, the panel after an N pair's last is known, and is staged
    # under it as any other (see _build_pair); in a loop of N pairs it is not.
    across = n0 == n_pairs

    def ahead_of(q, n_tile, n_upper, n_index):
        """The panel staged under the inputs of K pair `q`, past its part's last pair: the
        stream's next, as (N tile, upper, K tile, upper, rows), or None at the dot's end."""
        if q + 1 < pt:
            return (n_tile, n_upper) + k_pairs[q + 1]
        if across and n_index is not None and n_index + 1 < n_count:
            return (n_tile_of(n_index + 1), n_index + 1 < n_pairs) + k_pairs[0]
        return None

    def epilogue_at(i, q):
        """Whether K pair `q` of N pair `i` pops an N pair's last K pair, the epilogue run at those
        pops: its inputs carry the epilogue's work, and its next panel is staged as a whole after
        them, not among them."""
        return u.epi is not None and any(pop is not None and pop.last for _, _, pop in plan(i, q))

    def staged_by_before(i, q, n_index):
        """Whether the pair before K pair `q` of N pair `i` in the stream staged its panel."""
        if q > 0:
            return not epilogue_at(i, q - 1)
        return across and n_index is not None and n_index > 0 and not epilogue_at(i - 1, pt - 1)

    def emit_n(n_ip, i, n_tile, n_upper, n_index):
        """The K pairs of one N pair. `i` stands for the pair in the plans: the pair itself, or
        the first of the looped ones. `n_index` is the pair's number when the code is its own,
        None in the loop, where the pair a pop is of is `back` pairs below the loop's. A loop of
        K pairs stages each next one; a part's last pair is emitted on its own, as what it stages
        is another part's, another N pair's or none."""
        from mlir.dialects import arith

        def target(dn):
            if n_index is not None:
                return const_index(n_tile_of(n_index - dn)), n_index - dn < n_pairs
            return (n_tile if dn == 0 else arith.SubIOp(n_tile, const_index(2 * dn)).result), True
        before = 0
        for k_first, part_pairs, k_upper, k_rows in k_parts:
            q = before
            end = before + part_pairs
            while q < end:
                pl, hi = plan(i, q), q + 1
                kept = reuse and n_index == 0 and q == 0    # its panel is in the unit already
                staged_before = staged_by_before(i, q, n_index)
                apart = epilogue_at(i, q)
                while (not kept and not apart and q + 1 < end and hi + 1 < end
                       and plan(i, hi) == pl):
                    hi += 1
                k_ip, k_tile = _trip_loop(n_ip, k_first + 2 * (q - before), hi - q, 2)
                if apart:
                    ahead = None
                elif q + 1 < end:
                    with k_ip:
                        ahead = (n_tile, n_upper, arith.AddIOp(k_tile, const_index(2)).result,
                                 k_upper, k_rows)
                else:
                    ahead = ahead_of(q, n_tile, n_upper, n_index)
                _build_pair(u, k_ip, n_tile, n_upper, k_tile, k_upper, k_rows, pl, target,
                            not kept and not staged_before, ahead)
                q = hi
            before += part_pairs
    for i in range(n0):
        emit_n(ip, i, const_index_at(ip, n_tile_of(i)), True, i)
    if n0 < n_pairs:
        l_ip, n_tile = _trip_loop(ip, 2 * n0, n_pairs - n0, 2)
        emit_n(l_ip, n0, n_tile, True, None)
    if odd_n:
        emit_n(ip, n_pairs, const_index_at(ip, odd_n[0]), False, n_pairs)
    for b_lo, b_hi, (g_b, short) in _plan_drain(total, lag, inputs, u.short):
        n_b, k_b = divmod(g_b, pt)
        d_ip, row = _trip_loop(ip, b_lo % inputs, b_hi - b_lo, 1)
        with d_ip:
            _build_pops(u, const_index(n_tile_of(n_b)), row, n_b < n_pairs, k_b == 0, k_b == last_k,
                        u.last_rows if short else u.rows)
    return staged(n_count - 1, last_k)


def _plan_drain(total, lag, inputs, short):
    """What the stream leaves to pop, as runs (b_lo, b_hi, (pair, short)) over the stream's
    positions; a short last input's pops are a run of their own."""
    out = []
    for b in range(total - lag, total):
        key = (b // inputs, short(b % inputs))
        if out and out[-1][2] == key:
            out[-1][1] = b + 1
        else:
            out.append([b, b + 1, key])
    return [tuple(x) for x in out]


def _build_pair(u, ip, n_tile, n_upper, k_tile, k_upper, k_rows, segments, target, weights=True,
                ahead=None):
    """A panel of weights -- none when `weights` is False, the unit holding it from the dot before
    or from the pair before -- then the pair's inputs segment by segment, each behind the pops its
    segment says; `target(back)` is the N tile and whether it has an upper tile of the pair a pop
    is of. The first input is emitted on its own, as it latches the panel. `ahead`, the stream's
    next panel (N tile, upper, K tile, upper, rows) or None, is staged under these inputs, as the
    v6e pushes the next tile between the vmatmuls of the one in use (_Staging)."""
    from mlir.dialects import arith
    st = _Staging(u, ahead) if ahead is not None else None

    def row_of(m, seg_lo, pop):
        """The row popped before input `m` (an int, or an index value)."""
        if isinstance(m, int):
            return const_index(pop.row + m - seg_lo)
        if pop.row == seg_lo:
            return m
        if pop.row > seg_lo:
            return arith.AddIOp(m, const_index(pop.row - seg_lo)).result
        return arith.SubIOp(m, const_index(seg_lo - pop.row)).result

    if weights:
        with ip:
            _build_weights(u, n_tile, k_tile, n_upper, k_upper, k_rows)
    for seg_lo, seg_hi, pop in segments:
        m = seg_lo
        while m < seg_hi:
            # Inputs [m, hi) are one loop body: the first input alone, then runs of inputs that
            # stage the same (_Staging.key), then the inputs past the staging.
            staging = st is not None and m < st.end
            hi = m + 1
            if m > 0:
                stop = min(seg_hi, st.end) if staging else seg_hi
                while hi < stop and (not staging or st.key(hi) == st.key(m)):
                    hi += 1
            at, mv = _trip_loop(ip, m, hi - m, 1)
            if hi - m == 1:
                mv = m
            if pop is not None:
                with at:
                    row = row_of(mv, seg_lo, pop)
                    tile, upper = target(pop.back)
                    _build_pops(u, tile, row, upper, pop.first, pop.last, u.rows_of(pop.row))
            with at:
                _build_input(u, k_tile, const_index(mv) if isinstance(mv, int) else mv, k_upper,
                             k_rows, mv == 0 if isinstance(mv, int) else False, u.rows_of(m))
                if staging:
                    st.build(m, None if hi - m == 1 else mv)
            m = hi


class _Staging:
    """How the panel `ahead` is staged under a pair's `u.inputs` inputs: its P prep/push pairs
    (lanes // rows a half of the depth, the first half then the second, deepest first, as
    _build_weights stages them), `per` = ceil(P / inputs) after each input from the first (which
    latched) on until all are: a pair an input where there are as many inputs, so each input's
    loop body stays one input."""

    def __init__(self, u, ahead):
        self.u = u
        self.n_tile, self.n_upper, self.k_tile, self.k_upper, self.k_rows = ahead
        self.half = u.lanes // u.rows
        self.P = (2 if self.k_upper else 1) * self.half
        self.per = -(-self.P // u.inputs)
        self.end = -(-self.P // self.per)
        self.held = (u.lanes if self.k_upper else 0) + self.k_rows

    def _pairs(self, m):
        """(the half's imm bits, the row, of its rows how many are B's) of each pair after input m."""
        u, out = self.u, []
        for j in range(m * self.per, min(self.P, (m + 1) * self.per)):
            h, k = divmod(j, self.half)
            row = h * u.lanes + u.lanes - u.rows - u.rows * k
            out.append((SECOND_HALF if h else FIRST_HALF, row,
                        max(0, min(u.rows, self.held - row))))
        return out

    def key(self, m):
        """What input m stages but where: each pair's half and how many of its rows are B's."""
        return tuple((half, live) for half, _, live in self._pairs(m))

    def build(self, m, at):
        """The pairs after input m, or with `at` (an index value, m and on) after the input `at` of
        a loop over inputs that stage as m does."""
        from mlir.dialects import arith
        u = self.u
        idx = lambda v: const_index(v) if isinstance(v, int) else v  # noqa: E731
        n_tile, k_tile = idx(self.n_tile), idx(self.k_tile)
        k_at = arith.MulIOp(k_tile, const_index(u.lanes // u.pack)).result
        step = u.rows * self.per // u.pack
        for half, row, live in self._pairs(m):
            off = row // u.pack
            if at is not None:
                off = arith.SubIOp(const_index(off + step * m),
                                   arith.MulIOp(at, const_index(step)).result).result
            _push_pair(u, n_tile, k_at, self.n_upper, half, off, live)


class _Unit:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def short(self, m):
        """Is input `m` the last and shorter than a register?"""
        return m == self.inputs - 1 and self.last_rows < self.rows

    def rows_of(self, m):
        """The rows of input `m`; only a short one's segment starts at it."""
        return self.last_rows if self.short(m) else self.rows


def const_index_at(ip, v):
    with ip:
        return const_index(v)


def _pairs(tiles, last, lanes):
    """(first tile, trips, whether each trip has an upper tile, rows of each trip's last tile): the
    pairs, then an odd last tile; a last pair whose upper tile holds `last` < `lanes` rows apart."""
    whole, odd = tiles // 2, tiles % 2
    apart = not odd and last < lanes
    out = []
    if whole - apart:
        out.append((0, whole - apart, True, lanes))
    if apart:
        out.append((tiles - 2, 1, True, last))
    if odd:
        out.append((tiles - 1, 1, False, last))
    return out


def _live_lanes(word_ty, rows):
    """Which lanes of a register hold one of a short last K tile's `rows`."""
    from mlir.dialects import arith, vector
    idx_ty = ir.VectorType.get(ir.VectorType(word_ty).shape, ir.IndexType.get())
    lane = ir.Operation.create("torchsim.vlane_idx", results=[idx_ty], operands=[],
                               attributes={}).results[0]
    return arith.CmpIOp(arith.CmpIPredicate.ult, lane,
                        vector.BroadcastOp(idx_ty, const_index(rows)).result).result


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
    return [const_index(0), arith.AddIOp(flat, const_index(at) if isinstance(at, int) else at).result]


def _words(u, bank, at, rows=None):
    """One register of the operand at `at` -- an offset in words where the bank is packed -- as
    the words the unit takes; of `rows` < a register's, the rest zero words and never read."""
    from mlir.dialects import vector
    rows = u.rows if rows is None else rows
    if rows == 0:
        return u.zero
    n = rows // u.pack
    if u.pack == 1:
        got = bitcast(transfer_read(ir.VectorType.get([n], u.vec.element_type), bank, at, u.pad,
                                    True), ir.VectorType.get([n], u.word.element_type))
    else:
        got = transfer_read(ir.VectorType.get([n], u.word.element_type), bank, at, u.word_pad, True)
    if n == u.words:
        return got
    return vector.InsertStridedSliceOp(got, u.zero, [0], [1]).result


def _build_weights(u, n_tile, k_tile, n_upper, k_upper, k_rows):
    """A panel staged as the v6e stages it: the first half of the depth, then the second (none with
    no upper K tile, XLA's K = 128), a register of rows at a time, deepest first; the vmatprep has
    the upper N tile's columns. Rows past the last tile's `k_rows`, or of no tile, are zeros."""
    from mlir.dialects import arith
    # Offsets are in the bank's elements, words where a word is `pack` of B's.
    k_at = arith.MulIOp(k_tile, const_index(u.lanes // u.pack)).result
    held = (u.lanes if k_upper else 0) + k_rows
    for half, base in ((FIRST_HALF, 0), (SECOND_HALF, u.lanes)):
        if base and not k_upper:
            continue
        for row in range(base + u.lanes - u.rows, base - 1, -u.rows):
            _push_pair(u, n_tile, k_at, n_upper, half, row // u.pack,
                       max(0, min(u.rows, held - row)))


def _push_pair(u, n_tile, k_at, n_upper, half, at, live):
    """One register of rows of a panel at in-tile offset `at` (an int or an index value) into the
    `half` of the depth: the vmatprep with the upper N tile's columns, then the vmatpush; `live`
    of its rows are B's, the rest zeros."""
    extent = u.K // u.pack
    upper = (_words(u, u.b, _at(n_tile, True, extent, k_at, 1, at), live) if n_upper
             else u.zero)
    vcix("vcix.iv", [upper, u.rvl], [],
         {"opcode": a64(VMATPREP), "imm": a64(u.fmt), "rd": a64(0)})
    lower = _words(u, u.b, _at(n_tile, False, extent, k_at, 1, at), live)
    vcix("vcix.iv", [lower, u.rvl], [],
         {"opcode": a64(VMATPUSH), "imm": a64(_simm5(u.fmt | half)), "rd": a64(0)})


def _build_input(u, k_tile, m_idx, k_upper, k_rows, latch, m_rows):
    """One register of `m_rows` rows of A: the vmatprep carries the upper K tile's lanes, zeros
    where the pair has none, and the vmatmul the lower; the pair's last tile is zero in the lanes
    past its `k_rows`. One that latches makes the staged weights active."""
    from mlir.dialects import arith
    extent, span = u.M // u.pack, u.rows // u.pack

    def lanes_of(words, last):
        if last and k_rows < u.lanes:
            return arith.SelectOp(u.live, words, u.zero).result
        return words
    upper = (lanes_of(_words(u, u.a, _at(k_tile, True, extent, m_idx, span, 0), m_rows), True)
             if k_upper else u.zero)
    vcix("vcix.iv", [upper, u.rvl], [],
         {"opcode": a64(VMATPREP), "imm": a64(u.fmt | SECOND), "rd": a64(0)})
    lower = lanes_of(_words(u, u.a, _at(k_tile, False, extent, m_idx, span, 0), m_rows),
                     not k_upper)
    vcix("vcix.iv", [lower, u.rvl], [],
         {"opcode": a64(VMATMUL), "imm": a64(u.fmt | (LATCH if latch else 0)), "rd": a64(0)})


def _build_pops(u, n_tile, m_idx, n_upper, first, last, m_rows):
    """One vmatmul's rows, a register of f32 at a time, lower N tile's columns then upper's, added
    into C (written as is at the first K pair of a C proven zero), the epilogue at the last; an odd
    N tile's upper half and rows past `m_rows` are popped and dropped."""
    from mlir.dialects import arith, vector
    for row in range(0, u.rows, u.words):
        live = max(0, min(u.words, m_rows - row))
        for upper in (False, True):
            got = bitcast(vcix("vcix.v.i", [u.rvl], [u.word],
                               {"opcode": a64(VPOP), "imm": a64(SECOND if upper else 0),
                                "rs2": a64(0)}).results[0], u.acc)
            if (upper and not n_upper) or not live:
                continue
            if live < u.words:
                got = vector.ExtractStridedSliceOp(ir.VectorType.get([live], u.acc.element_type), got,
                                                 [0], [live], [1]).result
            at = _at(n_tile, upper, u.M, m_idx, u.rows, row)
            if first and u.zero_c:
                total = got
            else:
                total = arith.AddFOp(transfer_read(got.type, u.c, at, u.acc_pad, True), got).result
            transfer_write(total, u.c, at, True)
            if last and u.epi is not None:
                mxu.build_epilogue(u.epi, total, at)
