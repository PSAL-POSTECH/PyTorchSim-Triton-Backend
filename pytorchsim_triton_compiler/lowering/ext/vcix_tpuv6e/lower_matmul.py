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
from ...passes.lib_bank_layout import (BANK_VIEW_OP, MIN_VEC, bank_view,  # noqa: E402
                                       index_bank_casts)
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
# vcix.v.sv (sf.vc.v.vv) opcode 2: the misc unit's bf16 pack of two f32 registers, vs2's elements then vs1's
PACK = 2
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


def run(module, *, target, grid=(), lmul=None, dot_lmul=None, dot_widths=None, **_):
    """Every matmul in `module` lowered in place, in three steps over the whole function.

    1. Shape: a loop around a dot (Inductor's mm template walks K in blocks) gives its first trip
       and its last apart (_peel_k_loops), the only trips whose C is known before or read after.
    2. What each dot needs, decided for every dot before any is rewritten (_facts), on two rules:
       nothing is computed that nobody reads or that is known -- the lane tiles past the
       operands' data, a C known to be zero, a zero fill every later write covers -- and a value
       is used where it is made -- the epilogue at the last pops, a panel the unit still holds.
    3. Each run of dots on the same weights (_groups) rewritten as one stream; the fills step 2
       found dead go.
    `grid` is the launch grid, which bounds the program ids a masked load's clamp is made of. A
    dot's pushes, inputs and pops each take a group of registers (_group): `lmul` is the kernel's
    LMUL and `dot_lmul` the retry's, by the dot's number in program order; the LMUL each dot was
    built at goes to `dot_widths`, by that number."""
    from ....machine import ISA_MAX_LMUL
    from ...passes.lib_ir import walk
    vectorlane, vlen = target.vector_lanes, target.vlen_bits
    queue_entries, delay = _stated(target)
    index_bank_casts(module)
    _peel_k_loops(module)
    dots = [o for o in walk(module.operation) if o.name in mm.MATMUL_OPS]
    lmul = ISA_MAX_LMUL if lmul is None else lmul
    groups_of = [_group(o, vectorlane, vlen, queue_entries, min(lmul, (dot_lmul or {}).get(n, lmul)))
                 for n, o in enumerate(dots, 1)]
    if dot_widths is not None:
        dot_widths.update(enumerate(groups_of, 1))
    widths = [vlen * g for g in groups_of]
    places = [ir.Location.file(mm.DOT_FILE, n, 0) for n in range(1, len(dots) + 1)]
    facts = [_facts(o, dots[i - 1] if i else None, vectorlane, w, grid)
             for i, (o, w) in enumerate(zip(dots, widths))]
    dead = mxu.dead_initializers([f.init for f in facts if f.init is not None])
    units = []
    for o, w, f, at in zip(dots, widths, facts, places):
        with at:
            units.append(_prepare(o, vectorlane, w, queue_entries, f))
    held = None
    groups, why = _groups(dots, facts, units, dead)
    for group in groups:
        u = units[group[0]]
        at = dots[group[-1]].operation
        block = [o.operation for o in at.block.operations]
        before = block[block.index(at) - 1] if block.index(at) else None
        # At the last dot, where everything the run reads is defined, and at its place (DOT_FILE).
        with places[group[-1]]:
            held = _build_dot([units[i] for i in group], ir.InsertionPoint(at),
                              u.n_parts, u.k_parts, u.inputs,
                              _lag(vectorlane, u.words, u.rows, queue_entries, delay),
                              held if facts[group[0]].follows else None)
        if group[0] in why:
            _stamp_apart(before, at, why[group[0]])
        for i in group:
            _finish(dots[i], units[i])
    for nest in dead:
        nest.erase()
    return module


def _group(op, lanes, vlen, queue, lmul):
    """The registers a push, an input and a pop of `op` each take, its LMUL: at most `lmul` (the
    kernel's, or the retry's for this dot), and no more than a push stages (a lane tile of rows,
    half the depth), the queues hold, or the dot's M fills, nor one whose short last group the unit
    refuses (_refuse). An epilogue that runs at the pops (find_epilogue) runs at a pop's elements:
    in its widest type they fit the registers an instruction reaches, as legalize fits a loop's
    (p06_legalize._choose_widths). Whether the stream's registers fit is llc's to say: a spill in
    it narrows the dot (pipeline._narrow_for)."""
    from ....machine import ISA_MAX_LMUL
    A, C = op.operands[0], op.operands[2]
    elt, acc_elt = ir.MemRefType(A.type).element_type, ir.MemRefType(C.type).element_type
    M, N, K = mm.get_mnk(op)
    pack = 32 // elem_bits(elt) if elem_bits(elt) in (16, 32) else 1
    got = lmul
    while got > 1:
        words = vlen * got // 32
        rows = words * pack
        if rows <= lanes and rows <= queue and rows // 2 < M:
            try:
                _refuse(op, M, K, lanes, rows, pack, queue, elt, acc_elt)
                g = mxu.Pops(rows, words, ((-(-N // lanes), None, -(-M // rows)), K, M, rows, lanes))
                walk = mxu.runs_at_pops(op, g, ir.VectorType.get([words], acc_elt))
                if walk is None or words * mxu.widest_element_bits(walk) <= vlen * ISA_MAX_LMUL:
                    break
            except mm.ArrayRefusal:
                pass
        got //= 2
    return got


def _groups(dots, facts, units, dead):
    """The dots as runs in order, a dot joining the one before when it follows it (_follows) and
    nothing keeps them apart (_apart); and, by dot, why one that follows was kept apart."""
    out, why = [], {}
    for i, o in enumerate(dots):
        reason = _apart(dots[i - 1], o, units[i - 1], units[i], dead) if facts[i].follows else ""
        if i and reason is None:
            out[-1].append(i)
            continue
        if i and reason:
            why[i] = reason
        out.append([i])
    return out, why


def _stamp_apart(before, at, reason):
    """`reason` stamped as KEPT_APART on the first loop of the stream built between `before` and
    `at` (on its first op of no result when it has none: what makes a value can fold away)."""
    ops = [o.operation for o in at.block.operations]
    new = ops[(ops.index(before) + 1 if before is not None else 0):ops.index(at)]
    o = next((x for x in new if x.regions), next((x for x in new if not len(x.results)), new[0]))
    o.attributes[mm.KEPT_APART] = ir.StringAttr.get(f"not one stream with the dot before: {reason}")


def _plan(u):
    """What a dot's stream is but for its A, C and what follows C."""
    return (u.M, u.K, u.rows, u.inputs, u.last_rows, u.k_rows, u.fmt, u.n_parts, u.k_parts)


_VIEWS = (lib_pack.PACK_OP, BANK_VIEW_OP)


def _apart(before, op, ub, uo, dead):
    """Why `op` cannot join `before`'s stream, or None: a different plan, an op between them that is
    not `before`'s epilogue or store, a dead fill, or of no effect, or that epilogue writing `op`'s."""
    if _plan(ub) != _plan(uo):
        return "a different plan"
    ops = [o.operation for o in op.operation.block.operations]
    a, b = ops.index(before.operation), ops.index(op.operation)
    kept = [x for x in (ub.epi.nest if ub.epi is not None else None,
                        ub.split.store if ub.split is not None else None) if x is not None]
    gone = [d.operation if hasattr(d, "operation") else d for d in dead]
    for o in ops[a + 1:b]:
        if any(o == k or mxu.within(o, k) for k in kept) or any(o == d for d in gone):
            continue
        if o.regions or not o.name.startswith(_PURE + _VIEWS + ("arith.", "index.")):
            return f"{o.name} between them"
    if ub.epi is not None:
        outs = [mxu.buffer_root(v) for v in ub.epi.outputs]
        if any(mxu.buffer_root(v) in outs for v in op.operands[:3]):
            return "the epilogue before writes an operand"
    return None


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


def _bare_bf16_cast(epi):
    """Is the epilogue nothing but a cast of the sum to bf16, stored?"""
    body = list(epi.body)
    return (not epi.inputs and len(body) == 1 and body[0].name == "arith.truncf"
            and str(ir.VectorType(body[0].results[0].type).element_type) == "bf16"
            and epi.stored == body[0].results[0])


def _read_after(op, buf, skip):
    """Whether an op after `op` -- later in its block, then in each block around it, a loop around
    it too (its next trip) -- reads `buf`'s buffer, `skip` (an op, with what it holds) aside; one
    that does not touch its data (a dealloc, touches_data) is no reader."""
    from ...passes.lib_ir import touches_data, walk, writes_through_this_operand
    root = mxu.buffer_root(buf)

    def reads(o):
        for x in [o] + [y for y in walk(o) if y != o]:
            if skip is not None and mxu.within(x, skip):
                continue
            for k, v in enumerate(list(x.operands)):
                if isinstance(v.type, ir.MemRefType) and mxu.buffer_root(v) == root \
                        and touches_data(x, mxu._Use(k)) \
                        and not writes_through_this_operand(x, mxu._Use(k)):
                    return True
        return False
    o = op.operation
    while o is not None and o.name != "func.func":
        ops = [x.operation for x in o.block.operations]
        at = next(i for i, x in enumerate(ops) if x == o)
        if any(reads(x) for x in ops[at + 1:]):
            return True
        o = mxu._parent(o)
        o = o.operation if o is not None and hasattr(o, "operation") else o
        if o is not None and o.name in ("scf.for", "affine.for") and reads(o):
            return True
    return False


def _zero_init(op, lanes, vlen):
    """lib_mxu's proof that C holds zero over every element the pops touch -- here
    all of C: its last writer in the block is a zero-splat store nest over them. None otherwise.
    (A zero C is filled where it is written: torchsim-bufferize makes a zero fill again before each
    dot that takes it, so none is copied in.)"""
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
    filled = mxu.store_nest(writer, C)
    if filled is None or not mxu.covers(filled, runs):
        return None
    return mxu.ZeroInit(op.operation, C, runs, (writer, filled))


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


def _prepare(op, lanes, vlen, queue, facts):
    """The _Unit `op`'s stream is built from (_build_dot). A dimension short of its tile takes the
    whole last tile, as the v6e pads (K: zero weights and A lanes; N: unread lanes of C; M: zero
    words, pops dropped); lane tiles past the operands' data (facts.live) are not computed."""
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
    # The sum's last write to C is left out when nothing after the dot reads C but its epilogue,
    # which takes the sum at the pops: C would hold what nobody reads.
    u.c_read_after = u.epi is None or _read_after(op, C, u.epi.nest)
    if u.epi is not None:
        mxu.build_waits(ip, u.epi)
    # An epilogue that only casts the f32 sum to bf16 goes through the unit's pack: two registers of one N
    # tile, end to end in the buffer, rounded and packed two to a word -- the v6e's vpack -- and stored as
    # words, so the casts take no 16-bit vtype and a pop run no vsetvli each way (see _build_pops).
    u.dpack = None
    if (u.epi is not None and rows == 2 * words and _bare_bf16_cast(u.epi)):
        with ip:
            u.dpack = lib_pack.pack(u.epi.d.value, op)
    # The store of what the dot made goes out an N pair at a time, as each pair's last pops land.
    u.split = _SplitStore.of(op, u.epi.d.value if u.epi is not None and u.epi.d is not None
                             else C, u.epi, lanes)
    u.n_parts, u.k_parts, u.n_live = n_parts, k_parts, n_live
    # The epilogue's rows the pops are in (mxu.Row), by in-lane row: opened before a pair's first
    # last pops, closed after its last.
    u.open = {}
    return u


def _finish(op, u):
    """`op` gone once its stream is built: its store covered by the N pairs', its epilogue run at
    its pops."""
    if u.split is not None:
        u.split.done(u.n_live)
    if u.open:
        raise RuntimeError(f"{NAME}: the epilogue's rows {sorted(u.open)} were never closed")
    if u.epi is not None:
        u.epi.nest.erase()
    op.erase()


# Ops with no effect a store's operands may be made of: moved before the dot, they are the same.
_PURE = ("arith.", "index.", "affine.apply", "memref.reinterpret_cast", "memref.cast",
         "memref.subview", "memref.get_global")


def _hoist_operands(store, dot, after):
    """Whether every value `store` takes is defined before `dot` once the ops among `after` (the
    ops after `dot`, in order) that only compute them are moved in front of it; moves them."""
    later = {o: i for i, o in enumerate(after)}
    need, seen = [], set()

    def visit(v):
        o = v.owner
        o = o.operation if hasattr(o, "operation") else o
        if not isinstance(o, ir.Operation) or o not in later or o in seen:
            return True
        if not o.name.startswith(_PURE) or o.regions:
            return False
        seen.add(o)
        if not all(visit(x) for x in o.operands):
            return False
        need.append(o)
        return True
    if not all(visit(v) for v in store.operands):
        return False
    for o in sorted(need, key=lambda o: later[o]):
        o.move_before(dot)
    return True


class _SplitStore:
    """The one store of a dot's result -- the epilogue's buffer, or C -- cut into one store an N
    pair: emitted where the pair's last pops have written its columns, it drains them while the
    stream goes on, as the v6e writes results back while it computes. A store that moves its whole
    tile, after the dot in its block with nothing between reading or writing the buffer; one the
    descriptor already cuts, gathers or strides at run time is left as it is."""

    def __init__(self, store, lanes):
        from ...passes.lib_bank_layout import bank_strides
        self.store, self.lanes = store, lanes
        tx = store.opview
        self.shape = list(ir.MemRefType(tx.sram.type).shape)
        self.stride0 = bank_strides(tx.sram)[0]
        self.dram_stride0 = ir.IntegerAttr(ir.ArrayAttr(tx.dram_stride)[0]).value
        self.masked = ([] if tx.masked_axes is None
                       else [ir.IntegerAttr(a).value for a in ir.ArrayAttr(tx.masked_axes)])
        self.emitted = []

    @classmethod
    def of(cls, op, out, epi, lanes):
        from ....dialects.torchsim import TransferOp, is_load
        from ...passes.lib_ir import walk
        root = mxu.buffer_root(out)
        ops = [o.operation for o in op.operation.block.operations]
        at = next(i for i, o in enumerate(ops) if o == op.operation)
        for o in ops[at + 1:]:
            if epi is not None and mxu.within(o, epi.nest):
                continue
            touches = [v for x in [o] + [y for y in walk(o) if y != o] for v in x.operands
                       if isinstance(v.type, ir.MemRefType) and mxu.buffer_root(v) == root]
            if not touches:
                continue
            if o.name != TransferOp.OPERATION_NAME or is_load(o):
                return None
            tx = o.opview
            if (mxu.buffer_root(tx.sram) != root or tx.index is not None or len(tx.dyn_strides)
                    or tx.dyn_offset_stride is not None or tx.dyn_fill is not None
                    or any(a in o.attributes for a in ("subtile_size", "cut", "indirect_dim"))):
                return None
            split = cls(o, lanes)
            if len(split.shape) != 2 or split.shape[0] % lanes or not set(split.masked) <= {0, 1}:
                return None
            if not _hoist_operands(o, op.operation, ops[at + 1:]):
                return None
            return split
        return None

    def emit(self, ip, tile, width):
        """The store of lane tiles tile .. tile + width - 1 (axis 0 of the buffer), at `ip`."""
        from mlir.dialects import arith
        n0, w = tile * self.lanes, width * self.lanes
        tx = self.store.opview
        clamp = list(tx.clamp)
        first = len(list(self.store.operands)) - len(clamp) - len(list(tx.dyn_strides)) \
            - (tx.dyn_fill is not None) - (tx.dyn_offset_stride is not None)
        with ip:
            # The operands first, then the store that takes them.
            dram_idx = arith.AddIOp(tx.dram_idx, const_index(n0 * self.dram_stride0)).result
            # Axis 0 rides the lanes: a lane tile further on is one row of axis 0 further in a bank.
            sram_idx = arith.AddIOp(tx.sram_idx, const_index(tile * self.stride0)).result
            if 0 in self.masked:
                k = self.masked.index(0)
                for j in (2 * k, 2 * k + 1):
                    v = arith.SubIOp(clamp[j], const_index(n0)).result
                    v = arith.MaxSIOp(v, const_index(0)).result
                    clamp[j] = arith.MinSIOp(v, const_index(w)).result
            new = self.store.clone(ip=ip)
            new = new.operation if hasattr(new, "operation") else new
            new.operands[1] = dram_idx
            new.operands[3] = sram_idx
            for j, v in enumerate(clamp):
                new.operands[first + j] = v
            i64 = ir.IntegerType.get_signless(64)
            new.attributes["subtile_size"] = ir.ArrayAttr.get(
                [ir.IntegerAttr.get(i64, w)] + [ir.IntegerAttr.get(i64, d) for d in self.shape[1:]])
        self.emitted.append((tile, width))

    def done(self, n_tiles):
        """Every lane tile the dot computes stored once, and the whole store gone."""
        covered = sorted(t + j for t, w in self.emitted for j in range(w))
        if covered != list(range(n_tiles)):
            raise RuntimeError(f"{NAME}: the N pairs' stores cover lane tiles {covered}, "
                               f"not 0..{n_tiles - 1}: {self.store}")
        self.store.erase()


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


def _build_dot(us, ip, n_parts, k_parts, inputs, lag, held=None):
    """The stream of the dots `us` (_groups): per (N pair, K pair) a panel, then the inputs of us[0],
    us[1], ..., popped `lag` positions behind; then the drain. `held` is the panel the dot before
    left staged (_follows), not staged again; the panel left staged is returned."""
    u = us[0]
    per = inputs * len(us)      # the inputs of one pair, over every dot
    pt = sum(part[1] for part in k_parts)
    last_k = pt - 1
    n_pairs = n_parts[0][1] if n_parts[0][2] else 0
    odd_n = None if n_parts[-1][2] else n_parts[-1]
    n_count = n_pairs + (1 if odd_n else 0)
    total = n_count * pt * per
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

    def short(v):
        return u.short(v % inputs)

    k_pairs = [(first + 2 * t, upper, rows) for first, pairs, upper, rows in k_parts
               for t in range(pairs)]

    def staged(i, q):
        """What K pair `q` of N pair `i` stages: its tiles, their rows and its format."""
        return (n_tile_of(i), i < n_pairs) + k_pairs[q] + (u.fmt, u.K)
    reuse = held is not None and held == staged(0, 0)

    def plan(i, q):
        """The inputs of K pair `q` of N pair `i` as segments (v_lo, v_hi, pop): before input v,
        nothing, or the rows of input `pop.row + v - v_lo` of the pair the pop is of. A dot's
        inputs, its pops, and a short last input are segments of their own."""
        segments = []
        for v in range(per):
            p = (i * pt + q) * per + v
            if p < lag:
                key, pop = (None, v // inputs, short(v)), None
            else:
                b = p - lag
                n_b, k_b = divmod(b // per, pt)
                key = (i - n_b, k_b, v // inputs, short(v), b % per // inputs, short(b % per))
                pop = _Pop(b % per, i - n_b, k_b == 0, k_b == last_k)
            if segments and segments[-1][2] == key:
                segments[-1][1] = v + 1
            else:
                segments.append([v, v + 1, key, pop])
        return tuple((lo, hi, pop) for lo, hi, _, pop in segments)

    def store(at, pair, g):
        """N pair `pair`'s columns of dot `g` stored at `at`, its last pops behind it, and its
        epilogue's rows closed."""
        if us[g].split is not None:
            us[g].split.emit(at, n_tile_of(pair), 2 if pair < n_pairs else 1)
        _close_rows(us[g], at, n_tile_of(pair), pair < n_pairs)

    def ends(pl):
        """Whether a plan's pops finish an N pair of a dot: its last K pair's last input popped."""
        return any(pop is not None and pop.last and pop.row % inputs + (hi - lo) - 1 == inputs - 1
                   for lo, hi, pop in pl)

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
        return (any(x.epi is not None for x in us)
                and any(pop is not None and pop.last for _, _, pop in plan(i, q)))

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

        def tile_at(dn):
            """The N tile a pop `dn` pairs back is of, None in the loop, and whether it has an
            upper one."""
            if n_index is not None:
                return n_tile_of(n_index - dn), n_index - dn < n_pairs
            return None, True

        def finished(at, dn, g):
            if n_index is None:
                raise RuntimeError(f"{NAME}: an N pair's store inside the loop over N pairs")
            store(at, n_index - dn, g)
        before = 0
        for k_first, part_pairs, k_upper, k_rows in k_parts:
            q = before
            end = before + part_pairs
            while q < end:
                pl, hi = plan(i, q), q + 1
                kept = reuse and n_index == 0 and q == 0    # its panel is in the unit already
                # A K pair whose pops finish an N pair is its own code: the store goes after it once.
                single = kept or (any(x.split is not None for x in us) and ends(pl))
                staged_before = staged_by_before(i, q, n_index)
                apart = epilogue_at(i, q)
                # A loop's trips share one body: a pair whose panel the one before staged is not
                # looped with one that stages its own.
                while (not single and not apart and q + 1 < end and hi + 1 < end
                       and plan(i, hi) == pl and staged_by_before(i, hi, n_index) == staged_before):
                    hi += 1
                if hi - q > 1 and any(pop is not None and pop.last
                                      and us[pop.row // inputs].epi is not None
                                      for _, _, pop in pl):
                    raise RuntimeError(f"{NAME}: an epilogue's pops in a loop over K pairs")
                k_ip, k_tile = _trip_loop(n_ip, k_first + 2 * (q - before), hi - q, 2)
                if apart:
                    ahead = None
                elif q + 1 < end:
                    with k_ip:
                        ahead = (n_tile, n_upper, arith.AddIOp(k_tile, const_index(2)).result,
                                 k_upper, k_rows)
                else:
                    ahead = ahead_of(q, n_tile, n_upper, n_index)
                _build_pair(us, k_ip, n_tile, n_upper, k_tile, k_upper, k_rows, pl, target,
                            tile_at, not kept and not staged_before, finished, ahead,
                            staged_before and not kept)
                q = hi
            before += part_pairs
    for i in range(n0):
        emit_n(ip, i, const_index_at(ip, n_tile_of(i)), True, i)
    if n0 < n_pairs:
        l_ip, n_tile = _trip_loop(ip, 2 * n0, n_pairs - n0, 2)
        emit_n(l_ip, n0, n_tile, True, None)
    if odd_n:
        emit_n(ip, n_pairs, const_index_at(ip, odd_n[0]), False, n_pairs)
    for b_lo, b_hi, (g_b, d, is_short) in _plan_drain(total, lag, per, inputs, short):
        n_b, k_b = divmod(g_b, pt)
        if k_b == last_k:
            _open_rows(us[d], ip, n_tile_of(n_b), n_b < n_pairs)
        d_ip, row, close = _pop_loop(ip, b_lo % per - d * inputs, b_hi - b_lo, us[d])
        with d_ip:
            _build_pops(us[d], const_index(n_tile_of(n_b)), row, n_b < n_pairs, k_b == 0,
                        k_b == last_k, u.last_rows if is_short else u.rows, n_tile_of(n_b))
        close()
        if k_b == last_k and b_hi % inputs == 0:
            store(ip, n_b, d)
    return staged(n_count - 1, last_k)


def _plan_drain(total, lag, per, inputs, short):
    """What the stream leaves to pop, as runs (b_lo, b_hi, (pair, dot, short)) over the stream's
    positions; a dot's pops, and a short last input's, are runs of their own."""
    out = []
    for b in range(total - lag, total):
        key = (b // per, b % per // inputs, short(b % per))
        if out and out[-1][2] == key:
            out[-1][1] = b + 1
        else:
            out.append([b, b + 1, key])
    return [tuple(x) for x in out]


def _build_pair(us, ip, n_tile, n_upper, k_tile, k_upper, k_rows, segments, target, tile_at,
                weights=True, finished=None, ahead=None, staged_rest=False):
    """A panel (unless `weights` is False, the unit holding it from the dot before or the pair
    before; with `staged_rest`, the pairs the pair before did not stage), then the pair's inputs
    segment by segment behind their pops; `target(back)` is a pop's N tile and whether it has an
    upper one, `tile_at(back)` the same as numbers. The first input latches the panel; a segment
    that ends a dot's N pair is followed by `finished(ip, back, dot)`. `ahead`, the stream's next
    panel (N tile, upper, K tile, upper, rows) or None, is staged under these inputs, as the v6e
    pushes the next tile between the vmatmuls of the one in use (_Staging)."""
    from mlir.dialects import arith
    u = us[0]
    inputs = u.inputs
    st = _Staging(u, ahead, inputs * len(us)) if ahead is not None else None

    def row_of(m, m_lo, p_row):
        """The row popped before input `m` (an int, or an index value) of a segment from `m_lo`."""
        if isinstance(m, int):
            return const_index(p_row + m - m_lo)
        if p_row == m_lo:
            return m
        if p_row > m_lo:
            return arith.AddIOp(m, const_index(p_row - m_lo)).result
        return arith.SubIOp(m, const_index(m_lo - p_row)).result

    if weights:
        with ip:
            _build_weights(u, n_tile, k_tile, n_upper, k_upper, k_rows)
    elif staged_rest:
        with ip:
            _Staging(u, (n_tile, n_upper, k_tile, k_upper, k_rows), inputs * len(us)).build_rest()
    for v_lo, v_hi, pop in segments:
        g_in, m_lo = v_lo // inputs, v_lo % inputs
        if pop is not None:
            g_pop, p_row = pop.row // inputs, pop.row % inputs
            if pop.last:
                _open_rows(us[g_pop], ip, *tile_at(pop.back))
        ends = (finished is not None and pop is not None and pop.last
                and p_row + (v_hi - v_lo) - 1 == inputs - 1)
        v = v_lo
        while v < v_hi:
            # Inputs [v, hi) are one loop body: the first input alone, then runs of inputs that
            # stage the same (_Staging.key), then the inputs past the staging.
            staging = st is not None and v < st.end
            hi = v + 1
            if v > 0:
                stop = min(v_hi, st.end) if staging else v_hi
                while hi < stop and (not staging or st.key(hi) == st.key(v)):
                    hi += 1
            m = v - g_in * inputs
            at, mv, close = _pop_loop(ip, m, hi - v, us[g_pop] if pop is not None else None)
            if hi - v == 1:
                mv = m
            if pop is not None:
                with at:
                    tile, upper = target(pop.back)
                    _build_pops(us[g_pop], tile, row_of(mv, m_lo, p_row), upper, pop.first,
                                pop.last, u.rows_of(p_row + m - m_lo), tile_at(pop.back)[0])
            with at:
                _build_input(us[g_in], k_tile, const_index(mv) if isinstance(mv, int) else mv,
                             k_upper, k_rows, v == 0, u.rows_of(m))
                if staging:
                    st.build(v, None if hi - v == 1 else (mv, m))
            close()
            v = hi
        if ends:
            finished(ip, pop.back, g_pop)


class _Staging:
    """How the panel `ahead` is staged under a pair's `inputs` inputs: its P prep/push pairs
    (lanes // rows a half of the depth, the first half then the second, deepest first, as
    _build_weights stages them), one after each input from the first (which latched) on. With
    fewer inputs than pairs the rest is pushed before the next pair's first input (build_rest):
    more after an input keeps their registers live across its pops and spills."""

    def __init__(self, u, ahead, inputs):
        self.u = u
        self.n_tile, self.n_upper, self.k_tile, self.k_upper, self.k_rows = ahead
        self.half = u.lanes // u.rows
        self.P = (2 if self.k_upper else 1) * self.half
        self.end = min(inputs, self.P)
        self.held = (u.lanes if self.k_upper else 0) + self.k_rows

    def _pair(self, j):
        """(the half's imm bits, the row, of its rows how many are B's) of pair j."""
        u = self.u
        h, k = divmod(j, self.half)
        row = h * u.lanes + u.lanes - u.rows - u.rows * k
        return SECOND_HALF if h else FIRST_HALF, row, max(0, min(u.rows, self.held - row))

    def key(self, m):
        """What input m stages but where: its pair's half and how many of its rows are B's."""
        half, _, live = self._pair(m)
        return half, live

    def _k_at(self):
        from mlir.dialects import arith
        k_tile = const_index(self.k_tile) if isinstance(self.k_tile, int) else self.k_tile
        return arith.MulIOp(k_tile, const_index(self.u.lanes // self.u.pack)).result

    def _n_tile(self):
        return const_index(self.n_tile) if isinstance(self.n_tile, int) else self.n_tile

    def build(self, m, at):
        """The pair after input m, or with `at` = (index value, its value at m) after each input of
        a loop over inputs that stage as m does."""
        from mlir.dialects import arith
        u = self.u
        half, row, live = self._pair(m)
        off = row // u.pack
        if at is not None:
            step = u.rows // u.pack
            off = arith.SubIOp(const_index(off + step * at[1]),
                               arith.MulIOp(at[0], const_index(step)).result).result
        _push_pair(u, self._n_tile(), self._k_at(), self.n_upper, half, off, live)

    def build_rest(self):
        """The pairs no input staged."""
        k_at, n_tile = self._k_at(), self._n_tile()
        for j in range(self.end, self.P):
            half, row, live = self._pair(j)
            _push_pair(self.u, n_tile, k_at, self.n_upper, half, row // self.u.pack, live)


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
    """Which lanes of a register group hold one of a short last K tile's `rows`. A lane's number is
    the same in each of its elements, and the unit writes it as 64 bits: it is made as wide as one
    instruction reaches at 64 bits, as 32 bits, and repeated across the group."""
    from mlir.dialects import arith, vector
    from .... import machine
    words = ir.VectorType(word_ty).shape[0]
    made = min(words, machine.VLEN_BITS * machine.ISA_MAX_LMUL // 64)
    i32 = ir.IntegerType.get_signless(32)
    lane = ir.Operation.create("torchsim.vlane_idx",
                               results=[ir.VectorType.get([made], ir.IndexType.get())],
                               operands=[], attributes={}).results[0]
    lane = arith.IndexCastOp(ir.VectorType.get([made], i32), lane).result
    if made < words:
        lane = ir.Operation.create(
            "vector.shuffle", results=[ir.VectorType.get([words], i32)], operands=[lane, lane],
            attributes={"mask": ir.DenseI64ArrayAttr.get([i % made for i in range(words)])}).results[0]
    return arith.CmpIOp(arith.CmpIPredicate.ult, lane,
                        vector.BroadcastOp(ir.VectorType.get([words], i32),
                                           arith.ConstantOp(i32, rows).result).result).result


def _open_rows(u, ip, n_tile, upper):
    """The rows of `u`'s epilogue at N tile `n_tile` (and the one above it with `upper`) opened
    at `ip`, those not open yet."""
    if u.epi is None:
        return
    if n_tile is None:
        raise RuntimeError(f"{NAME}: an epilogue's rows inside the loop over N pairs")
    for n in range(n_tile, n_tile + 1 + upper):
        if n not in u.open:
            with ip:
                u.open[n] = mxu.open_row(u.epi, n)


def _close_rows(u, ip, n_tile, upper):
    """The open rows of `u`'s epilogue at N tile `n_tile` (and the one above it with `upper`)
    closed at `ip`."""
    for n in range(n_tile, n_tile + 1 + upper):
        row = u.open.pop(n, None)
        if row is not None:
            with ip:
                mxu.close_row(u.epi, row)


def _pop_loop(ip, first, trips, u):
    """_trip_loop at step 1 for a run of `u`'s pops (u None: none), the loop carrying the running
    reductions of `u`'s open rows from trip to trip; the third value, called once the body is
    built, ends the loop with them and has the rows take its results."""
    from mlir.dialects import affine, arith
    rows = [] if u is None else [r for r in u.open.values() if r.acc is not None]
    if trips == 1 or not rows:
        inner, v = _trip_loop(ip, first, trips, 1)
        return inner, v, lambda: None
    with ip:
        lp = affine.AffineForOp(0, trips, 1, iter_args=[r.acc for r in rows])
    for r, a in zip(rows, lp.inner_iter_args):
        r.acc = a
    inner = ir.InsertionPoint(lp.body)
    with inner:
        v = lp.induction_variable
        v = v if first == 0 else arith.AddIOp(v, const_index(first)).result

    def close():
        with ir.InsertionPoint(lp.body):
            affine.AffineYieldOp([r.acc for r in rows])
        for r, res in zip(rows, lp.results):
            r.acc = res
    return inner, v, close


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


def _build_pops(u, n_tile, m_idx, n_upper, first, last, m_rows, n_at=None):
    """One vmatmul's rows, a register of f32 at a time, lower N tile's columns then upper's, added
    into C (written as is at the first K pair of a C proven zero), the epilogue at the last, on the
    rows of N tile `n_at` (and the one above); an odd N tile's upper half and rows past `m_rows`
    are popped and dropped."""
    from mlir.dialects import arith, vector
    pack = last and getattr(u, "dpack", None) is not None and m_rows == u.rows
    held = {}
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
            # The sum's last write to C is left out when nobody reads C but the epilogue, which takes it here.
            if not (last and not u.c_read_after):
                transfer_write(total, u.c, at, True)
            if pack:
                # The first row half's sum is kept until the second's; the two go to one pack, written as
                # words where the first half's elements sit.
                bits = bitcast(total, u.word)
                if not row:
                    held[upper] = bits
                else:
                    got = vcix("vcix.v.sv", [held[upper], bits], [u.word], {"opcode": a64(PACK)}).results[0]
                    transfer_write(got, u.dpack, _at(n_tile, upper, u.M // 2, m_idx, u.rows // 2, 0), True)
            elif last and u.epi is not None:
                if n_at is None or n_at + upper not in u.open:
                    raise RuntimeError(f"{NAME}: an epilogue's pops of a row not open")
                mxu.build_epilogue(u.epi, total, at, u.open[n_at + upper],
                                   lambda: arith.AddIOp(arith.MulIOp(m_idx, const_index(u.rows)).result,
                                                        const_index(row)).result)
