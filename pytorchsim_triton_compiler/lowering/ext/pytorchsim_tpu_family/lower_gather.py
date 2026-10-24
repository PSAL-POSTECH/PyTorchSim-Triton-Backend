"""lower_gather: every `pytorchsim.gather` along the lanes onto the cross-lane unit (tpu::Xlu) -- the
index rows set as its pattern, each source row pushed through ARBITRARY and popped into the
destination; a gather inside each lane's bank is the shared vrgather lowering (lib_gather_in_bank)."""

from ...passes.lib_bank_layout import bank_depth, bank_view
from ...passes.lib_gather_in_bank import lower_gather_in_bank
from ...passes.lib_ir import const_index, elem_bits, transfer_read, transfer_write, walk_ops, zero_of
from ...passes.lib_vcix import bitcast, i64 as a64, vcix, word_vector
from ...passes.lower import LOWER_GATHER
from .lib_xlu import RPU_ARBITRARY, RPU_BYPASS, simm5 as lib_simm5
from .lib_xlu import (SF_VC_IVV_SE, XLU_POP_OPCODE, XLU_PUSH_OPCODE, XLU_SET_PATTERN_OPCODE, as_signed_imm,
                      call_intrinsic, crossing_bits, crossing_words, crossings_of, element_from_words)
from .lib_xlu_move import QUEUE_KEY, refuse

import mlir.ir as ir  # noqa: E402

NAME = "pytorchsim-lower-gather"

SLOT = LOWER_GATHER.name

FORBIDS = LOWER_GATHER.lowers

MARKERS = FORBIDS


def run(module, *, target, lmul=None, **_):
    """Every gather lowered where it stands, a push or a pop a group of `lmul` registers at most."""
    from ....machine import ISA_MAX_LMUL
    queue = int(target.vcix[QUEUE_KEY]) if QUEUE_KEY in target.vcix else None
    lmul = ISA_MAX_LMUL if lmul is None else lmul
    for group in _gather_groups(module):
        _lower_gather(group, target.vector_lanes, target.vlen_bits, queue, lmul)
    return module


def _gathers_in(module):
    return [o.operation for region in module.operation.regions for b in region.blocks
            for o in walk_ops(b) if o.operation.name == "pytorchsim.gather"]


def _gather_groups(module):
    """The gathers, those along the lanes that follow one another reading one index buffer, none
    writing what another reads, as one group: one pattern serves them all, as the v6e's one
    vset.pattern serves each vperm after it."""
    groups = []
    for op in _gathers_in(module):
        last = groups[-1][-1] if groups else None
        if (last is not None and ir.IntegerAttr(op.attributes["axis"]).value == 0
                and ir.IntegerAttr(last.attributes["axis"]).value == 0
                and op.operands[1] == last.operands[1] and _follows(op, last)
                and all(op.operands[0] != g.operands[2] and op.operands[2] not in (g.operands[0], g.operands[1])
                        for g in groups[-1])):
            groups[-1].append(op)
        else:
            groups.append([op])
    return groups


def _follows(op, before):
    """Whether `op` comes right after `before` in one block, nothing between them."""
    if before.parent != op.parent:
        return False
    prev = None
    for o in _block_ops(op):
        if o == op:
            return prev == before
        prev = o
    return False


def _block_ops(op):
    """The ops of the block that holds `op`, in order."""
    for region in op.parent.regions:
        for b in region.blocks:
            ops = [o.operation for o in b.operations]
            if any(o == op for o in ops):
                return ops
    return []


def _consumer_rows(group, depth):
    """The loop right after `group` that reads a destination and walks the banks a register group of
    rows a trip, in order from the first, touching each destination only at its own rows: (that loop,
    the loads of each destination, whether the loop stores each one, the rows a trip), or None."""
    from ...passes.lib_ir import loop_bounds, trip_count, walk
    ops = _block_ops(group[-1])
    at = [k for k, o in enumerate(ops) if o == group[-1].operation]
    loop, outer = (ops[at[0] + 1] if at and at[0] + 1 < len(ops) else None), []
    while loop is not None and loop.name == "affine.for":
        body = list(loop.regions[0].blocks[0].operations)[:-1]
        if len(body) != 1 or body[0].name != "affine.for":
            break
        outer.append(loop)
        loop = body[0].operation
    if loop is None or loop.name != "affine.for":
        return None
    if any(trip_count(o) != 1 or loop_bounds(o)[0] != 0 for o in outer):
        return None
    first, end, step = loop_bounds(loop)
    if first != 0 or end != depth or not step or depth % step:
        return None
    ivs = [o.regions[0].blocks[0].arguments[0] for o in outer + [loop]]
    body = loop.regions[0].blocks[0]
    dsts = [g.operands[2] for g in group]
    loads, stored = {k: [] for k in range(len(dsts))}, set()
    for o in walk(loop):
        for k, d in enumerate(dsts):
            if not any(v == d for v in o.operands):
                continue
            mem = {"affine.vector_load": 0, "affine.vector_store": 1}.get(o.name)
            if mem is None or o.operands[mem] != d or o.parent.operation != loop:
                return None
            m = ir.AffineMapAttr(o.attributes["map"]).value
            vec = ir.VectorType(o.results[0].type if mem == 0 else o.operands[0].type)
            if (m != ir.AffineMap.get_identity(len(ivs)) or vec.shape != [step]
                    or list(o.operands)[mem + 1:] != ivs):
                return None
            (loads[k].append(o) if mem == 0 else stored.add(k))
    if not any(loads.values()):
        return None
    return loop, loads, stored, step, ivs


def _lower_gather(group, lanes, vlen, queue=None, lmul=1):
    """Gathers on buffers that share one index (_gather_groups). Along the lane axis (axis 0 of the
    banks): each register group of rows, the index rows set once as the pattern (xlu_set_pattern),
    every gather's source rows pushed with the shuffle before ARBITRARY, every push before the first
    pop, each pop into its destination's rows -- lane l of row r takes lane index[l, r] of source row
    r, as the v6e's vset.pattern + a vperm per register. When the next loop reads the destinations a
    register group of rows a trip (_consumer_rows), the pushes go a trip's rows at a time before it
    and each trip pops its own rows, as the v6e pops a vperm where its value is used. An axis longer
    than the lanes is refused: an index along it may name an element of another row."""
    from mlir.dialects import arith
    op = group[0]
    index = op.operands[1]
    axis = ir.IntegerAttr(op.attributes["axis"]).value
    if axis != 0:
        return lower_gather_in_bank(op, lanes, vlen, lmul)
    for g in group:
        bits = elem_bits(ir.MemRefType(g.operands[0].type).element_type)
        if not crossings_of(bits)[0]:
            return refuse("gather", g, crossings_of(bits)[1])
        if ir.MemRefType(g.operands[0].type).shape[0] > lanes:
            return refuse("gather", g, f"its axis is longer than the {lanes} lanes, so an index may "
                                    "name another register's element")
    depth = bank_depth(index, lanes)
    if depth is None or any(bank_depth(v, lanes) != depth for g in group for v in (g.operands[0], g.operands[2])):
        return refuse("gather", op, "its sources, index and destinations are not one bank shape")
    i64, i32 = ir.IntegerType.get_signless(64), ir.IntegerType.get_signless(32)
    ielt = ir.MemRefType(index.type).element_type
    simm5 = as_signed_imm(lib_simm5(RPU_ARBITRARY, 0, RPU_BYPASS))
    w = min([depth, lmul * vlen // 64] + [lmul * vlen // crossing_bits(ir.MemRefType(g.operands[0].type).element_type)
                                          for g in group])
    consumer = _consumer_rows(group, depth) if queue is not None else None
    if consumer is not None:
        w = consumer[3]
    groups = [(at, min(w, depth - at)) for at in range(0, depth, w)]
    with ir.InsertionPoint(op):
        idx_bank = bank_view(index, lanes)
        members = [(bank_view(g.operands[0], lanes), bank_view(g.operands[2], lanes),
                    ir.MemRefType(g.operands[0].type).element_type) for g in group]
        const = lambda t, v: arith.ConstantOp(t, v).result
        set_family = const(i64, a64(XLU_SET_PATTERN_OPCODE))

        def push(at, n):
            ity = ir.VectorType.get([n], ielt)
            lanes_of = transfer_read(ity, idx_bank, [const_index(0), const_index(at)], zero_of(ielt), True)
            if ielt != i32:
                lanes_of = (arith.TruncIOp if elem_bits(ielt) > 32 else arith.ExtSIOp)(
                    ir.VectorType.get([n], i32), lanes_of).result
            rvl = const(i64, a64(n))
            call_intrinsic(SF_VC_IVV_SE, [set_family, lanes_of, lanes_of, const(i64, a64(0)), rvl])
            for src_bank, _dst_bank, elt in members:
                bits, wide = elem_bits(elt), ir.VectorType.get([n], elt)
                v = transfer_read(wide, src_bank, [const_index(0), const_index(at)], zero_of(elt), True)
                for h in range(crossings_of(bits)[0]):
                    vcix("vcix.iv", [crossing_words(bitcast(v, word_vector(wide)) if v.type != word_vector(wide)
                                                    else v, bits)[h], rvl], [],
                         {"opcode": a64(XLU_PUSH_OPCODE), "imm": a64(simm5), "rd": a64(0)})

        def pop(at, n):
            rvl = const(i64, a64(n))
            for _src_bank, dst_bank, elt in members:
                bits, wide = elem_bits(elt), ir.VectorType.get([n], elt)
                wide_w = word_vector(wide)
                for h in range(crossings_of(bits)[0]):
                    got = vcix("vcix.v.i", [rvl], [ir.VectorType.get([n], i32)],
                               {"opcode": a64(XLU_POP_OPCODE), "imm": a64(simm5), "rs2": a64(0)}).results[0]
                    there = None
                    if h:
                        there = transfer_read(wide, dst_bank, [const_index(0), const_index(at)], zero_of(elt), True)
                        there = bitcast(there, wide_w) if there.type != wide_w else there
                    got = element_from_words(got, h, bits, wide_w, there)
                    transfer_write(bitcast(got, wide) if got.type != wide else got, dst_bank,
                                   [const_index(0), const_index(at)], True)
        if queue is None:
            for at, n in groups:
                push(at, n)
                pop(at, n)
        else:
            for at, n in groups:
                push(at, n)
            if consumer is None:
                for at, n in groups:
                    pop(at, n)
    if consumer is not None:
        _pop_in_consumer(group, consumer, simm5)
    for g in group:
        g.erase()
    return True


def _pop_in_consumer(group, consumer, simm5):
    """Each trip of the consumer loop pops its rows of every destination, in the order they were
    pushed, and reads them from the pop; they reach the destination too unless the loop stores
    its own over them."""
    from mlir.dialects import affine, arith
    loop, loads, stored, n, ivs = consumer
    i32, i64 = ir.IntegerType.get_signless(32), ir.IntegerType.get_signless(64)
    body = loop.regions[0].blocks[0]
    popped = []
    with ir.InsertionPoint(list(body.operations)[0]), loop.location:
        rvl = arith.ConstantOp(i64, a64(n)).result
        for g in group:
            elt = ir.MemRefType(g.operands[0].type).element_type
            bits, wide = elem_bits(elt), ir.VectorType.get([n], elt)
            wide_w = word_vector(wide)
            got = None
            for h in range(crossings_of(bits)[0]):
                word = vcix("vcix.v.i", [rvl], [ir.VectorType.get([n], i32)],
                            {"opcode": a64(XLU_POP_OPCODE), "imm": a64(simm5), "rs2": a64(0)}).results[0]
                got = element_from_words(word, h, bits, wide_w, got)
            popped.append(bitcast(got, wide) if got.type != wide else got)
        for k, (g, got) in enumerate(zip(group, popped)):
            if k not in stored:
                affine.AffineVectorStoreOp(got, g.operands[2], ivs, ir.AffineMap.get_identity(len(ivs)))
    for k, got in enumerate(popped):
        for o in loads[k]:
            o.results[0].replace_all_uses_with(got)
            o.erase()
