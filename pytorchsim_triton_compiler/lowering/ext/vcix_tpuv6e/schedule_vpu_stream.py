"""tritonpytorchsim-schedule-vpu-stream-tpuv6e: every innermost loop that pushes into the SFU's EUP and pops
it in the same trip (lower_vector_functions' exp), pipelined: UNROLL trips make a group, and a group pushes
the group DEPTH ahead before it pops its own, so the EUP's latency runs under other work instead of in front
of the pops. The push and what feeds it (stage 0) move ahead -- a prologue runs the first DEPTH groups' --
and the pops and what they feed (stage 1) stay, with an epilogue for the last DEPTH groups. Constants go to
either stage. The body is then ordered by width (_order_body), so a group's converts share one vtype and its
arithmetic another.

A loop that cannot be reordered so keeps its order and says why on EUP_PIPELINE: a trip count not a constant
multiple of UNROLL with DEPTH + 1 groups, a trip that does not pop what it pushes, a stage-0 value other
than a constant read in stage 1, a push that reads what the loop carries, a stage-0 write, or a stage-1
write to a buffer a stage-0 read may share."""
import sys

from .... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402
from mlir.dialects import affine  # noqa: E402

from ...passes.lib_ir import clone_ops, const_index, loop_bounds, producer, trip_count, walk  # noqa: E402
from ...passes.lower import SCHEDULE_VPU  # noqa: E402
from . import lib_mxu as mxu  # noqa: E402
from .lower_vector_functions import POP, PUSH  # noqa: E402

NAME = "tritonpytorchsim-schedule-vpu-stream-tpuv6e"

REQUIRES = ("vcix_tpuv6e",)

SLOT = SCHEDULE_VPU.name

MARKERS = ("llvm.call_intrinsic",)

FORBIDS = ()

EUP_PIPELINE = "torchsim.eup_pipeline"

# Trips a group covers, and groups a push runs ahead of its pops. A short loop's trip is shorter than a group's
# way through the EUP (a register a cycle in, then tpu_sfu_latency_cycles), so the pops wait unless the pushes
# run two trips ahead. Trips of two with their converts and their arithmetic each under one vtype measured
# slower (4584 cycles against 4310 for a 1024 x 1024 softmax block, gem5 with sequential vset fetch kept): the
# stores of a group then trail it.
UNROLL, DEPTH = 1, 2

# What the EUP holds, from the machine's vcix block: a register a cycle goes into a pipeline of LATENCY_KEY
# stages, and out into a result queue of QUEUE_KEY registers; when the queue is full the pipeline stops, and a
# push waits. A trip pushes before it pops, so a pop finds its own group and the DEPTH ahead of it inside.
QUEUE_KEY, LATENCY_KEY = "tpu_sfu_queue_registers", "tpu_sfu_latency_cycles"


def eup_registers(target):
    """The registers the EUP holds between a push and its pop, or None when the machine does not say."""
    if QUEUE_KEY not in target.vcix or LATENCY_KEY not in target.vcix:
        return None
    return int(target.vcix[QUEUE_KEY]) + int(target.vcix[LATENCY_KEY])

# What a loop keeps of its own when it is rebuilt: everything but its bounds and step.
_BOUNDS = ("lowerBoundMap", "upperBoundMap", "step", "operandSegmentSizes")

def run(module, **_):
    """Every innermost loop that pushes into the EUP, pipelined DEPTH groups deep, or marked with why not."""
    from .... import machine
    return schedule(module, eup_registers(machine.target()), machine.VLEN_BITS)


def schedule(module, holds, vlen):
    """run's work for an EUP that holds `holds` registers (None: not known) of `vlen` bits."""
    for loop in [o for o in walk(module.operation) if _pushes_innermost(o)]:
        why = _pipeline(loop, holds, vlen)
        if why:
            loop.attributes[EUP_PIPELINE] = ir.StringAttr.get(why)
    return module


def _intrinsic(op):
    return ir.StringAttr(op.attributes["intrin"]).value if op.name == "llvm.call_intrinsic" else None


def _pushes_innermost(op):
    if op.name != "affine.for":
        return False
    body = [o.operation for o in op.regions[0].blocks[0].operations]
    return (not any(o.name == "affine.for" for o in walk(op) if o is not op)
            and any(_intrinsic(o) == PUSH for o in body))


def _writes(op):
    return op.name in ("affine.vector_store", "affine.store", "vector.transfer_write", "memref.store",
                       "vector.store")


def _reads(op):
    return op.name in ("affine.vector_load", "affine.load", "vector.transfer_read", "memref.load",
                       "vector.load")


def _memref(op):
    return op.operands[1] if _writes(op) else op.operands[0]


def _fresh(root):
    if isinstance(root, tuple):
        return True
    p = producer(root)
    return p is not None and p.name in ("memref.alloc", "memref.alloca")


def _own_trip(write, read, iv, step):
    """Whether `write` and `read` each touch only their own trip's elements: one affine vector access of one
    memref through one map and the same operands, the loop's step apart in the last index and no wider than it.
    A read moved ahead to a later trip then never meets a write of an earlier one."""
    if write.name != "affine.vector_store" or read.name != "affine.vector_load" or \
            write.operands[1] != read.operands[0] or write.attributes["map"] != read.attributes["map"]:
        return False
    w_idx, r_idx = list(write.operands)[2:], list(read.operands)[1:]
    if len(w_idx) != len(r_idx) or any(a != b for a, b in zip(w_idx, r_idx)):
        return False
    amap = ir.AffineMapAttr(write.attributes["map"]).value
    dims = [i for i, v in enumerate(w_idx[:amap.n_dims]) if v == iv]
    if len(dims) != 1 or any(v == iv for v in w_idx[amap.n_dims:]):
        return False
    results = list(amap.results)
    if any(_iv_coefficient(e, dims[0]) != 0 for e in results[:-1]):
        return False
    coef = _iv_coefficient(results[-1], dims[0])
    width = ir.VectorType(read.results[0].type)
    return coef not in (None, 0) and width.rank == 1 and width.shape[0] <= abs(coef) * step


def _iv_coefficient(e, dim):
    """The coefficient of dimension `dim` in `e` when `e` is that times a constant plus terms free of it,
    else None."""
    if isinstance(e, ir.AffineDimExpr):
        return 1 if ir.AffineDimExpr(e).position == dim else 0
    if isinstance(e, (ir.AffineConstantExpr, ir.AffineSymbolExpr)):
        return 0
    if isinstance(e, ir.AffineAddExpr):
        a, b = _iv_coefficient(ir.AffineAddExpr(e).lhs, dim), _iv_coefficient(ir.AffineAddExpr(e).rhs, dim)
        return None if a is None or b is None else a + b
    if isinstance(e, ir.AffineMulExpr):
        m = ir.AffineMulExpr(e)
        for x, k in ((m.lhs, m.rhs), (m.rhs, m.lhs)):
            if isinstance(k, ir.AffineConstantExpr):
                c = _iv_coefficient(x, dim)
                return None if c is None else c * ir.AffineConstantExpr(k).value
    return 0 if not _involves(e, dim) else None


def _involves(e, dim):
    if isinstance(e, ir.AffineDimExpr):
        return ir.AffineDimExpr(e).position == dim
    if isinstance(e, ir.AffineBinaryExpr):
        b = ir.AffineBinaryExpr(e)
        return _involves(b.lhs, dim) or _involves(b.rhs, dim)
    return False


def _pipeline(loop, holds, vlen):
    """`loop` rebuilt DEPTH groups of UNROLL trips deep and its body ordered; None when done, else why not."""
    if holds is None:
        return f"the target states no {QUEUE_KEY} and {LATENCY_KEY}: what the EUP holds is not known"
    trips, bounds = trip_count(loop), loop_bounds(loop)
    if trips is None or trips % UNROLL or trips // UNROLL < DEPTH + 1:
        return (f"a trip count of {trips}: pipelining needs a constant multiple of {UNROLL} trips, "
                f"at least {(DEPTH + 1) * UNROLL}")
    first, _, step = bounds
    body = loop.regions[0].blocks[0]
    ops = [o.operation for o in body.operations]
    ops, yielded = ops[:-1], list(ops[-1].operands)
    pushes = [o for o in ops if _intrinsic(o) == PUSH]
    pops = [o for o in ops if _intrinsic(o) == POP]
    if len(pushes) != len(pops):
        return f"{len(pushes)} pushes and {len(pops)} pops a trip: a trip pops what it pushes"
    held = (DEPTH + 1) * UNROLL * sum(-(-ir.VectorType(p.operands[2].type).shape[0]
                                         * ir.VectorType(p.operands[2].type).element_type.width // vlen)
                                      for p in pushes)
    if held > holds:
        return f"the EUP would hold {held} registers at a pop, more than the {holds} it holds"
    iv, carried = body.arguments[0], list(body.arguments[1:])
    inside = set(ops)
    stage0, todo = set(), list(pushes)
    while todo:
        o = todo.pop()
        if o in stage0:
            continue
        stage0.add(o)
        for v in o.operands:
            p = producer(v)
            if p is not None and p in inside:
                todo.append(p)
            elif v in carried:
                return "a push reads what the loop carries"
    stage0_ops = [o for o in ops if o in stage0]
    stage1_ops = [o for o in ops if o not in stage0]
    consts = {o for o in stage0_ops if o.name == "arith.constant"}
    made0 = {r for o in stage0_ops if o not in consts for r in o.results}
    if any(v in made0 for o in stage1_ops for v in o.operands) or any(v in made0 for v in yielded):
        return "stage 1 reads a stage-0 value that is not a constant"
    if any(_writes(o) for o in stage0_ops):
        return "stage 0 writes memory"
    for wo in [o for o in stage1_ops if _writes(o)]:
        w = mxu.buffer_root(_memref(wo))
        for ro in [o for o in stage0_ops if _reads(o)]:
            r = mxu.buffer_root(_memref(ro))
            if not (_fresh(w) and _fresh(r) and w != r) and not _own_trip(wo, ro, iv, step):
                return "a stage-1 write may share a buffer with a stage-0 read"
    used1 = {v for o in stage1_ops for v in o.operands}
    shared = [o for o in stage0_ops if o in consts and any(r in used1 for r in o.results)]
    groups, span = trips // UNROLL, UNROLL * step
    end = first + (groups - DEPTH) * span
    inits = list(loop.operands)[len(list(loop.operands)) - len(carried):]

    def at(base, k):
        """The index `base` + k: an affine.apply of the loop's own induction variable, else a constant."""
        if isinstance(base, int):
            return const_index(base + k)
        if not k:
            return base
        return ir.Operation.create("affine.apply", results=[ir.IndexType.get()], operands=[base], attributes={
            "map": ir.AffineMapAttr.get(ir.AffineMap.get(1, 0, [ir.AffineAddExpr.get(
                ir.AffineDimExpr.get(0), ir.AffineConstantExpr.get(k))]))}).results[0]

    def stage0_group(base, ip):
        for u in range(UNROLL):
            clone_ops(stage0_ops, {iv: at(base, u * step)}, ip)

    def stage1_group(base, values, ip):
        for u in range(UNROLL):
            m = {iv: at(base, u * step), **dict(zip(carried, values))}
            clone_ops(shared + stage1_ops, m, ip)
            values = [m.get(v, v) for v in yielded]
        return values

    before = ir.InsertionPoint(loop)
    with before, loop.location:
        for g in range(DEPTH):
            stage0_group(first + g * span, before)
        new = affine.AffineForOp(first, end, span, iter_args=inits)
    for name in [loop.attributes[i].name for i in range(len(loop.attributes))
                 if loop.attributes[i].name not in _BOUNDS]:
        new.operation.attributes[name] = loop.attributes[name]
    nbody = new.operation.regions[0].blocks[0]
    niv, ncarried = nbody.arguments[0], list(nbody.arguments[1:])
    inner = ir.InsertionPoint(nbody)
    with inner, loop.location:
        stage0_group(at(niv, DEPTH * span), inner)
        got = stage1_group(niv, ncarried, inner)
        affine.AffineYieldOp(got)
    _order_body(nbody)
    values = list(new.results)
    with before, loop.location:
        for g in range(DEPTH):
            values = stage1_group(end + g * span, values, before)
    for r, f in zip(loop.results, values):
        r.replace_all_uses_with(f)
    loop.erase()
    return None


def _phase(op, stage1):
    """Where `op` goes in a pipelined body: scalars, loads, 16-bit vector ops, wider ones and the pushes of the
    group ahead; then the pops, and what they feed, of this group."""
    from .schedule_mxu_stream import _narrow, _scalar
    if _intrinsic(op) == PUSH:
        return 4
    if _intrinsic(op) == POP:
        return 5
    if op in stage1:
        return 6
    if _scalar(op):
        return 0
    if _reads(op):
        return 1
    return 2 if _narrow(op) else 3


def _order_body(body):
    """`body`'s ops in phase order, every operand before its user and the pushes, the pops and the
    writes each in the order they were."""
    from .schedule_mxu_stream import _topological
    ops = [o.operation for o in body.operations][:-1]
    stage1, todo = set(), [o for o in ops if _intrinsic(o) == POP]
    while todo:
        o = todo.pop()
        if o in stage1:
            continue
        stage1.add(o)
        todo += [u.owner for r in o.results for u in r.uses if u.owner in set(ops)]
    stage1 |= {o for o in ops if _writes(o)}
    index = {o: i for i, o in enumerate(ops)}
    preds = [set() for _ in ops]
    for i, o in enumerate(ops):
        for v in o.operands:
            p = producer(v)
            if p is not None and p in index:
                preds[i].add(index[p])
    for chain in ([o for o in ops if _intrinsic(o) == PUSH], [o for o in ops if _intrinsic(o) == POP],
                  [o for o in ops if _writes(o)]):
        for a, b in zip(chain, chain[1:]):
            preds[index[b]].add(index[a])
    succs = [[] for _ in ops]
    for i, ps in enumerate(preds):
        for p in ps:
            succs[p].append(i)
    key = [(_phase(o, stage1), i, i) for i, o in enumerate(ops)]
    order = _topological(len(ops), preds, succs, key)
    term = [o.operation for o in body.operations][-1]
    for i in order:
        ops[i].move_before(term)
