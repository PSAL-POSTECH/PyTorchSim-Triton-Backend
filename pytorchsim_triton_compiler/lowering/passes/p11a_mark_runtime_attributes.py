"""tritonpytorchsim-mark-runtime-attributes: every TOG-side quantity whose backward slice reaches a
memory read is given a torchsim.runtime_attribute -- S at a read outside every loop, A before the loop
nest with R at a read inside one -- or the kernel is refused, by name, where no record can state it."""
from collections import namedtuple

from ...contract.kernel_object import HIDDEN_BYTES_MAX, RUNTIME_ATTRIBUTE_HEAD, grid3
from ...contract.stop import Reason, Refused
from ...dialects.torchsim import (RuntimeAttributeOp, TransferOp, WaitOp, branch_controls, effects_on,
                                  is_load, loop_like, moved_shape, view_root)
from .lib_ir import const_int, funcs, is_memref, walk

NAME = "tritonpytorchsim-mark-runtime-attributes"
MARKERS = ("func.func",)

RECORD = RuntimeAttributeOp.OPERATION_NAME
PLUMBING = "torchsim.runtime_attribute_plumbing"
"""Stamped on every op this pass makes besides the records: what feeds a record alone, which the TOG
does not segment and the ELF and the sample erase with the records."""

MAX_ATTRIBUTES, MAX_RANK = 64, 8

_CASTS = ("arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui", "arith.trunci")

SOURCES = "torchsim.runtime_attribute_sources"
"""Stamped on every record: the function arguments its value is read from, by argument number -- what a
run must hold as the launch did for its records to be the launch's."""

UNRECORDED = "torchsim.runtime_attribute_unrecorded"
"""Stamped on a gather or a read whose record has no static capacity, with the refusal: the ELF runs
without it, and the TOG, which would read it, is refused there."""

Read = namedtuple("Read", "op value loops")
Sink = namedtuple("Sink", "what op value kind replayed")
Dim = namedtuple("Dim", "loop stride lb ub step")


def refuse(message):
    """The kernel stops at lower for a value no runtime attribute can state."""
    return Refused(Reason.RUNTIME_ATTRIBUTE, "runtime attribute: " + message)


class NoCapacity(Refused):
    """A record no static capacity bounds: what the TOG cannot read, while the ELF needs no record."""

    def __init__(self, message):
        super().__init__(Reason.RUNTIME_ATTRIBUTE, "runtime attribute: " + message)


def refuse_unrecorded(module):
    """The refusal of the first gather or read of `module` left unrecorded for want of a capacity."""
    import mlir.ir as ir
    for o in walk(module.operation):
        if UNRECORDED in o.attributes:
            raise refuse(ir.StringAttr(o.attributes[UNRECORDED]).value.removeprefix("runtime attribute: "))


def run(module, *, grid=(1,), **_):
    """Mark every function of `module`: one record per read point, ids 0, 1, ... in program order;
    refused when the launch's rows of them would outgrow the hidden output."""
    import math
    for f in funcs(module):
        _Marker(f.operation).mark()
    words = RUNTIME_ATTRIBUTE_HEAD
    for o in records(module):
        form = o.opview.form.value
        if form in ("S", "A", "G"):
            rank = o.opview.rank.value
            cap = list(o.opview.capacity) if o.opview.capacity is not None else []
            words += 2 + rank + math.prod(cap)
    programs = math.prod(grid3(grid))
    if words > RUNTIME_ATTRIBUTE_HEAD and programs * words * 8 > HIDDEN_BYTES_MAX:
        raise refuse(f"the hidden output would take {programs * words * 8} bytes ({programs} programs "
                     f"x {words} words)")


def records(module):
    """Every runtime_attribute op of `module`, in program order."""
    return [o for o in walk(module.operation) if o.name == RECORD]


def erase_records(module):
    """The module as if this pass had not run: every record erased, S and R by their value, then
    every op it made for them and every stamp it left."""
    for o in walk(module.operation):
        if UNRECORDED in o.attributes:
            del o.attributes[UNRECORDED]
    for o in records(module):
        if len(o.results):
            o.results[0].replace_all_uses_with(o.operands[0])
        o.erase()
    for o in reversed([o for o in walk(module.operation) if PLUMBING in o.attributes]):
        o.erase()


def _same(a, b):
    return getattr(a, "operation", a) == getattr(b, "operation", b)


def _in(x, xs):
    return any(_same(x, y) for y in xs)


def _parent_op(op):
    p = op.parent
    return None if p is None else p.operation


def _holds_tog_op(op):
    """Whether a transfer or a wait sits in `op`'s regions: what makes a loop or a branch the TOG's."""
    return any(o.name in (TransferOp.OPERATION_NAME, WaitOp.OPERATION_NAME) for o in walk(op))


def _short(op):
    return " ".join(str(getattr(op, "operation", op)).split("\n")[0].split())[:120]


class _Marker:
    """The analysis of one function: its sinks, the reads they rest on, and the records they take."""

    def __init__(self, func):
        import mlir.ir as ir
        self.ir = ir
        self.func = func
        self.ops = walk(func)
        self.values = []

    def at(self, op):
        """`op`'s place in program order: the key it is known by here."""
        return next(k for k, o in enumerate(self.ops) if _same(o, op))

    def vkey(self, v):
        """The key value `v` is known by here."""
        for k, x in enumerate(self.values):
            if x == v:
                return k
        self.values.append(v)
        return len(self.values) - 1

    def loops_of(self, op):
        """The loops enclosing `op` inside the function, outermost first."""
        out, p = [], _parent_op(op)
        while p is not None and not _same(p, self.func):
            if loop_like(p) is not None:
                out.append(p)
            p = _parent_op(p)
        return out[::-1]

    def sinks(self):
        """Every value the TOG states or runs by: a transfer's address, clamps and offset stride, a
        loop's bounds and step, a branch's controls; `replayed` when its op holds a transfer or a wait."""
        out = []
        for o in self.ops:
            if o.name == TransferOp.OPERATION_NAME:
                tx = o.opview
                vals = [("the address", tx.dram_idx)] + [("a clamp", c) for c in tx.clamp]
                if tx.dyn_offset_stride is not None:
                    vals.append(("the offset stride", tx.dyn_offset_stride))
                out += [Sink(f"{what} of {_short(o)}", o, v, "transfer", True) for what, v in vals]
                continue
            loop = loop_like(o)
            if loop is not None:
                for kind, bounds in (("bound", loop.lbs + loop.ubs), ("step", loop.steps)):
                    out += [Sink(f"the {kind} of {_short(o)}", o, b, kind, _holds_tog_op(o))
                            for b in bounds if not isinstance(b, int)]
                continue
            out += [Sink(f"the condition of {_short(o)}", o, c, "branch", _holds_tog_op(o))
                    for c in branch_controls(o) or ()]
        return out

    @staticmethod
    def is_read(op):
        """Whether `op` reads memory: a read (or effects it does not state) on a memref operand."""
        return any(is_memref(v) and effects_on(op, v) & {"read", "unknown"} for v in op.operands)

    def reads_of(self, value, seen=None):
        """The read points `value` rests on, walking its operands back: through a loop's iter_args to
        their inits and yields, ending at a loop iv, a function argument or a read; any other block
        argument is refused."""
        seen = [] if seen is None else seen
        if any(x == value for x in seen):
            return []
        seen.append(value)
        if isinstance(value, self.ir.BlockArgument):
            owner = value.owner.owner
            if owner is None or _same(owner, self.func):
                return []
            loop = loop_like(owner)
            for arg, init, yielded in (loop.iter_args if loop is not None else ()):
                if arg == value:
                    return self.reads_of(init, seen) + (
                        self.reads_of(yielded, seen) if yielded is not None else [])
            if loop is not None and any(iv == value for iv in loop.ivs):
                return []
            raise refuse(f"a TOG-side value rests on a block argument of {_short(owner)} that is no "
                         f"loop iv or iter_arg")
        op = value.owner.operation
        if self.is_read(op):
            return [op]
        loop = loop_like(op)
        if loop is not None:
            _, init, yielded = loop.iter_args[list(op.results).index(value)]
            return self.reads_of(init, seen) + (self.reads_of(yielded, seen) if yielded is not None else [])
        out = []
        for v in op.operands:
            if not is_memref(v):
                out += self.reads_of(v, seen)
        k = list(op.results).index(value)
        for region in op.regions:
            for block in region.blocks:
                term = list(block.operations)[-1].operation
                if k < len(term.operands):
                    out += self.reads_of(term.operands[k], seen)
        return out

    def mark(self):
        """Refuse what no record states, then record every read point a sink rests on, each record
        stamped with the SOURCES its value is read from."""
        reached = []
        for s in self.sinks():
            got = self.reads_of(s.value)
            if not got:
                continue
            if not s.replayed:
                raise refuse(f"the control of compute segment {_short(s.op)} rests on {_short(got[0])}")
            if s.kind == "step":
                raise refuse(f"the step of {_short(s.op)} rests on {_short(got[0])}")
            reached += [r for r in got if not _in(r, reached)]
        if not reached:
            return
        if len(reached) > MAX_ATTRIBUTES:
            raise refuse(f"{len(reached)} reads reach the TOG; a TOG declares at most {MAX_ATTRIBUTES}")
        reads = [Read(r, self.result_of(r), self.loops_of(r)) for r in sorted(reached, key=self.at)]
        plans = [(read, self.planned(read.op, lambda r=read: self.array_plan(r)) if read.loops else None)
                 for read in reads]
        plans = [p for p in plans if p[1] is not False]
        for i, (read, plan) in enumerate(plans):
            seen = []
            src = self.read_sources(read.op, seen)
            if plan is None:
                made = self.record_scalar(read, i)
            else:
                made = self.record_array(read, plan, i)
            for rec in made:
                rec.attributes[SOURCES] = self.ir.DenseI64ArrayAttr.get(sorted(src))

    def value_sources(self, value, seen):
        """The function arguments `value` is read from: those of every read point it rests on."""
        out = set()
        for r in self.reads_of(value):
            out |= self.read_sources(r, seen)
        return out

    def read_sources(self, op, seen):
        """The function arguments what `op` reads comes from: its buffers' and its other operands'."""
        out = set()
        for v in op.operands:
            out |= self.buffer_sources(v, seen) if is_memref(v) else self.value_sources(v, seen)
        return out

    def buffer_sources(self, mem, seen):
        """The function arguments the contents of `mem` come from: the argument its view rests on, else
        what every op writing that buffer reads (a load its DRAM view and address)."""
        ir = self.ir
        root = view_root(mem)
        if _in(root, seen):
            return set()
        seen.append(root)
        if isinstance(root, ir.BlockArgument) and _same(root.owner.owner, self.func):
            return {root.arg_number}
        out = set()
        for o in self.ops:
            if any(is_memref(v) and _same(view_root(v), root) and "write" in effects_on(o, v)
                   for v in o.operands):
                out |= self.read_sources(o, seen)
        return out

    def planned(self, op, plan):
        """`plan()`, or False with `op` stamped UNRECORDED when no static capacity bounds its record."""
        try:
            return plan()
        except NoCapacity as e:
            op.attributes[UNRECORDED] = self.ir.StringAttr.get(str(e))
            return False

    def array_plan(self, read):
        """array_of(read) and the capacity of each of its dims."""
        source, outer, const, dims = self.array_of(read)
        return source, outer, const, dims, self.capacity(source, dims, read)

    def result_of(self, op):
        """The scalar integer `op` reads, or a refusal naming why no record holds it."""
        ir = self.ir
        if len(op.results) != 1:
            raise refuse(f"{_short(op)} reaches the TOG and reads no single value")
        v = op.results[0]
        if isinstance(v.type, (ir.IntegerType, ir.IndexType)):
            return v
        if isinstance(v.type, ir.FloatType):
            raise refuse(f"a TOG-side value rests on the float value {_short(op)}")
        raise refuse(f"a TOG-side value rests on the vector value {_short(op)}, which no record can hold")

    def attrs(self, i, rank, form, segments, **arrays):
        ir = self.ir
        i32 = ir.IntegerType.get_signless(32)
        out = {"id": ir.IntegerAttr.get(i32, i), "rank": ir.IntegerAttr.get(i32, rank),
               "form": ir.StringAttr.get(form), "operandSegmentSizes": ir.DenseI32ArrayAttr.get(segments)}
        out.update({k: ir.DenseI64ArrayAttr.get(list(v)) for k, v in arrays.items()})
        return out

    def yield_record(self, read, rec):
        """Every use of the read value but `rec`'s own takes what `rec` yields."""
        for use in list(read.value.uses):
            owner = use.owner.operation if hasattr(use.owner, "operation") else use.owner
            if not _same(owner, rec):
                owner.operands[use.operand_number] = rec.results[0]

    def record_scalar(self, read, i):
        ir = self.ir
        with ir.InsertionPoint(self.next_op(read.op)), read.op.location:
            rec = ir.Operation.create(RECORD, results=[read.value.type], operands=[read.value],
                                      attributes=self.attrs(i, 0, "S", [1, 0, 0, 0, 0]))
        self.yield_record(read, rec)
        return [rec]

    @staticmethod
    def next_op(op):
        """The op after `op` in its block: its block's terminator follows every non-terminator."""
        ops = [o.operation for o in op.block.operations]
        return ops[next(k for k, o in enumerate(ops) if _same(o, op)) + 1]

    def array_of(self, read):
        """(source, outer terms, constant, dims) of the DRAM elements an in-loop read takes over every
        trip of the loops it moves with, refused where the address is no affine function of their trips
        and of values defined above the nest."""
        source, address = self.dram_address(read)
        root = view_root(source)
        if any(is_memref(v) and _same(view_root(v), root) and "write" in effects_on(o, v)
               for o in self.ops for v in o.operands):
            raise refuse(f"{_short(read.op)} reads {_short(source.owner)}, which the kernel also writes")
        terms, const = self.affine(address, read)
        dims = []
        for loop in read.loops:
            stride = terms.pop(("trip", self.at(loop)), 0)
            if stride:
                lb, ub, step = self.bounds_of(loop, read)
                dims.append(Dim(loop, stride, lb, ub, step))
        if len(dims) > MAX_RANK:
            raise refuse(f"the record of {_short(read.op)} would have rank {len(dims)}, above {MAX_RANK}")
        return source, [(self.values[k], c) for (_, k), c in terms.items()], const, dims

    def dram_address(self, read):
        """(the DRAM memref, the element index into it) of what `read` reads: its own index when it
        reads DRAM, else the address the one load transfer filling its scratchpad element in the same
        trip reads; a tile anything else writes is a value the kernel computed."""
        op = read.op
        mem = next(v for v in op.operands if is_memref(v))
        root = view_root(mem)
        if self.ir.MemRefType(root.type).memory_space is None:
            idx = [v for v in op.operands if not is_memref(v)]
            if self.ir.MemRefType(mem.type).rank != 1 or len(idx) != 1:
                raise refuse(f"{_short(op)} reads DRAM through a memref of other than one dim")
            return mem, idx[0]
        writers = []
        for o in self.ops:
            if any(is_memref(v) and _same(view_root(v), root) and "write" in effects_on(o, v)
                   for v in o.operands) and not _in(o, writers):
                writers.append(o)
        if len(writers) != 1 or writers[0].name != TransferOp.OPERATION_NAME or not is_load(writers[0]):
            raise refuse(f"a TOG-side value rests on {_short(op)}, a value the kernel computed inside "
                         f"{_short(read.loops[-1])}")
        w = writers[0]
        if [self.at(x) for x in self.loops_of(w)] != [self.at(x) for x in read.loops]:
            raise refuse(f"{_short(op)} reads a tile {_short(w)} fills in another trip")
        tx = w.opview
        if (tx.index is not None or tx.masked_axes is not None or "piece_box" in w.attributes
                or "cut" in w.attributes):
            raise refuse(f"{_short(op)} reads a tile {_short(w)} fills by a gather, a mask or in pieces")
        n = 1
        for d in moved_shape(w):
            n *= d
        if n != 1 or self.ir.MemRefType(tx.dram.type).rank != 1:
            raise refuse(f"{_short(op)} reads one element of a {n}-element tile {_short(w)}")
        return tx.dram, tx.dram_idx

    def bounds_of(self, loop, read):
        """(lb, ub, step) of a loop an in-loop read moves with: bounds defined above the loop nest, a
        step that is a positive constant."""
        got = loop_like(loop)
        lb, ub, step = got.lbs[0], got.ubs[0], got.steps[0]
        step = step if isinstance(step, int) else const_int(step)
        if step is None or step <= 0:
            raise refuse(f"the step of {_short(loop)} rests on a value that is no positive constant")
        for b in (lb, ub):
            if not isinstance(b, int) and not self.above(b, read.loops[0]):
                raise refuse(f"the base, or trip count of {_short(loop)}, of {_short(read.op)} is "
                             f"defined inside the loop nest")
        return lb, ub, step

    def above(self, v, loop):
        """Whether `v` is defined outside `loop`."""
        o = v.owner.owner if isinstance(v, self.ir.BlockArgument) else v.owner
        o = None if o is None else o.operation
        while o is not None:
            if _same(o, loop):
                return False
            o = _parent_op(o)
        return True

    def affine(self, v, read, stack=()):
        """`v` as ({term: coefficient}, constant): a term a loop's trip ("trip", at(loop)) or a value
        defined above the nest ("value", vkey(v)). Refused when it rests on a read, a non-affine op or
        an iter_arg that is no induction."""
        c = const_int(v)
        if c is not None:
            return {}, c
        if self.above(v, read.loops[0]):
            return {("value", self.vkey(v)): 1}, 0
        if isinstance(v, self.ir.BlockArgument):
            owner = v.owner.owner.operation
            loop = loop_like(owner)
            if loop is not None and any(iv == v for iv in loop.ivs):
                lb, _, step = self.bounds_of(owner, read)
                terms, const = ({}, lb) if isinstance(lb, int) else ({("value", self.vkey(lb)): 1}, 0)
                terms[("trip", self.at(owner))] = step
                return terms, const
            if loop is not None:
                return self.induction(v, owner, loop, read, stack)
            raise refuse(f"the address of {_short(read.op)} rests on a block argument of {_short(owner)}")
        op = v.owner.operation
        if self.is_read(op):
            raise refuse(f"the address of {_short(read.op)} rests on {_short(op)}, read earlier in the "
                         f"same trip (two-level indirection)")
        if _in(op, stack):
            raise refuse(f"the address of {_short(read.op)} is not affine in the trips of its loops")
        ops = list(op.operands)
        stack = stack + (op,)
        if op.name in _CASTS:
            return self.affine(ops[0], read, stack)
        if op.name in ("arith.addi", "arith.subi"):
            (ta, ca), (tb, cb) = self.affine(ops[0], read, stack), self.affine(ops[1], read, stack)
            sign = 1 if op.name == "arith.addi" else -1
            out = dict(ta)
            for t, k in tb.items():
                out[t] = out.get(t, 0) + sign * k
            return {t: k for t, k in out.items() if k}, ca + sign * cb
        if op.name == "arith.muli":
            (ta, ca), (tb, cb) = self.affine(ops[0], read, stack), self.affine(ops[1], read, stack)
            if ta and tb:
                raise refuse(f"the address of {_short(read.op)} moves with a trip by a non-constant "
                             f"{_short(op)}")
            (t, c), k = ((ta, ca), cb) if not tb else ((tb, cb), ca)
            return {x: y * k for x, y in t.items() if y * k}, c * k
        if self.rests_on_a_read(v):
            raise refuse(f"the address of {_short(read.op)} rests on {_short(op)}, read earlier in the "
                         f"same trip (two-level indirection)")
        raise refuse(f"the address of {_short(read.op)} is not affine in the trips of its loops: "
                     f"{_short(op)}")

    def rests_on_a_read(self, v):
        try:
            return bool(self.reads_of(v))
        except Refused:
            return True

    def induction(self, v, owner, loop, read, stack):
        """An iter_arg as its init plus a constant per trip when its yield is itself plus one; refused
        as a chase or an accumulation when its yield rests on a read."""
        _, init, yielded = next(t for t in loop.iter_args if t[0] == v)
        step = self.self_step(yielded, v) if yielded is not None else None
        if step is not None:
            terms, const = self.affine(init, read, stack)
            terms = dict(terms)
            key = ("trip", self.at(owner))
            terms[key] = terms.get(key, 0) + step
            return {t: k for t, k in terms.items() if k}, const
        got = self.reads_of(yielded) if yielded is not None else []
        if got and self.passes_through(yielded, got[0]):
            raise refuse(f"the address of {_short(read.op)} rests on {_short(got[0])}, which an earlier "
                         f"trip of {_short(owner)} read (pointer chasing)")
        if got:
            raise refuse(f"the address of {_short(read.op)} rests on an iter_arg, which accumulates data "
                         f"read in {_short(owner)}")
        raise refuse(f"the address of {_short(read.op)} rests on an iter_arg of {_short(owner)} that is "
                     f"no induction")

    def self_step(self, yielded, arg):
        """The constant `yielded` adds to `arg` each trip, or None when it is not `arg` plus one."""
        v, step = yielded, 0
        while v != arg:
            if isinstance(v, self.ir.BlockArgument):
                return None
            op = v.owner.operation
            ops = list(op.operands)
            if op.name in _CASTS:
                v = ops[0]
            elif op.name in ("arith.addi", "arith.subi") and const_int(ops[1]) is not None:
                step += const_int(ops[1]) * (1 if op.name == "arith.addi" else -1)
                v = ops[0]
            elif op.name == "arith.addi" and const_int(ops[0]) is not None:
                step += const_int(ops[0])
                v = ops[1]
            else:
                return None
        return step

    def passes_through(self, v, op):
        """Whether `v` is the result of `op` but for integer casts."""
        while not isinstance(v, self.ir.BlockArgument):
            o = v.owner.operation
            if _same(o, op):
                return True
            if o.name not in _CASTS:
                return False
            v = o.operands[0]
        return False

    def capacity(self, source, dims, read):
        """Each dim's bound on its extent: its loop's trip count when its bounds are constants, else
        how many elements of `source` its stride reaches; refused when neither is static."""
        ir = self.ir
        ty = ir.MemRefType(source.type)
        numel = ty.shape[0] if ty.shape[0] != ir.ShapedType.get_dynamic_size() else None
        out = []
        for d, dim in enumerate(dims):
            lb = dim.lb if isinstance(dim.lb, int) else const_int(dim.lb)
            ub = dim.ub if isinstance(dim.ub, int) else const_int(dim.ub)
            if lb is not None and ub is not None:
                out.append(max(1, -(-(ub - lb) // dim.step)))
            elif numel is not None:
                out.append((numel - 1) // abs(dim.stride) + 1)
            else:
                raise NoCapacity(f"attribute of {_short(read.op)} has no static capacity on dim {d}")
        return out

    def record_array(self, read, plan, i):
        """The A record before the outermost loop, its base and extents made there; R at the read, its
        trips made beside it."""
        from mlir.dialects import arith
        ir = self.ir
        source, outer, const, dims, cap = plan
        idx = ir.IndexType.get()
        made = []

        def keep(v):
            made.append(v.owner.operation)
            return v

        def cst(n):
            return keep(arith.ConstantOp(idx, ir.IntegerAttr.get(idx, n)).result)

        def index(v):
            if isinstance(v, int):
                return cst(v)
            return v if isinstance(v.type, ir.IndexType) else keep(arith.IndexCastOp(idx, v).result)

        with ir.InsertionPoint(read.loops[0]), read.op.location:
            base = cst(const)
            for v, k in outer:
                base = keep(arith.AddIOp(base, keep(arith.MulIOp(index(v), cst(k)).result)).result)
            extents = []
            for d in dims:
                span = keep(arith.SubIOp(index(d.ub), index(d.lb)).result)
                trips = keep(arith.CeilDivSIOp(span, cst(d.step)).result)
                extents.append(keep(arith.MaxSIOp(trips, cst(0)).result))
            whole = ir.Operation.create(RECORD, results=[], operands=[source, base] + extents,
                                        attributes=self.attrs(i, len(dims), "A", [0, 1, 1, 0, len(dims)],
                                                              strides=[d.stride for d in dims], capacity=cap))
        with ir.InsertionPoint(self.next_op(read.op)), read.op.location:
            trips = []
            for d in dims:
                t = keep(arith.SubIOp(index(loop_like(d.loop).ivs[0]), index(d.lb)).result)
                trips.append(keep(arith.DivSIOp(t, cst(d.step)).result) if d.step != 1 else t)
            rec = ir.Operation.create(RECORD, results=[read.value.type], operands=[read.value] + trips,
                                      attributes=self.attrs(i, len(dims), "R", [1, 0, 0, len(dims), 0]))
        self.yield_record(read, rec)
        for o in made:
            o.attributes[PLUMBING] = ir.UnitAttr.get()
        return [whole, rec]
