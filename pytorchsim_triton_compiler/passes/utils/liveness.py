"""What an integer of the kernel can be: `interval` of a clamp or an index, given that each program
id is below its axis of the launch grid (`program_ids`). Shared by every pass that asks how far a
masked tile reaches (pytorchsim-drop-unread-padding, the matrix unit's lowering)."""

from mlir.dialects import arith

from .ir_utils import const_int, op_of, producer

_CASTS = (arith.IndexCastOp, arith.IndexCastUIOp, arith.ExtSIOp, arith.ExtUIOp)
_RANGED = (arith.AddIOp, arith.SubIOp, arith.MulIOp, arith.MinSIOp, arith.MaxSIOp, arith.DivSIOp,
           arith.RemSIOp)


def interval(v, known, depth=0):
    """(low, high), inclusive, of the integer `v`, or None: constants, the values `known` maps to theirs
    (the program ids, each within its axis of the launch grid), and what arith's add, sub, mul,
    min, max, and div and rem of non-negatives by positives make of them. An over-estimate: the
    two sides of an op are taken apart, so `x - x` is not 0."""
    if v in known:
        return known[v]
    o = producer(v)
    if o is None or depth > 64:
        return None
    c = const_int(v)
    if c is not None:
        return c, c
    if isinstance(o.opview, _CASTS):
        return interval(o.operands[0], known, depth + 1)
    if not isinstance(o.opview, _RANGED):
        return None
    a = interval(o.operands[0], known, depth + 1)
    b = interval(o.operands[1], known, depth + 1)
    if a is None or b is None:
        return None
    if isinstance(o.opview, arith.AddIOp):
        return a[0] + b[0], a[1] + b[1]
    if isinstance(o.opview, arith.SubIOp):
        return a[0] - b[1], a[1] - b[0]
    if isinstance(o.opview, arith.MulIOp):
        corners = [x * y for x in a for y in b]
        return min(corners), max(corners)
    if isinstance(o.opview, arith.MinSIOp):
        return min(a[0], b[0]), min(a[1], b[1])
    if isinstance(o.opview, arith.MaxSIOp):
        return max(a[0], b[0]), max(a[1], b[1])
    if a[0] < 0 or b[0] <= 0:
        return None
    if isinstance(o.opview, arith.DivSIOp):
        return a[0] // b[1], a[1] // b[0]
    return 0, min(a[1], b[1] - 1)


def program_ids(op, grid):
    """What is known of the kernel's program ids, for interval(): each pytorchsim.program_id of `op`'s
    function is below its axis of `grid` (grid_bound's answer; an axis it leaves None is unknown);
    with no grid said, nothing is known of them."""
    from mlir.dialects import func as func_d
    from ...dialect.pytorchsim import ProgramIdOp
    from .ir_utils import walk
    fn = getattr(op, "operation", op)
    while not isinstance(fn.opview, func_d.FuncOp):
        fn = op_of(fn.parent)
    if not grid:
        return {}
    g = (list(grid) + [1, 1, 1])[:3]
    return {o.results[0]: (0, max(1, int(g[o.opview.axis.value])) - 1)
            for o in walk(fn) if isinstance(o.opview, ProgramIdOp) and g[o.opview.axis.value] is not None}


def grid_bound(module, grid, rule=True):
    """The extent per axis (x, y, z) the compile assumes of a launch: for an axis, the largest first dead
    program among the DRAM accesses whose address moves with its id, where an access is dead once one of its
    mask extents (tts.load/store dims) is <= 0 for every later id; `grid`'s extent where `rule` is off, or
    any such access (unmasked, low-bounded only, gather, ...) has no dead program; None for an axis with no id."""
    from ...dialect.pytorchsim import ProgramIdOp
    from .ir_utils import funcs, walk
    fallback = (list(grid) + [1, 1, 1])[:3] if grid else [None] * 3
    out = [None] * 3
    for fn in funcs(module):
        ops = walk(fn)
        accesses = [o for o in ops if o.name in _ACCESSES]
        for a in sorted({o.opview.axis.value for o in ops if isinstance(o.opview, ProgramIdOp)}):
            got = _get_dead_from(accesses, a) if rule else None
            got = fallback[a] if got is None else got
            out[a] = got if out[a] is None else max(out[a], got)
    return tuple(out)


_ACCESSES = {"tts.load", "tts.store", "tts.gather", "tts.scatter", "memref.load", "memref.store", "tt.load",
             "tt.store", "tt.atomic_rmw", "tt.atomic_cas"}
_DYNAMIC = -(1 << 63)


def _get_dead_from(accesses, axis):
    """The first program id of `axis` past every access that moves with it, or None when one of them has none."""
    firsts = []
    for o in accesses:
        if not _is_moved_by(o, axis):
            continue
        p = _get_access_dead_from(o, axis)
        if p is None:
            return None
        firsts.append(p)
    return max(firsts) if firsts else None


def _get_access_dead_from(o, axis):
    """The first program id from which one of `o`'s mask extents stays <= 0, or None."""
    from .ir_utils import int_array
    if o.name not in ("tts.load", "tts.store") or "static_mask_dims" not in o.attributes:
        return None
    seg = [int(x) for x in o.attributes["operandSegmentSizes"]]
    first = 1 if o.name == "tts.load" else 2
    dyn = iter(list(o.operands)[first:first + seg[1 if o.name == "tts.load" else 2]])
    best = None
    for d in int_array(o.attributes["static_mask_dims"]):
        f = _eval_in_pid(next(dyn), axis) if d == _DYNAMIC else [(0, 0, d)]
        p = None if f is None else _get_dead_suffix(f)
        if p is not None:
            best = p if best is None else min(best, p)
    return None if best is None else max(1, best)


def _is_moved_by(o, axis):
    """Whether a program id of `axis` may reach an operand of `o`; a loop argument or unknown leaf says yes."""
    from mlir.ir import BlockArgument
    from ...dialect.pytorchsim import ProgramIdOp
    from .ir_utils import op_of, walk
    work, seen = list(o.operands), set()
    while work:
        v = work.pop()
        if v in seen:
            continue
        seen.add(v)
        if isinstance(v, BlockArgument):
            owner = op_of(v.owner.owner)
            if owner.name not in ("func.func", "linalg.generic"):
                return True
            continue
        d = op_of(v.owner)
        if isinstance(d.opview, ProgramIdOp):
            if d.opview.axis.value == axis:
                return True
            continue
        work += list(d.operands)
        for inner in walk(d):
            work += list(inner.operands)
    return False


def _eval_in_pid(v, axis, depth=0):
    """`v` as a piecewise affine function of the id of `axis`, [(start, a, b)] meaning a*p + b from start
    to the next start, or None when arith's add, sub, mul by a constant, min, max and casts do not make it."""
    from ...dialect.pytorchsim import ProgramIdOp
    o = producer(v)
    if o is None or depth > 64:
        return None
    if isinstance(o.opview, ProgramIdOp):
        return [(0, 1, 0)] if o.opview.axis.value == axis else None
    c = const_int(v)
    if c is not None:
        return [(0, 0, c)]
    if isinstance(o.opview, _CASTS):
        return _eval_in_pid(o.operands[0], axis, depth + 1)
    if not isinstance(o.opview, (arith.AddIOp, arith.SubIOp, arith.MulIOp, arith.MinSIOp, arith.MaxSIOp)):
        return None
    f, g = (_eval_in_pid(x, axis, depth + 1) for x in o.operands)
    if f is None or g is None:
        return None
    if isinstance(o.opview, arith.MulIOp):
        k, h = (f, g) if f == [(0, 0, f[0][2])] else (g, f)
        return None if k != [(0, 0, k[0][2])] else [(s, a * k[0][2], b * k[0][2]) for s, a, b in h]
    cuts = sorted({s for s, _, _ in f} | {s for s, _, _ in g})
    if isinstance(o.opview, (arith.MinSIOp, arith.MaxSIOp)):
        cuts = sorted(set(cuts) | {x for s in cuts for x in _get_crossings(f, g, s, cuts)})
    out = []
    for s in cuts:
        (_, a1, b1), (_, a2, b2) = _piece_at(f, s), _piece_at(g, s)
        if isinstance(o.opview, arith.AddIOp):
            out.append((s, a1 + a2, b1 + b2))
        elif isinstance(o.opview, arith.SubIOp):
            out.append((s, a1 - a2, b1 - b2))
        else:
            lower = isinstance(o.opview, arith.MinSIOp)
            first = (a1 * s + b1, a1) <= (a2 * s + b2, a2)
            out.append((s, a1, b1) if first == lower else (s, a2, b2))
    return out


def _piece_at(f, p):
    return [x for x in f if x[0] <= p][-1]


def _get_crossings(f, g, s, cuts):
    """The integer points around where the two pieces of f and g live at `s` cross, inside that segment."""
    (_, a1, b1), (_, a2, b2) = _piece_at(f, s), _piece_at(g, s)
    end = next((c for c in cuts if c > s), None)
    if a1 == a2:
        return []
    x = (b2 - b1) // (a1 - a2)
    return [p for p in (x, x + 1) if p > s and (end is None or p < end)]


def _get_dead_suffix(f):
    """The smallest p from which the piecewise affine `f` is <= 0 for every larger p, or None if it never is."""
    s, a, b = f[-1]
    if a > 0 or (a == 0 and b > 0):
        return None
    ends = [f[i + 1][0] - 1 for i in range(len(f) - 1)] + [None]
    for i in range(len(f) - 1, -1, -1):
        s, a, b = f[i]
        e = ends[i]
        if e is not None and a * e + b > 0:
            return e + 1
        if a * s + b <= 0:
            if i == 0:
                return s
            continue
        return -(-b // -a)
    return 0


class ProgramIdArgMissing(RuntimeError):
    """A function holds a program id and not the trailing i32 arguments the ABI hands it in."""


def lower_program_ids(module):
    """Each pytorchsim.program_id made the function argument the kernel object's ABI hands its axis
    (kernel_object.TRAILING_GRID_PARAMS, after the kernel's own): what the TOG and LLVM read; until then the
    op states the id's range to the integer range analysis."""
    from ...contract.kernel_object import TRAILING_GRID_PARAMS
    from ...dialect.pytorchsim import ProgramIdOp
    from mlir.ir import IntegerType
    from .ir_utils import funcs, walk
    for func in funcs(module):
        for op in [o for o in walk(func.operation) if isinstance(o.opview, ProgramIdOp)]:
            args = list(func.regions[0].blocks[0].arguments)
            tail = args[len(args) - len(TRAILING_GRID_PARAMS):]
            if len(args) < len(TRAILING_GRID_PARAMS) or not all(
                    isinstance(a.type, IntegerType) and IntegerType(a.type).width == 32 for a in tail):
                raise ProgramIdArgMissing(
                    f"{func.attributes['sym_name']} holds a program id and does not end in the "
                    f"{len(TRAILING_GRID_PARAMS)} i32 arguments {TRAILING_GRID_PARAMS}")
            axis = op.opview.axis.value
            op.results[0].replace_all_uses_with(tail[TRAILING_GRID_PARAMS.index("pid" + "XYZ"[axis])])
            op.erase()
    return module
