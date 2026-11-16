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
    """The extent per axis (x, y, z) the compile assumes of a launch: for an axis the kernel reads a
    program id on, ceil(N / B) where N is the constant a `N - pid * B` clamp counts down from and B the
    constant the id is scaled by (the grid rule: an axis covers its numel in tile steps), the largest
    over the clamps; `grid`'s extent where `rule` is off or no such clamp is there; None for an axis
    with no program id."""
    from ...dialect.pytorchsim import ProgramIdOp
    from .ir_utils import funcs, walk
    fallback = (list(grid) + [1, 1, 1])[:3] if grid else [None] * 3
    out = [None] * 3
    for fn in funcs(module):
        ids = [o for o in walk(fn) if isinstance(o.opview, ProgramIdOp)]
        subs = [o for o in walk(fn) if isinstance(o.opview, (arith.SubIOp, arith.MinSIOp))]
        for a in sorted({o.opview.axis.value for o in ids}):
            got = _get_rule_extent([o for o in ids if o.opview.axis.value == a], subs) if rule else None
            got = fallback[a] if got is None else got
            out[a] = got if out[a] is None else max(out[a], got)
    return tuple(out)


def _get_rule_extent(ids, subs):
    """ceil(N / B) of the largest `N - pid * B` over `ids` (one axis's program ids), or None."""
    scaled = {}
    for i in ids:
        for u in i.results[0].uses:
            o = op_of(u.owner)
            if isinstance(o.opview, arith.MulIOp):
                b = const_int(o.operands[1 if o.operands[0] == i.results[0] else 0])
                if b is not None and b > 0:
                    scaled[o.results[0]] = b
    work = list(scaled)
    while work:
        v = work.pop()
        for u in v.uses:
            o = op_of(u.owner)
            if isinstance(o.opview, _CASTS) and o.results[0] not in scaled:
                scaled[o.results[0]] = scaled[v]
                work.append(o.results[0])
    best = None
    for o in subs:
        n, x = _get_clamp_bound(o, scaled)
        if n is not None and n > 0:
            ext = -(-n // scaled[x])
            best = ext if best is None else max(best, ext)
    return best


def _get_clamp_bound(o, scaled):
    """(N, the scaled value) of `N - x` or of `min(x + c, N)` with N constant, else (None, None)."""
    if isinstance(o.opview, arith.SubIOp):
        n, x = const_int(o.operands[0]), o.operands[1]
        return (n, x) if x in scaled else (None, None)
    for a, b in (o.operands, reversed(list(o.operands))):
        n, p = const_int(b), producer(a)
        if n is None:
            continue
        if a in scaled:
            return n, a
        if p is not None and isinstance(p.opview, arith.AddIOp):
            for u, c in (p.operands, reversed(list(p.operands))):
                if u in scaled and const_int(c) is not None:
                    return n, u
    return None, None


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
