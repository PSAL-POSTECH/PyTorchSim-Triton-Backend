"""What an integer of the kernel can be: `interval` of a clamp or an index, given that each program
id is below its axis of the launch grid (`program_ids`). Shared by every pass that asks how far a
masked tile reaches (torchsim-fit-to-numel, the matrix unit's lowering)."""

from .lib_ir import const_int, op_of, producer

def interval(v, known, depth=0):
    """(low, high), inclusive, of the integer `v`, or None: constants, the values `known` gives
    (the program ids, each within its axis of the launch grid), and what arith's add, sub, mul,
    min, max, and div and rem of non-negatives by positives make of them. An over-estimate: the
    two sides of an op are taken apart, so `x - x` is not 0."""
    for value, bounds in known:
        if value == v:
            return bounds
    o = producer(v)
    if o is None or depth > 64:
        return None
    c = const_int(v)
    if c is not None:
        return c, c
    if o.name in ("arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui"):
        return interval(o.operands[0], known, depth + 1)
    if o.name not in ("arith.addi", "arith.subi", "arith.muli", "arith.minsi", "arith.maxsi",
                      "arith.divsi", "arith.remsi"):
        return None
    a = interval(o.operands[0], known, depth + 1)
    b = interval(o.operands[1], known, depth + 1)
    if a is None or b is None:
        return None
    if o.name == "arith.addi":
        return a[0] + b[0], a[1] + b[1]
    if o.name == "arith.subi":
        return a[0] - b[1], a[1] - b[0]
    if o.name == "arith.muli":
        corners = [x * y for x in a for y in b]
        return min(corners), max(corners)
    if o.name == "arith.minsi":
        return min(a[0], b[0]), min(a[1], b[1])
    if o.name == "arith.maxsi":
        return max(a[0], b[0]), max(a[1], b[1])
    if a[0] < 0 or b[0] <= 0:
        return None
    if o.name == "arith.divsi":
        return a[0] // b[1], a[1] // b[0]
    return 0, min(a[1], b[1] - 1)


def program_id_args(func):
    """The kernel's program ids on x, y and z: the last three integer arguments of its function."""
    import mlir.ir as ir
    args = [a for a in func.regions[0].blocks[0].arguments if isinstance(a.type, ir.IntegerType)]
    return list(args[-3:]) if len(args) >= 6 else []


def program_ids(op, grid):
    """What is known of the kernel's program ids, for interval(): each below its axis of the launch
    `grid`."""
    func = getattr(op, "operation", op)
    while func.name != "func.func":
        func = op_of(func.parent)
    ids = program_id_args(func)
    if not ids or not grid:
        return []           # no launch grid said: nothing is known of the ids
    g = (list(grid) + [1, 1, 1])[:3]
    return [(arg, (0, max(1, int(n)) - 1)) for arg, n in zip(ids, g)]
