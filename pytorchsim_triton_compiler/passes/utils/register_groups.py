"""register_groups: the register groups a lane's own row is held in on RVV -- the LMULs a group takes
(GROUPS), the group a row of n elements fits (register_group), the integer element a value is moved as
(integer_view), and the registers a run of ops keeps live at once (registers_live); a row no group holds
is refused (LaneBroadcastRefused, LaneReductionRefused)."""

GROUPS = (1, 2, 4, 8)


class LaneBroadcastRefused(Exception):
    """A lane value's broadcast no register group takes: refused, never scalarized."""


class LaneReductionRefused(Exception):
    """A lane reduction no RVV instruction or register group takes: refused, never scalarized."""


def integer_view(elem):
    """(the integer element a lane value of `elem` is gathered as, the cast there, the cast back)."""
    from mlir.dialects import arith
    from mlir.ir import BF16Type, F16Type, F32Type, F64Type, IndexType, IntegerType
    if isinstance(elem, IndexType):
        return IntegerType.get_signless(64), arith.IndexCastOp, arith.IndexCastOp
    if isinstance(elem, IntegerType) and elem.width == 1:
        return IntegerType.get_signless(8), arith.ExtUIOp, arith.TruncIOp
    if isinstance(elem, IntegerType) and elem.width in (8, 16, 32, 64):
        return elem, None, None
    if isinstance(elem, (BF16Type, F16Type, F32Type, F64Type)):
        return IntegerType.get_signless(elem.width), arith.BitcastOp, arith.BitcastOp
    raise LaneBroadcastRefused(f"a lane value of element type {elem} has no integer view to gather")


def register_group(n, bits, vlen):
    """(next_pow2(n), the LMUL of the group that holds that many `bits`-bit elements at `vlen`)."""
    whole = 1 << (n - 1).bit_length()
    m = next((m for m in GROUPS if whole * bits <= m * vlen), None)
    if m is None:
        raise LaneBroadcastRefused(
            f"a lane value broadcast to {n} x i{bits} needs {whole * bits // vlen} registers at "
            f"VLEN {vlen}, above one LMUL-{GROUPS[-1]} group")
    return whole, m


def registers_live(ops, carried, words, vlen, free=lambda v: False):
    """The vector registers `ops` (in order) keep live at once, each vector value `words` of its own
    element wide (an index 64 bits, a mask a register): the most their results hold together, the
    vectors they take from outside, and the `carried` ones a loop around them carries. A value `free`
    says holds none (a splat constant, which goes to an instruction as a scalar) counts none."""
    from mlir.ir import BlockArgument, IndexType, VectorType
    from .ir_utils import elem_bits

    def regs(v):
        if not isinstance(v.type, VectorType) or VectorType(v.type).shape == [1] or free(v):
            return 0
        e = VectorType(v.type).element_type
        bits = 64 if isinstance(e, IndexType) else elem_bits(e)
        return -(-words * bits // vlen)
    made = {r for o in ops for r in o.results}
    last = {}
    for i, o in enumerate(ops):
        for v in o.operands:
            last[v] = i
    outside = {v for o in ops for v in o.operands
               if v not in made and not isinstance(v, BlockArgument) and isinstance(v.type, VectorType)}
    peak = 0
    for i in range(len(ops)):
        peak = max(peak, sum(regs(r) for o in ops[:i + 1] for r in o.results if last.get(r, -1) >= i))
    return peak + sum(regs(v) for v in outside) + sum(regs(a) for a in carried)
