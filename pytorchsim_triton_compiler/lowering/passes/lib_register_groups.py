"""lib_register_groups: the register groups a lane's own row is held in on RVV -- the LMULs a group takes
(GROUPS), the group a row of n elements fits (register_group), and the integer element a value is moved
as (integer_view); a row no group holds is refused (LaneBroadcastRefused, LaneReductionRefused)."""

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
