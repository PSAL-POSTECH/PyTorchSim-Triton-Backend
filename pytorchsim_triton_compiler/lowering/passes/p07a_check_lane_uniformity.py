"""pytorchsim-check-lane-uniformity: a UNIFORM (one-bank) buffer is written only with values
proven the same in every lane, by a FOLD or by an EVERY_LANE move; a DISTRIBUTED one is never stored
to or read from as a scalar. Lanes differ only where an op declares LaneVariant or reads such a bank."""

from ...dialects.torchsim import effects_on, is_lane_variant, is_memory_effect_free
from .lib_bank_layout import in_scratchpad, is_uniform
from .lib_cross_lane import CROSS_LANE_KIND, EVERY_LANE, FOLD
from .lib_ir import funcs, is_memref, op_of, values, walk

NAME = "pytorchsim-check-lane-uniformity"


class LaneUniformityRefused(Exception):
    """An access breaks what its buffer's placement promises: a VARIANT value, or a move that is no
    FOLD and not EVERY_LANE, into a UNIFORM buffer, or a scalar access to a DISTRIBUTED one."""


def run(module, **_):
    """Refuse `module` at its first access that breaks its buffer's placement."""
    bad = violations(module)
    if bad:
        op, why = bad[0]
        raise LaneUniformityRefused(
            f"{len(bad)} access(es) break their scratchpad buffer's placement. First: {why}: "
            f"{' '.join(str(op).split())[:300]} at {op.location}")
    return module


def violations(module):
    """(op, why) of every access in `module` that breaks its buffer's placement."""
    out = []
    for func in funcs(module):
        uniform = uniform_values(func)
        for op in walk(func.operation):
            why = _refusal(op, uniform)
            if why is not None:
                out.append((op, why))
    return out


def uniform_values(func):
    """The values of `func` proven the same in every lane: the greatest set closed under _proven,
    so a loop-carried value is UNIFORM when its init and every yield are."""
    from mlir.ir import BlockArgument
    entry = list(func.regions[0].blocks)[0]
    got = {v for v in values(func.operation) if not is_memref(v)}
    while True:
        drop = {v for v in got if not (isinstance(v, BlockArgument) and v.owner == entry)
                and not _proven(v, got)}
        if not drop:
            return got
        got -= drop


def _is_uniform_value(v, uniform):
    return is_memref(v) or v in uniform


def _proven(v, uniform):
    from mlir.ir import BlockArgument
    if isinstance(v, BlockArgument):
        return _region_inputs_uniform(op_of(v.owner.owner), uniform)
    o = op_of(v.owner)
    if len(o.regions):
        return _region_inputs_uniform(o, uniform)
    if not all(_is_uniform_value(x, uniform) for x in o.operands):
        return False
    if is_lane_variant(o):
        return False
    effects = {x: effects_on(o, x) for x in list(o.operands) + list(o.results)}
    if any("unknown" in e for e in effects.values()):
        return False
    if is_memory_effect_free(o):
        return True
    if not any(effects.values()):
        return False
    return all(not e or (e == frozenset({"read"}) and is_memref(x) and _reads_the_same(x))
               for x, e in effects.items())


def _region_inputs_uniform(o, uniform):
    """Every operand `o` takes and every value its regions' terminators hand back is UNIFORM, so no
    block argument of it, nor result, can differ between lanes."""
    if not all(_is_uniform_value(x, uniform) for x in o.operands):
        return False
    for region in o.regions:
        for block in region.blocks:
            ops = list(block.operations)
            if ops and not all(_is_uniform_value(x, uniform) for x in ops[-1].operands):
                return False
    return True


def _reads_the_same(m):
    """Does every lane read the same data from memref `m`: DRAM, or a one-bank scratchpad buffer?"""
    return not in_scratchpad(m.type) or is_uniform(m)


def _refusal(op, uniform):
    """Why `op`'s access breaks its buffer's placement, or None. EVERY_LANE is the provider's statement
    read off the pairs its move carries; that the rounds keep those pairs, lib_xlu's units hold."""
    from mlir.ir import StringAttr
    if any(_is_scalar(r.type) for r in op.results):
        spread = [x for x in op.operands if is_memref(x) and in_scratchpad(x.type)
                  and "read" in effects_on(op, x) and not is_uniform(x)]
        if spread:
            return (f"a scalar {op.results[0].type} is read from DISTRIBUTED buffer "
                    f"{spread[0].type}: a scalar is one value for every lane, and no index picks "
                    f"the bank it reads")
    for m in op.operands:
        if not (is_memref(m) and in_scratchpad(m.type) and "write" in effects_on(op, m)):
            continue
        if len(op.regions):
            return (f"{op.name} writes scratchpad buffer {m.type} from a body of its own, which "
                    f"this check does not read")
        reads = [x for x in op.operands if is_memref(x) and "read" in effects_on(op, x)]
        data = [x for x in op.operands if not is_memref(x)]
        if is_uniform(m):
            if CROSS_LANE_KIND in op.attributes:
                kind = StringAttr(op.attributes[CROSS_LANE_KIND]).value
                if kind == FOLD or EVERY_LANE in op.attributes:
                    continue
                return (f"a {kind} move between lanes writes UNIFORM buffer {m.type}, and of the "
                        f"moves between lanes only a {FOLD} or an {EVERY_LANE} one leaves every "
                        f"lane the same")
            variant = [x for x in data if x not in uniform]
            if variant:
                return f"a lane-variant {variant[0].type} is written into UNIFORM buffer {m.type}"
            spread = [x for x in reads if not _reads_the_same(x)]
            if spread:
                return (f"UNIFORM buffer {m.type} is written from DISTRIBUTED buffer "
                        f"{spread[0].type}, which differs between lanes")
            continue
        scalar = [x for x in data if _is_scalar(x.type)]
        if scalar and not reads and not any(_is_vector(x.type) for x in data):
            return (f"a scalar {scalar[0].type} is stored into DISTRIBUTED buffer {m.type}: a "
                    f"scalar is one value for every lane, and no index picks the bank it writes")
    return None


def _is_scalar(ty):
    """An integer, float or index, or a vector of rank 0: one value, the same in every lane."""
    from mlir.ir import FloatType, IndexType, IntegerType, VectorType
    return (isinstance(ty, (IntegerType, FloatType, IndexType))
            or isinstance(ty, VectorType) and VectorType(ty).rank == 0)


def _is_vector(ty):
    from mlir.ir import VectorType
    return isinstance(ty, VectorType) and VectorType(ty).rank > 0
