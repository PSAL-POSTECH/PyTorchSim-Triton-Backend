"""pytorchsim-check-lane-values-stay-vector: no op TO_LLVM leaves makes a lane's value a scalar,
and no access through a scratchpad pointer carries what a lane's own bank cannot take. Read off the
types and the memory effects an op states, whatever the op is called."""

from ...dialects.torchsim import UNKNOWN, effects_on
from .lib_bank_layout import SPAD_SPACE
from .lib_ir import walk

NAME = "pytorchsim-check-lane-values-stay-vector"

INDEXED = ("a vector of scratchpad pointers: each lane's address is its own, which no lane's bank "
           "is addressed by")
TWO_MEMORIES = ("an access through a scratchpad pointer and another pointer at once: bytes moved "
                "between memories that no value carries, so no lane takes its own")
SCALAR_WRITE = ("a write through a scratchpad pointer of no vector: a scalar store takes no lane's "
                "offset, so it reaches lane 0's bank alone")
SUB_BYTE = ("an access through a scratchpad pointer whose vectors are all of sub-byte elements: a "
            "buffer of them holds a byte per element where the vector packs bits")


class LaneValueMadeScalar(Exception):
    """TO_LLVM left an op that makes a lane's own value a scalar, which the one scalar register
    file all lanes share holds for lane 0 only."""


class ScratchpadAccessRefused(Exception):
    """TO_LLVM left an access through a scratchpad pointer that a lane's bank cannot take as the
    lane's own; `reason` is the refused class."""

    def __init__(self, message, reason):
        super().__init__(message)
        self.reason = reason


def run(module, **_):
    """Refuse `module` if an op of it makes a vector's element a scalar, or accesses the scratchpad
    in a refused class."""
    made = ops_making_a_vector_scalar(module)
    if made:
        raise LaneValueMadeScalar(
            f"TO_LLVM left {len(made)} op(s) that make a lane's value a scalar, which every "
            f"lane's one shared scalar register file holds for lane 0 only. First: "
            f"{' '.join(str(made[0]).split())[:200]} at {made[0].location}")
    refused = spad_accesses_refused(module)
    if refused:
        op, why = refused[0]
        raise ScratchpadAccessRefused(
            f"TO_LLVM left {len(refused)} scratchpad access(es) a lane's bank cannot take. First, "
            f"{why}: {' '.join(str(op).split())[:200]} at {op.location}", why)
    return module


def ops_making_a_vector_scalar(module):
    """Every op of `module` with a vector operand and an integer or float scalar result."""
    from mlir.ir import FloatType, IntegerType, VectorType
    return [op for op in walk(module.operation)
            if any(isinstance(x.type, VectorType) for x in op.operands)
            and any(isinstance(r.type, (IntegerType, FloatType)) for r in op.results)]


def spad_accesses_refused(module):
    """(op, reason) for every op of `module` holding a vector of scratchpad pointers, stating it
    reaches memory through a scratchpad pointer and another or writes through one no vector, or
    reaching one -- as it states, or stating nothing -- with nothing but sub-byte vectors."""
    got = []
    for op in walk(module.operation):
        values = list(op.operands) + list(op.results)
        if any(_spad_pointers(v.type) for v in values):
            got.append((op, INDEXED))
            continue
        reached = [v for v in op.operands if _is_pointer(v.type) and _reaches(op, v)]
        if not any(_is_spad(v.type) for v in reached):
            continue
        if sum(effects_on(op, v) != UNKNOWN for v in reached) > 1:
            got.append((op, TWO_MEMORIES))
            continue
        if any(_is_spad(v.type) and "write" in effects_on(op, v) for v in reached) \
                and not any(_is_vector(v.type) for v in op.operands):
            got.append((op, SCALAR_WRITE))
            continue
        vectors = [v.type for v in values if _is_vector(v.type)]
        if vectors and all(_sub_byte(t) for t in vectors):
            got.append((op, SUB_BYTE))
    return got


def _reaches(op, v):
    """Whether `op` reads or writes memory through `v`, or states nothing that says it does not."""
    said = effects_on(op, v)
    return said == UNKNOWN or bool(said & {"read", "write"})


def _is_pointer(ty):
    from mlir.dialects import llvm
    return isinstance(ty, llvm.PointerType)


def _is_spad(ty):
    from mlir.dialects import llvm
    return _is_pointer(ty) and llvm.PointerType(ty).address_space == SPAD_SPACE


def _is_vector(ty):
    from mlir.ir import VectorType
    return isinstance(ty, VectorType)


def _spad_pointers(ty):
    from mlir.ir import VectorType
    return _is_vector(ty) and _is_spad(VectorType(ty).element_type)


def _sub_byte(ty):
    from mlir.ir import IntegerType, VectorType
    elem = VectorType(ty).element_type
    return isinstance(elem, IntegerType) and IntegerType(elem).width % 8 != 0
