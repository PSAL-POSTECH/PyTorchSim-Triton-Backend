"""The `torchsim` dialect: the python extension it lives in, the op classes ODS generates from the
same .td, and what ODS cannot state -- which of a transfer's attributes speak per axis, and the
`dram_arg` values that name no argument."""
import glob
import hashlib
import importlib.util
import os
import sys

BUILD = os.path.join(os.environ.get("TORCHSIM_PREFIX", "/workspace"), "torchsim-dialect-build")

_done = [False]
_ext = [None]


def _load_from_build(name, pattern):
    hits = sorted(glob.glob(os.path.join(BUILD, pattern)))
    if not hits:
        raise ImportError(
            f"{os.path.join(BUILD, pattern)} is missing: the dialect build at {BUILD} "
            f"did not produce it: `setup/toolchain.sh dialect`.")
    spec = importlib.util.spec_from_file_location(f"{__package__}.{name}", hits[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_generated(name):
    return _load_from_build(name, name + ".py")


def _refuse_a_stale_build():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "include", "TorchSim")
    want = {}
    for name in ("TorchSimOps.td", "TorchSimDialect.td"):
        path = os.path.join(src, name)
        if not os.path.exists(path):
            return
        want[name] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    try:
        got = _load_generated("_torchsim_stamp").TD_SHA256
    except ImportError:
        raise ImportError(
            f"the dialect build at {BUILD} carries no .td stamp, so nothing says which "
            f"schema its op classes came from. Rebuild it -- `ninja -C {BUILD}`.") from None
    stale = sorted(n for n in want if want[n] != got.get(n))
    if stale:
        raise ImportError(
            f"the dialect build at {BUILD} was generated from a different "
            f"{', '.join(stale)}: its op classes and operand table are that .td's, not "
            f"this checkout's, and an op built from them can be silently wrong. "
            f"Rebuild it -- `ninja -C {BUILD}`.")


_refuse_a_stale_build()
_load_generated("_torchsim_ops_gen")
from ._torchsim_ops_gen import *  # noqa: E402,F401,F403
from ._torchsim_ops_gen import TransferOp, WaitOp  # noqa: E402,F401
_TABLE = _load_generated("_torchsim_operands")


def register():
    """Put `torchsim` in the bindings' GLOBAL registry, once, its passes in the pass registry, and
    let `.opview` find the classes above."""
    if _done[0]:
        return
    from mlir._mlir_libs import _mlir, get_dialect_registry
    ext = _load_from_build("_torchsim_ext", "_torchsim_ext*.so")
    ext.insert_into_registry(get_dialect_registry())
    ext.register_passes()
    _ext[0] = ext
    _mlir.globals.append_dialect_search_prefix(__package__)
    _done[0] = True


EFFECTS = ("read", "write", "allocate", "free", "sync")
"""What effects_on can answer, in the bit order _torchsim_ext states them; "sync" is an effect on
the transfers in flight into or out of a buffer (torchsim::InFlight), not on its data."""
UNKNOWN = frozenset({"unknown"})


def effects_on(op, value):
    """Which of EFFECTS `op` states it has on `value`, off MemoryEffectOpInterface; UNKNOWN when
    `op` states no effects at all."""
    bits = _ext[0].effects_on(getattr(op, "operation", op), value)
    if bits < 0:
        return UNKNOWN
    return frozenset(name for k, name in enumerate(EFFECTS) if bits >> k & 1)


def view_root(value):
    """The buffer `value` is a view of, through every ViewLikeOpInterface in its memory space."""
    return _ext[0].view_root(value)


def operand_number(op, name, k=0):
    """Where the `k`-th operand of ODS argument `name` sits among `op`'s operands: the .td's order,
    and the op's own `operandSegmentSizes` where it has one."""
    from mlir.ir import DenseI32ArrayAttr
    op = getattr(op, "operation", op)
    i = _TABLE.OPERANDS[op.name].index(name)
    if "operandSegmentSizes" not in op.attributes:
        return i + k
    sizes = list(DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))
    return sum(sizes[:i]) + k


def tile_shape(sram_type):
    """The tile a scratchpad buffer of `sram_type` holds, a scalar's being one element -- the one
    definition of that, which a transfer's descriptor and its TOG node both state."""
    from mlir.ir import ShapedType
    return list(ShapedType(sram_type).shape) or [1]


def moved_shape(transfer):
    """The shape one transfer moves: its `subtile_size` if it has one, else its scratchpad tile.
    prod(moved_shape) is what it signals its semaphore with -- the one definition of that."""
    from mlir.ir import ArrayAttr, IntegerAttr
    op = getattr(transfer, "operation", transfer)
    if "subtile_size" in op.attributes:
        return [IntegerAttr(a).value for a in ArrayAttr(op.attributes["subtile_size"])]
    return tile_shape(op.opview.sram.type)


def is_load(transfer):
    """Whether a transfer moves DRAM into the scratchpad: the one definition every pass shares, so
    a load and a store are told apart in a single place."""
    return is_load_kind(getattr(transfer, "operation", transfer).opview.dma_kind.value)


def is_load_kind(dma_kind):
    """Whether a `dma_kind` value names a load -- for a transfer not built yet; else is_load."""
    return dma_kind.upper().startswith("MVIN")


def stated_attributes(op_name):
    """Every attribute ODS lets `op_name` state of itself, with the segment sizes a variadic op
    carries -- what is NOT one of these was put there by a pass."""
    return frozenset(_TABLE.ATTRIBUTES[op_name]) | {"operandSegmentSizes"}


AXIS_NUMBERS = "a list of axis numbers"
AXIS_NUMBER = "one axis number"
PER_TILE_AXIS = "one entry per tile axis"

PER_AXIS = {
    "dram_stride": PER_TILE_AXIS,
    "masked_axes": AXIS_NUMBERS,
    "dyn_stride_axes": AXIS_NUMBERS,
    "indirect_dim": AXIS_NUMBER,
    "subtile_size": PER_TILE_AXIS,
    "cut": PER_TILE_AXIS,
}
_unknown = set(PER_AXIS) - set(_TABLE.ATTRIBUTES[TransferOp.OPERATION_NAME])
if _unknown:
    raise ImportError(f"PER_AXIS names {sorted(_unknown)}, which the transfer's "
                      f"ODS does not declare")



def renumbered_per_axis(op, groups, shape):
    """`op`'s per-axis attributes restated for new axes that each stand for a group of old ones,
    without touching `op`: a per-tile entry is the group's last axis of extent > 1 (its first if
    none), as collapse_shape strides it; an axis named by number must be its group's only mover."""
    from mlir.ir import ArrayAttr, DenseI64ArrayAttr, IntegerAttr, IntegerType
    i64 = IntegerType.get_signless(64)
    rep = [next((d for d in reversed(g) if shape[d] != 1), g[0]) for g in groups]
    alone = {d: k for k, g in enumerate(groups) for d in g
             if all(shape[e] == 1 for e in g if e != d)}

    def axis(name, d):
        if d not in alone:
            raise NotImplementedError(
                f"{name} names tile axis {d}, which this rewrite peels, drops or folds with "
                f"another axis that moves, so no new axis is axis {d} alone: {op}")
        return alone[d]

    out = {}
    for name, kind in PER_AXIS.items():
        if name not in op.attributes:
            continue
        got = op.attributes[name]
        if kind == AXIS_NUMBERS:
            out[name] = ArrayAttr.get(
                [IntegerAttr.get(i64, axis(name, IntegerAttr(a).value)) for a in ArrayAttr(got)])
        elif kind == AXIS_NUMBER:
            out[name] = IntegerAttr.get(i64, axis(name, IntegerAttr(got).value))
        elif kind == PER_TILE_AXIS:
            try:
                vals = list(DenseI64ArrayAttr(got))
                out[name] = DenseI64ArrayAttr.get([vals[r] for r in rep])
            except (TypeError, ValueError):
                items = list(ArrayAttr(got))
                out[name] = ArrayAttr.get([items[r] for r in rep])
        else:
            raise RuntimeError(f"a per-axis kind nothing renumbers: {name}: {kind}")
    return out


def renumber_per_axis(op, groups, shape):
    """Write renumbered_per_axis(op, groups, shape) onto `op`."""
    for name, attr in renumbered_per_axis(op, groups, shape).items():
        op.attributes[name] = attr


DRAM_ARG_ROUND_TRIP = -1
DRAM_ARG_POINTER_TILE = -2

POINTER_TILE_BASE = 8
"""Where a DRAM_ARG_POINTER_TILE transfer counts its indices from. NOT 0: the DMA reads a
DRAM address of 0 as "outside the box, fill"; 8 is past it and a multiple of every width."""


register()
