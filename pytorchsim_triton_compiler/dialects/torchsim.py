"""The `torchsim` dialect: the python extension it lives in, the op classes ODS generates
from the same .td, and what ODS cannot state -- which of a transfer's attributes speak per
axis, and the `dram_arg` values that name no argument."""
import glob
import hashlib
import importlib.util
import os
import sys

BUILD = os.path.join(os.environ.get("TORCHSIM_PREFIX", "/workspace"), "torchsim-dialect-build")

_done = [False]


def _load_from_build(name, pattern):
    """A module the dialect build produced, loaded under our package's name so the
    generated code's relative imports land on _ods_common. Refusing names the rebuild,
    because a missing file here means that build never ran this far."""
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
    """One of the tablegen-generated python modules, by its plain name."""
    return _load_from_build(name, name + ".py")


def _refuse_a_stale_build():
    """The bindings are tablegen output, so a build left behind by an ODS change offers
    op classes nobody wrote. The stamp says which .td made them; this checkout says
    which one we mean. No .td here (an installed package) means no claim to check."""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "dialect", "include", "TorchSim")
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
    """Put `torchsim` in the bindings' GLOBAL registry, once, and let `.opview`
    find the classes above. UPSTREAM'S HOOKS: every Context starts from that
    registry, and the search prefix is where the bindings look for a dialect."""
    if _done[0]:
        return
    from mlir._mlir_libs import _mlir, get_dialect_registry
    ext = _load_from_build("_torchsim_ext", "_torchsim_ext*.so")
    ext.insert_into_registry(get_dialect_registry())
    _mlir.globals.append_dialect_search_prefix(__package__)
    _done[0] = True


def operand_number(op, name, k=0):
    """Where the `k`-th operand of ODS argument `name` sits among `op`'s operands:
    the .td's order, and the op's own `operandSegmentSizes` where it has one."""
    from mlir.ir import DenseI32ArrayAttr
    op = getattr(op, "operation", op)
    i = _TABLE.OPERANDS[op.name].index(name)
    if "operandSegmentSizes" not in op.attributes:
        return i + k
    sizes = list(DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))
    return sum(sizes[:i]) + k


def stated_attributes(op_name):
    """Every attribute ODS lets `op_name` state of itself, with the segment sizes
    a variadic op carries -- what is NOT one of these was put there by a pass."""
    return frozenset(_TABLE.ATTRIBUTES[op_name]) | {"operandSegmentSizes"}


AXIS_NUMBERS = "a list of axis numbers"
AXIS_NUMBER = "one axis number"
PER_TILE_AXIS = "one entry per tile axis"

PER_AXIS = {
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

DRAM_ARG_ROUND_TRIP = -1
DRAM_ARG_POINTER_TILE = -2

POINTER_TILE_BASE = 8
"""Where a DRAM_ARG_POINTER_TILE transfer counts its indices from. NOT 0: the DMA reads a
DRAM address of 0 as "outside the box, fill"; 8 is past it and a multiple of every width."""
