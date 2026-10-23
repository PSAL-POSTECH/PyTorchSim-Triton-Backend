"""The machine this compile targets, read once per process; its numbers are the Target's own fields."""
import os

from .contract import target as _target

_LAZY = {}


def target():
    """The machine, read once per process."""
    if "TARGET" not in _LAZY:
        _LAZY["TARGET"] = _read_the_target()
    return _LAZY["TARGET"]


def _read_the_target():
    said = os.environ.get("TORCHSIM_COMPILE_TARGET")
    if not said:
        raise _target.TargetError(
                "TORCHSIM_COMPILE_TARGET is not set, and there is no default machine. A "
                "target description states the scratchpad window and the DRAM "
                "range as well as the lane count, so half of it cannot be "
                "supplied from here -- whoever owns the machine hands over the "
                "whole of it. Name a machine file under $TOGSIM_DIR/configs (tpuv6e) "
                "or give a path.")
    return _target.load(said)


# The largest register group RVV's vtype can name (LMUL 8).
ISA_MAX_LMUL = 8

# The vector registers RVV names (v0..v31), every one VLEN bits.
VECTOR_REGISTERS = 32
