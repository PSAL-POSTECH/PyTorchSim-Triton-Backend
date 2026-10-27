"""The machine this compile targets, and the numbers that follow from it."""
import os

from ..contract import target as _target

_LAZY = {}


def _get(name):
    if name not in _LAZY:
        _LAZY[name] = globals()["_lazy_" + name]()
    return _LAZY[name]


def subtarget():
    """The machine, read once per process."""
    if "SUBTARGET" not in _LAZY:
        _LAZY["SUBTARGET"] = _read_the_subtarget()
    return _LAZY["SUBTARGET"]


def _read_the_subtarget():
    said = os.environ.get("PYTORCHSIM_COMPILE_TARGET")
    if not said:
        raise _target.TargetError(
                "PYTORCHSIM_COMPILE_TARGET is not set, and there is no default machine. A "
                "subtarget description states the scratchpad window and the DRAM "
                "range as well as the lane count, so half of it cannot be "
                "supplied from here -- whoever owns the machine hands over the "
                "whole of it. Name a machine file under $TOGSIM_DIR/configs (tpuv6e) "
                "or give a path.")
    return _target.load(said)


def _lazy_ISA_MAX_LMUL():
    """The widest register group a vector loop may take (vpu_max_lmul, RVV's 8 unsaid)."""
    return subtarget().max_lmul


def _lazy_VECTOR_REGISTERS():
    """The vector registers the core has, every one VLEN bits (vpu_vector_registers, RVV's 32 unsaid)."""
    return subtarget().vector_registers


def _lazy_VECTORLANE_SIZE():
    return subtarget().vector_lanes


def _lazy_VLEN_BITS():
    return subtarget().vlen_bits


def _lazy_SPAD_SIZE():
    return subtarget().spad_size


def _lazy_SPAD_VADDR():
    return subtarget().spad_vaddr




def __getattr__(name):
    if "_lazy_" + name not in globals():
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _get(name)


def __dir__():
    return sorted(list(globals()) + [k[len("_lazy_"):] for k in globals()
                                     if k.startswith("_lazy_")])
