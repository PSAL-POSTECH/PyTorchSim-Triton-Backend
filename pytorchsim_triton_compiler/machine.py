"""The machine this compile targets, and the numbers that follow from it."""
import os

from .contract import target as _target

_LAZY = {}


def _get(name):
    if name not in _LAZY:
        _LAZY[name] = globals()["_lazy_" + name]()
    return _LAZY[name]


def target():
    """The machine, read once per process."""
    if "TARGET" not in _LAZY:
        _LAZY["TARGET"] = _read_the_target()
    return _LAZY["TARGET"]


def _read_the_target():
    said = os.environ.get("PYTORCHSIM_COMPILE_TARGET")
    if not said:
        raise _target.TargetError(
                "PYTORCHSIM_COMPILE_TARGET is not set, and there is no default machine. A "
                "target description states the scratchpad window and the DRAM "
                "range as well as the lane count, so half of it cannot be "
                "supplied from here -- whoever owns the machine hands over the "
                "whole of it. Name a machine file under $TOGSIM_DIR/configs (tpuv6e) "
                "or give a path.")
    return _target.load(said)


def _lazy_ISA_MAX_LMUL():
    """The widest register group a vector loop may take (vpu_max_lmul, RVV's 8 unsaid)."""
    return target().max_lmul


def _lazy_VECTOR_REGISTERS():
    """The vector registers the core has, every one VLEN bits (vpu_vector_registers, RVV's 32 unsaid)."""
    return target().vector_registers


def _lazy_VECTORLANE_SIZE():
    return target().vector_lanes


def _lazy_VLEN_BITS():
    return target().vlen_bits


def _lazy_SPAD_SIZE():
    return target().spad_size


def _lazy_SPAD_VADDR():
    return target().spad_vaddr


def banked_bytes():
    """Size of the address window Spike redirects to the scratchpad, the target's own sum."""
    return target().banked_bytes()


def __getattr__(name):
    if "_lazy_" + name not in globals():
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _get(name)


def __dir__():
    return sorted(list(globals()) + [k[len("_lazy_"):] for k in globals()
                                     if k.startswith("_lazy_")])
