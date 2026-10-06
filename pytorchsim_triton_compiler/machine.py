"""The machine this compile targets, and the numbers that follow from it."""
import os

from .contract import target as _target

_LAZY = {}


def _get(name):
    """A machine number, read on first use. See __getattr__ at the end."""
    if name not in _LAZY:
        _LAZY[name] = globals()["_lazy_" + name]()
    return _LAZY[name]


def target():
    """The machine, read once per process. ONE door on purpose: no `machine.TARGET`."""
    if "TARGET" not in _LAZY:
        _LAZY["TARGET"] = _read_the_target()
    return _LAZY["TARGET"]


def _read_the_target():
    """What TORCHSIM_COMPILE_TARGET names, parsed AND checked by the contract -- so every
    number below is one a valid description stated."""
    said = os.environ.get("TORCHSIM_COMPILE_TARGET")
    if not said:
        raise _target.TargetError(
                "TORCHSIM_COMPILE_TARGET is not set, and there is no default machine. A "
                "target description states the scratchpad window and the DRAM "
                "range as well as the lane count, so half of it cannot be "
                "supplied from here -- whoever owns the machine hands over the "
                "whole of it. PyTorchSim writes one from its TOGSim YAML; the "
                "suite's is tests/targets/tpuv6e.json, which "
                "`source /workspace/torchsim-env.sh` points at. "
                "tools/emit_target.py writes one from a TOGSim YAML.")
    return _target.load(said)


# RVV's OWN ceiling on one register group, not this target's: no description states it,
# and pipeline prices every vector against it.
ISA_MAX_LMUL = 8


def _lazy_VECTORLANE_SIZE():
    return target().vector_lanes


def _lazy_VLEN_BITS():
    return target().vlen_bits


def _lazy_SPAD_SIZE():
    return target().spad_size


def _lazy_SPAD_VADDR():
    return target().spad_vaddr


def _lazy_SPAD_PADDR():
    return target().spad_paddr


def banked_bytes():
    """Size of the address window Spike redirects to the scratchpad, the target's own sum."""
    return target().banked_bytes()


def check_memory_map():
    """The target's own verdict on its scratchpad/DRAM layout."""
    return target().memory_map_problems()


def __getattr__(name):
    if "_lazy_" + name not in globals():
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _get(name)


def __dir__():
    return sorted(list(globals()) + [k[len("_lazy_"):] for k in globals()
                                     if k.startswith("_lazy_")])
