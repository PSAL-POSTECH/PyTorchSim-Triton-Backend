"""The devices a kernel object can be run on.

Each reads the object's manifest and nothing else of the compiler's, which is
what lets a new one be written without opening a pass.
"""
from .base import Device, DeviceError, Launch          # noqa: F401
from .gem5 import Gem5Device                            # noqa: F401
from .spike import SpikeDevice                          # noqa: F401
from .togsim import TOGSimDevice                        # noqa: F401

#: Every device this build can reach, by the name a caller asks for.
REGISTRY = {d.name: d for d in (SpikeDevice(), Gem5Device(), TOGSimDevice())}


def get(name):
    """The device called `name`."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise DeviceError(
            f"no device named {name!r}. Have: {', '.join(sorted(REGISTRY))}.")
