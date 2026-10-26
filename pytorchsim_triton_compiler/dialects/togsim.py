"""The `togsim` dialect: the tile-operation graph's ops, TOGSim API calls one each (lowering/tog builds
them from a lowered kernel, lower_to_emitc writes them as C++). Its op classes come from TogsimOps.td
through the same build and extension as `pytorchsim`; registering either registers both."""
from . import pytorchsim as _pytorchsim
from .pytorchsim import _load_generated

_load_generated("_togsim_ops_gen")
from ._togsim_ops_gen import *  # noqa: E402,F401,F403

register = _pytorchsim.register
