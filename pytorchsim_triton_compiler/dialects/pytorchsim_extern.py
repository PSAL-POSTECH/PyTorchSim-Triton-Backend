"""The `pytorchsim_extern` dialect: the ops a kernel writes itself through tl.extra.torchsim
(pytorchsim_extern.vcix), lowered by its target's pack. Its op classes come from PyTorchSimExternOps.td
through the same build and extension as `pytorchsim`; registering either registers both."""
from . import pytorchsim as _pytorchsim
from .pytorchsim import _load_generated

_load_generated("_pytorchsim_extern_ops_gen")
from ._pytorchsim_extern_ops_gen import *  # noqa: E402,F401,F403

register = _pytorchsim.register
