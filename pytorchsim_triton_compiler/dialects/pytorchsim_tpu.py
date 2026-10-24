"""The `pytorchsim_tpu` dialect: the TPU family's own ops, made and lowered inside its pack
(lowering/ext/pytorchsim_tpu_family). Its op classes come from PyTorchSimTpuOps.td through the same build
and extension as `pytorchsim`; registering either registers both."""
from . import pytorchsim as _pytorchsim
from .pytorchsim import _load_generated

_load_generated("_pytorchsim_tpu_ops_gen")
from ._pytorchsim_tpu_ops_gen import *  # noqa: E402,F401,F403

register = _pytorchsim.register
