"""The `pytorchsim_tpu` dialect: the TPU family's own ops, made and lowered inside its pack
(lowering/ext/tpu_kind). Its op classes come from PyTorchSimTpuOps.td through the same build
and extension as `pytorchsim`; registering either registers both."""
from . import pytorchsim as _pytorchsim
from .pytorchsim import _load_generated

_load_generated("_pytorchsim_tpu_ops_gen")
from ._pytorchsim_tpu_ops_gen import *  # noqa: E402,F401,F403
from ._pytorchsim_tpu_ops_gen import VpuBankViewOp, VpuBankWordsOp  # noqa: E402

register = _pytorchsim.register

#: A buffer as one lane's bank of it, and a bank read as 32-bit words.
BANK_VIEW = VpuBankViewOp.OPERATION_NAME
BANK_WORDS = VpuBankWordsOp.OPERATION_NAME
