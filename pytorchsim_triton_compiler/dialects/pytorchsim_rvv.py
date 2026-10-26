"""The `pytorchsim_rvv` dialect: the RVV target kind's own ops, made by the generic RVV lowering
(lowering/ext/rvv_kind) in any pack that adopts it. Its op classes come from PyTorchSimRvvOps.td through
the same build and extension as `pytorchsim`; registering either registers both."""
from . import pytorchsim as _pytorchsim
from .pytorchsim import _load_generated

_load_generated("_pytorchsim_rvv_ops_gen")
from ._pytorchsim_rvv_ops_gen import *  # noqa: E402,F401,F403
from ._pytorchsim_rvv_ops_gen import BankViewOp  # noqa: E402

register = _pytorchsim.register

#: A buffer as one lane's bank of it.
BANK_VIEW = BankViewOp.OPERATION_NAME
