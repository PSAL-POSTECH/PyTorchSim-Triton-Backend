"""The `pytorchsim_tpu` dialect: the tpu target's own ops, made and lowered by targets/tpu/backend,
whose CMakeLists.txt registers them with the dialect build. Its op classes come from PyTorchSimTpuOps.td
through the same build and extension as `pytorchsim`; registering either registers both."""
from ...dialect import pytorchsim as _pytorchsim
from ...dialect.pytorchsim import _load_generated

_load_generated("_pytorchsim_tpu_ops_gen")
from ...dialect._pytorchsim_tpu_ops_gen import *  # noqa: E402,F401,F403
from ...dialect._pytorchsim_tpu_ops_gen import VpuPackedViewOp, _Dialect  # noqa: E402

register = _pytorchsim.register

#: The dialect's name, which keys the target's registration (its CMakeLists.txt) and its lane reduces.
DIALECT = _Dialect.DIALECT_NAMESPACE

#: A lane's bank read packed, as 32-bit words.
PACKED_VIEW = VpuPackedViewOp.OPERATION_NAME
