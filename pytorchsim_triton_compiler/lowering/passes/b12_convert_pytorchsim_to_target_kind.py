"""convert-pytorchsim-to-target-kind: every pytorchsim op the target kind computes on its own units, written
as its ops (TargetKind.convert_to_target_kind): the end of layer B, none of them left after it."""
from ...dialects.pytorchsim import GatherOp, LaneIdOp
from . import lib_matmul
from .lib_bank_layout import DISTRIBUTED_REDUCTION_OP
from .target_kind import target_kind

FORBIDS = lib_matmul.MATMUL_OPS + (LaneIdOp.OPERATION_NAME, DISTRIBUTED_REDUCTION_OP, GatherOp.OPERATION_NAME)


def run(module, **opts):
    return target_kind().convert_to_target_kind(module, **opts)
