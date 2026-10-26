"""pytorchsim-lower-convert: every pytorchsim.convert_layout onto the carrier the target kind has for it
(TargetKind.lower_convert)."""
from ...dialects.pytorchsim import ConvertLayoutOp
from .target_kind import target_kind

FORBIDS = (ConvertLayoutOp.OPERATION_NAME,)


def run(module, **opts):
    return target_kind().lower_convert(module, **opts)
