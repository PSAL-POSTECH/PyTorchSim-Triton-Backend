"""pytorchsim-schedule-vpu: where the vector loops' unit ops are placed once their functions are lowered,
pipelined across trips (TargetKind.schedule_vpu)."""
from .target_kind import target_kind


def run(module, **opts):
    return target_kind().schedule_vpu(module, **opts)
