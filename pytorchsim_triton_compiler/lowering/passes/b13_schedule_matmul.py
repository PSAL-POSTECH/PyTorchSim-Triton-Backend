"""pytorchsim-schedule-matmul: where the matrix unit's ops are issued (TargetKind.schedule_matmul)."""
from .target_kind import target_kind


def run(module, **opts):
    return target_kind().schedule_matmul(module, **opts)
