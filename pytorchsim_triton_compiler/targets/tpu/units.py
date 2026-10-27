"""units: the lane-index unit (tpu::Misc vlane_idx, pytorchsim_tpu.misc.lane_idx), read by the
vector-functions and cross-lane lowerings; no SLOT, so not a pass."""
import mlir.ir as ir  # noqa: E402

from .pytorchsim_tpu import MiscLaneIdxOp


def lane_number(rvl, width):
    """Each lane's own number, `width` elements of it."""
    return MiscLaneIdxOp(ir.VectorType.get([width], ir.IntegerType.get_signless(64)), rvl).result
