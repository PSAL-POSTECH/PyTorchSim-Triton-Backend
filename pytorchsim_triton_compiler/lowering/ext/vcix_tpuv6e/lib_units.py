"""lib_units: vcix_tpuv6e's lane-index unit (tpu::Misc vlane_idx, vcix.v.i opcode 0), read by the
vector-functions and cross-lane lowerings; no SLOT, so not a pass."""
from ...passes.lib_vcix import i64, vcix

import mlir.ir as ir  # noqa: E402


def lane_number(rvl, width):
    """Each lane's own number, `width` elements of it: the VCIX op of opcode 0."""
    i64_ty = ir.IntegerType.get_signless(64)
    return vcix("vcix.v.i", [rvl], [ir.VectorType.get([width], i64_ty)],
                {"opcode": i64(0), "rs2": i64(0), "imm": i64(0)}).results[0]
