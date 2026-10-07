"""cycle_table: the precomputed tile_id -> (cycle, overlapping_cycle) table the C++ trace pipeline
looks up at runtime (docs/design/togsim_cpp_trace.md sec 6)."""


from . import togsim_ops as ts
from ._mlir_util import walk_ops
from .build_tog import (
    ir,
    VECTOR_COMPUTE,
    MATMUL_COMPUTE,  # noqa: F401
    MATMUL_PRELOAD,
    CROSS_LANE_COMPUTE,
)


def overlap_offset(compute_type, lanes):
    """Cycles of a tile that the pipeline cannot hide, or None where none of it overlaps."""
    if compute_type in (VECTOR_COMPUTE, CROSS_LANE_COMPUTE):
        return None
    return 0 if compute_type == MATMUL_PRELOAD else lanes


def _compute_types(skeleton_module):
    items = []
    for op in walk_ops(skeleton_module.body):
        if op.operation.name != ts.COMPUTE:
            continue
        tid = ir.IntegerAttr(op.operation.attributes[ts.ATTR_TILE_ID]).value
        ct = ir.IntegerAttr(op.operation.attributes[ts.ATTR_COMPUTE_TYPE]).value
        items.append((tid, ct))
    items.sort()
    return [t for _, t in items]

