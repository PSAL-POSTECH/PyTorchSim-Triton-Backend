"""cycle_table: the precomputed tile_id -> (cycle, overlapping_cycle) table TOGSim looks
up as it replays a TOG."""


from . import togsim_ops as ts
from ._mlir_util import walk_ops
from .build_tog import ir


def overlap_offset(compute_type, lanes):
    """DEPRECATED column, kept for the TOG ABI: None (overlap 0) for every compute type. gem5 issues
    a core's computes in order and a pop waits for its row; TOGSim serializes them itself."""
    return None


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

