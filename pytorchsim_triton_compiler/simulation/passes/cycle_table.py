"""cycle_table: the precomputed tile_id -> (cycle, overlapping_cycle) table the C++ trace
pipeline looks up at runtime (docs/design/togsim_cpp_trace.md sec 6)."""

import json

from . import togsim_ops as ts
from ._mlir_util import walk_ops
from .build_tog import (
    ir,
    VECTOR_COMPUTE,
    MATMUL_COMPUTE,  # noqa: F401
    MATMUL_PRELOAD,
    CROSS_LANE_COMPUTE,
)


def overlapping_cycle(cycle, compute_type, x_offset, w_offset):
    """Hideable (pipeline-overlapped) portion of `cycle`. Mirrors
    tog_generator.generate_tile_graph."""
    if compute_type in (VECTOR_COMPUTE, CROSS_LANE_COMPUTE):
        return 0
    offset = w_offset if compute_type == MATMUL_PRELOAD else x_offset
    return max(int(cycle) - int(offset), 0)


def _compute_types(skeleton_module):
    """tile_id-ordered list of compute_type ints, from the skeleton's
    torchsim.compute ops."""
    items = []
    for op in walk_ops(skeleton_module.body):
        if op.operation.name != ts.COMPUTE:
            continue
        tid = ir.IntegerAttr(op.operation.attributes[ts.ATTR_TILE_ID]).value
        ct = ir.IntegerAttr(op.operation.attributes[ts.ATTR_COMPUTE_TYPE]).value
        items.append((tid, ct))
    items.sort()
    return [t for _, t in items]


def build_cycle_table(skeleton_module, cycle_list, x_offset, w_offset):
    """Return `[(cycle, overlapping_cycle), ...]` indexed by tile_id."""
    types = _compute_types(skeleton_module)
    if len(cycle_list) != len(types):
        raise ValueError(
            "cycle_list (%d) does not match #compute tiles (%d)"
            % (len(cycle_list), len(types)))
    return [(int(c), overlapping_cycle(c, t, x_offset, w_offset))
            for c, t in zip(cycle_list, types)]


def dump_cycle_table(table, path, x_offset=None, w_offset=None):
    """Serialize the table as a sidecar JSON next to the trace `.so`. The P3 C6
    loader reads it and sets compute_cycle + overlapping_cycle on each emitted
    Instruction."""
    with open(path, "w") as fh:
        json.dump({"x_offset": x_offset, "w_offset": w_offset,
                   "table": [list(e) for e in table]}, fh)
    return path


def load_cycle_table(path):
    with open(path) as fh:
        return json.load(fh)


def dump_cycle_table_tsv(table, path, origins=None):
    """Plain `cycle<TAB>overlapping` per line, in tile_id order -- the trivial format the C++
    `--cycle_table` loader (main.cc, P3 trace pipeline) reads with ifstream (no JSON
    dependency in TOGSim)."""
    with open(path, "w") as fh:
        for cycle, overlapping in table:
            fh.write("%d\t%d\n" % (int(cycle), int(overlapping)))
        if origins:
            fh.write("# origins: %s\n" % ", ".join(sorted(str(o) for o in origins)))
    return path
