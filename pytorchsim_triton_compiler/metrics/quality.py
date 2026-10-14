"""How good the compile was, measured off the IR the compiler just produced."""
import json
import os

FILENAME = "08-quality.json"


def count(module):
    """What this compile moves, off the live module: DMA loads and stores, and the moves BETWEEN
    LANES by carrier -- round trips through a staging buffer (dram_arg -1) and the cross-lane unit's
    moves, by the kind of move each is."""
    from mlir.ir import IntegerAttr, StringAttr

    from ..dialects.torchsim import DRAM_ARG_ROUND_TRIP, is_load_kind
    from ..lowering.ext.vcix_tpuv6.lib_xlu import emitter_of
    from ..lowering.passes.lib_cross_lane import CROSS_LANE_ATTR, CROSS_LANE_OP
    from ..lowering.passes.lib_ir import walk

    got = {"mvin": 0, "mvout": 0, "round_trips": 0, "xlu": {}}
    for op in walk(module.operation):
        if op.name == CROSS_LANE_OP and CROSS_LANE_ATTR in op.attributes:
            kind = emitter_of(IntegerAttr(op.attributes[CROSS_LANE_ATTR]).value)
            got["xlu"][kind] = got["xlu"].get(kind, 0) + 1
            continue
        try:
            kind = StringAttr(op.attributes["dma_kind"]).value
        except (KeyError, IndexError, ValueError):
            continue
        load = is_load_kind(kind)
        got["mvin"] += load
        got["mvout"] += not load
        got["round_trips"] += (load and "dram_arg" in op.attributes
                               and IntegerAttr(op.attributes["dram_arg"]).value
                               == DRAM_ARG_ROUND_TRIP)
    return got


def _stamped_shape(op):
    from mlir.ir import ShapedType, TypeAttr
    try:
        if op.results:
            return list(ShapedType(op.results[0].type).shape)
        return list(ShapedType(TypeAttr(op.attributes["type"]).value).shape)
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def count_one_lane(module):
    """(one-lane buffers, those among them with an axis to spread) off the live module."""
    from ..lowering.passes.lib_bank_layout import lanes_of_a_buffer
    from ..lowering.passes.lib_ir import walk
    from mlir.ir import ShapedType, TypeAttr

    stamped = spreadable = 0
    for op in walk(module.operation):
        if lanes_of_a_buffer(op) != 1 or _stamped_shape(op) is None:
            continue
        stamped += 1
        ty = None
        if op.results:
            ty = op.results[0].type
        elif "type" in op.attributes:
            try:
                ty = TypeAttr(op.attributes["type"]).value
            except (ValueError, TypeError):
                ty = None
        try:
            shape = list(ShapedType(ty).shape)
        except (TypeError, ValueError):
            continue
        spreadable += any(d > 1 for d in shape)
    return stamped, spreadable


def write(workdir, counts, one_lane=None):
    """Record what `count` and `count_one_lane` measured, beside the IR."""
    if counts is None:
        return None
    got = dict(counts)
    if one_lane is not None:
        got["one_lane"], got["one_lane_spreadable"] = one_lane
    path = os.path.join(workdir, FILENAME)
    with open(path, "w") as fh:
        json.dump(got, fh, indent=2)
        fh.write("\n")
    return path


def read(workdir):
    """(mvin, mvout) as this compile recorded them, or (0, 0)."""
    got = read_all(workdir)
    return got.get("mvin", 0), got.get("mvout", 0)


def read_one_lane(workdir):
    """(one-lane buffers, those with something to spread), or (0, 0)."""
    got = read_all(workdir)
    return got.get("one_lane", 0), got.get("one_lane_spreadable", 0)


def read_all(workdir):
    """Everything this compile recorded about its own quality, or {}."""
    try:
        with open(os.path.join(workdir, FILENAME)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}
