"""How good the compile was, measured off the IR the compiler just produced."""
import json
import os

FILENAME = "08-quality.json"


def count(module):
    """What this compile moves, off the live module: DMA loads and stores, and the moves the pack's
    units make -- the neutral DISTRIBUTED_REDUCTION_OP and every op of the pack's own dialect, by op name."""
    from mlir.ir import StringAttr

    from ..dialects.pytorchsim import is_load_kind, is_target_op
    from ..lowering.passes.lib_bank_layout import DISTRIBUTED_REDUCTION_OP
    from ..lowering.passes.lib_ir import walk

    got = {"loads": 0, "stores": 0, "moves": {}}
    for op in walk(module.operation):
        if op.name == DISTRIBUTED_REDUCTION_OP or is_target_op(op):
            got["moves"][op.name] = got["moves"].get(op.name, 0) + 1
            continue
        try:
            kind = StringAttr(op.attributes["direction"]).value
        except (KeyError, IndexError, ValueError):
            continue
        load = is_load_kind(kind)
        got["loads"] += load
        got["stores"] += not load
    return got


def _stamped_shape(op):
    from mlir.ir import ShapedType, TypeAttr
    try:
        if op.results:
            return list(ShapedType(op.results[0].type).shape)
        return list(ShapedType(TypeAttr(op.attributes["type"]).value).shape)
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def count_uniform(module):
    """(UNIFORM buffers -- one bank, the same in every lane --, those among them with an axis that
    could spread) off the live module: information about placement, not a cost."""
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


def write(workdir, counts, uniform=None):
    """Record what `count` and `count_uniform` measured, beside the IR."""
    if counts is None:
        return None
    got = dict(counts)
    if uniform is not None:
        got["uniform"], got["uniform_spreadable"] = uniform
    path = os.path.join(workdir, FILENAME)
    with open(path, "w") as fh:
        json.dump(got, fh, indent=2)
        fh.write("\n")
    return path


def read(workdir):
    """(loads, stores) as this compile recorded them, or (0, 0)."""
    got = read_all(workdir)
    return got.get("loads", 0), got.get("stores", 0)


def read_uniform(workdir):
    """(UNIFORM buffers, those with something to spread), or (0, 0)."""
    got = read_all(workdir)
    return got.get("uniform", 0), got.get("uniform_spreadable", 0)


def read_all(workdir):
    """Everything this compile recorded about its own quality, or {}."""
    try:
        with open(os.path.join(workdir, FILENAME)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}
