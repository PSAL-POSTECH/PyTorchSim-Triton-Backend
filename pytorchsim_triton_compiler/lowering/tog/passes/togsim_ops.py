"""Shared vocabulary for the skeleton+API MLIR form (C1): the `togsim` dialect's ops (dialects/togsim.py)."""

DMA    = "togsim.dma"
COMPUTE = "togsim.compute"
MEMORY_BAR = "togsim.memory_barrier"
ATTRIBUTE = "togsim.attribute_read"

OP_NAMES = (DMA, COMPUTE, MEMORY_BAR)

ATTR_DIR       = "dir"
ATTR_DIMS      = "dims"
ATTR_DYN_DIMS  = "dyn_dims"
ATTR_STRIDES   = "strides"
ATTR_ELEM      = "elem"
ATTR_IS_ASYNC  = "is_async"
ATTR_TILE_ID   = "tile_id"
ATTR_COMPUTE_TYPE = "compute_type"
ATTR_READ_BUFS  = "read_bufs"
ATTR_WRITE_BUFS = "write_bufs"
ATTR_ARG_ID    = "arg_id"
ATTR_INDIRECT  = "indirect"
ATTR_GATHER    = "gather"
ATTR_BUFFER_COUNT = "togsim.buffer_count"
ATTR_RUNTIME_ATTRIBUTES = "togsim.runtime_attribute_ranks"
ATTR_HELD_BUFFERS = "togsim.held_buffers"

DIR_LOAD  = 0
DIR_STORE = 1


def is_togsim_op(op):
    """True if `op` (an Operation or a wrapping view) is one of ours."""
    name = getattr(op, "name", None)
    if name is None:
        name = getattr(getattr(op, "operation", None), "name", None)
    return name in OP_NAMES
