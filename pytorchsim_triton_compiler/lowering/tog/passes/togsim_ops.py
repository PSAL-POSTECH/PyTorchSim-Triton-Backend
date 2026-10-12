"""Shared vocabulary for the skeleton+API MLIR form (C1)."""

DMA    = "torchsim.dma"
COMPUTE = "torchsim.compute"
MEMORY_BAR = "torchsim.memory_barrier"

OP_NAMES = (DMA, COMPUTE, MEMORY_BAR)

EMITC_CALLEE = {
    DMA:     "togsim_dma",
    COMPUTE: "togsim_compute",
    MEMORY_BAR: "togsim_memory_barrier",
}

ENTRY_SYMBOL = "togsim_kernel"

SPAD_BUFFER_COUNT_SYMBOL = "togsim_spad_buffer_count"
SPAD_BUFFER_BYTES_SYMBOL = "togsim_spad_buffer_bytes"

TILE_SYMBOL = "togsim_kernel_tile"

DISPATCH_CALLEE = "togsim_dispatch"

ATTR_DIR       = "dir"
ATTR_DIMS      = "dims"
ATTR_DYN_DIMS  = "dyn_dims"
ATTR_STRIDES   = "strides"
ATTR_ELEM_BITS = "elem_bits"
ATTR_IS_ASYNC  = "is_async"
ATTR_TILE_ID   = "tile_id"
ATTR_COMPUTE_TYPE = "compute_type"
ATTR_READ_BUFS  = "read_bufs"
ATTR_WRITE_BUFS = "write_bufs"
ATTR_EXPECTED  = "expected"
ATTR_ARG_ID    = "arg_id"
ATTR_INDIRECT  = "indirect"
ATTR_SPAD_BUFFER_BYTES = "togsim.spad_buffer_bytes"

DIR_LOAD  = 0
DIR_STORE = 1


def is_togsim_op(op):
    """True if `op` (an Operation or a wrapping view) is one of ours."""
    name = getattr(op, "name", None)
    if name is None:
        name = getattr(getattr(op, "operation", None), "name", None)
    return name in OP_NAMES
