"""Shared vocabulary for the skeleton+API MLIR form (C1)."""

DMA    = "torchsim.dma"
COMPUTE = "torchsim.compute"
MEMORY_BAR = "torchsim.memory_barrier"
ATTRIBUTE = "torchsim.attribute_read"

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
ATTR_SPAD_BUFFER_BYTES = "togsim.spad_buffer_bytes"
ATTR_RUNTIME_ATTRIBUTES = "togsim.runtime_attribute_ranks"

DIR_LOAD  = 0
DIR_STORE = 1


