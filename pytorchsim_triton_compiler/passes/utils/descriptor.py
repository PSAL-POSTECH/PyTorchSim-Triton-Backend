"""descriptor: what a pytorchsim.transfer states that a DMA descriptor has to fit, read once for
the pass that fits it (legalize-transfers) and the pass that writes it (convert-transfers-to-dma)."""

from ...dialect.pytorchsim import cut_offset, tile_shape
from .bank_layout import bank_strides, in_scratchpad, is_uniform, why_no_lane_axis
from .ir_utils import elem_bytes, int_array


class TransferFacts:
    """A transfer's operands and per-axis attributes, as the descriptor will need them."""

    def __init__(self, op):
        from mlir.ir import IntegerAttr, MemRefType
        from ...dialect.pytorchsim import DRAM_ARG_POINTER_TILE
        self.op = op
        tx = op.opview
        self.dram, self.dram_idx = tx.dram, tx.dram_idx
        self.sram, self.sram_idx = tx.sram, tx.sram_idx
        self.offset_buf = tx.index
        self.indirect = self.offset_buf is not None
        if not in_scratchpad(self.sram.type):
            raise RuntimeError(
                f"the transfer's scratchpad side has no recorded lane axis, and "
                f"this pass does not decide layouts: {why_no_lane_axis(self.sram)}: {op}")
        self.uniform = is_uniform(self.sram)
        self.dram_bound = tx.dram_bound
        self.pointer_tile = (IntegerAttr(tx.dram_arg).value
                             == DRAM_ARG_POINTER_TILE)
        self.cut = read_cut(op)
        self.piece_box = read_piece_box(op)
        self.gather_axis = (None if tx.indirect_dim is None
                            else IntegerAttr(tx.indirect_dim).value)
        self.dram_stride = int_array(tx.dram_stride)
        self.subtile = (None if tx.subtile_size is None
                        else int_array(tx.subtile_size))
        masked = [] if tx.masked_axes is None else int_array(tx.masked_axes)
        clamp = list(tx.clamp)
        self.masked_pairs = [(a, clamp[2 * i], clamp[2 * i + 1])
                             for i, a in enumerate(masked)]
        self.masked_fill = (0 if tx.masked_axes is None
                            else IntegerAttr(tx.masked_fill).value)
        self.dyn_fill = tx.dyn_fill
        strided = ([] if tx.dyn_stride_axes is None
                   else int_array(tx.dyn_stride_axes))
        self.dyn_strides = list(zip(strided, tx.dyn_strides))
        self.dyn_ostride = tx.dyn_offset_stride
        self.sram_ty = MemRefType(self.sram.type)
        self.elem = self.sram_ty.element_type
        self.space = self.sram_ty.memory_space
        self.bytes = elem_bytes(self.elem)
        self.dram_ty = MemRefType(self.dram.type)
        self.spad_rank = self.sram_ty.rank
        self.tile_shape = tile_shape(self.sram.type)
        self.tile_stride = bank_strides(self.sram) if self.spad_rank else [1]
        self.eff = [i for i, e in enumerate(self.tile_shape) if e > 1]


def read_piece_box(op):
    """The transfer's `piece_box`, as {tile axis: (lo, hi)}; None when it moves its whole tile."""
    box = op.opview.piece_box
    if box is None:
        return None
    out = {d: tuple(int_array(a)) for d, a in enumerate(box) if len(int_array(a))}
    return out or None


def read_cut(op):
    """The transfer's `cut`, as {tile axis: its Digits, outer to inner}; None when no axis is cut."""
    from ...dialect.pytorchsim import cut_digits
    cut = op.opview.cut
    if cut is None:
        return None
    out = {d: cut_digits(int_array(a)) for d, a in enumerate(cut) if len(int_array(a))}
    return out or None


def runs_of(digits, n, stride):
    """(start, count, DRAM delta) of each run an axis of extent `n` cut into `digits` splits into --
    one per value of its outer digits, split again where the inner one wraps -- the DRAM a descriptor
    walking the whole axis at `stride` from dram_idx + delta reads at the run's coordinates."""
    inner = digits[-1]
    block = inner.radix or n
    lap = (lambda j: (inner.phase + j * inner.stride) // inner.modulus) if inner.modulus else (
        lambda j: 0)
    runs = []
    for s in range(0, n, block):
        count, p0 = min(block, n - s), 0
        for p in range(1, count + 1):
            if p == count or lap(p) != lap(p0):
                runs.append((s + p0, p - p0, cut_offset(digits, s + p0) - (s + p0) * stride))
                p0 = p
    return runs
