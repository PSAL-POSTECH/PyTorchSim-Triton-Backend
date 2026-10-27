"""The tpu target's Target, TpuTarget (targets.target() makes it): the TPU's units each in a module of this
package; to_vcix encodes their ops as VCIX instructions."""
from ...contract import copy_node
from . import pytorchsim_tpu
from ..compiler import Target
from . import (
    convert_pytorchsim_to_pytorchsim_tpu, xlu, lower_convert, lower_dma, lower_vector_functions,
    schedule_mxu_stream, schedule_vpu_stream, to_vcix)


class TpuTarget(Target):
    """The TPU: the 20-slot descriptor DMA, the XLU's moves, the MXU, SFU, lane-index unit and XLU rules, the
    MXU's and EUP's schedules, and the resident operands' order (contract/copy_node)."""

    TRANSFORM_PASSES = (lower_convert,)
    LOWER_PASSES = (convert_pytorchsim_to_pytorchsim_tpu, schedule_mxu_stream, schedule_vpu_stream)

    DMA_DIMS = lower_dma.DMA_DIMS
    INDEX_KEY = lower_dma.INDEX_KEY
    TO_VCIX = to_vcix.TO_VCIX
    DIALECT = pytorchsim_tpu.DIALECT

    def lower_transfers(self, module, transfers, vectorlane):
        lower_dma.lower_transfers(module, transfers, vectorlane)

    def insn(self, func7, rs1, rs2):
        return lower_dma.insn(func7, rs1, rs2)

    def walk(self, op, vectorlane):
        return lower_dma.walk(op, vectorlane)

    def sram_footprints(self, op, vectorlane, interval_of):
        return lower_dma.sram_footprints(op, vectorlane, interval_of)

    def max_lmul(self, loop):
        return lower_vector_functions.max_lmul(loop)

    def prefetch_order(self, candidates):
        return copy_node.prefetch_order(candidates)

