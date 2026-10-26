"""The tpu target kind (contract.target.ACCELERATORS): KIND, the TPU's units each in a module of this
package; to_vcix encodes their ops as VCIX instructions."""
from ....contract import copy_node
from ....dialects import pytorchsim_tpu
from ...passes.target_kind import TargetKind
from . import (convert_pytorchsim_to_pytorchsim_tpu, lib_xlu, lower_convert, lower_dma,
               lower_vector_functions, schedule_mxu_stream, schedule_vpu_stream, to_vcix)


class TpuKind(TargetKind):
    """The TPU: the 20-slot descriptor DMA, the XLU's moves, the MXU, SFU, lane-index unit and XLU rules, the
    MXU's and EUP's schedules, and XLA's one resident operand (contract/copy_node)."""

    DMA_DIMS = lower_dma.DMA_DIMS
    INDEX_KEY = lower_dma.INDEX_KEY
    TO_VCIX = to_vcix.TO_VCIX
    BANK_VIEW = pytorchsim_tpu.BANK_VIEW
    CROSS_LANE_FOLDS = lib_xlu.XLU_FOLDS
    CROSS_LANE_FOLD_TYPE = lib_xlu.XLU_FOLD_TYPE

    def lower_transfers(self, module, transfers, vectorlane):
        lower_dma.lower_transfers(module, transfers, vectorlane)

    def insn(self, func7, rs1, rs2):
        return lower_dma.insn(func7, rs1, rs2)

    def walk(self, op, vectorlane):
        return lower_dma.walk(op, vectorlane)

    def sram_footprints(self, op, vectorlane, interval_of):
        return lower_dma.sram_footprints(op, vectorlane, interval_of)

    def lower_convert(self, module, **opts):
        return lower_convert.run(module, **opts)

    def convert_to_target_kind(self, module, **opts):
        return convert_pytorchsim_to_pytorchsim_tpu.run(module, **opts)

    def max_lmul(self, loop):
        return lower_vector_functions.max_lmul(loop)

    def schedule_matmul(self, module, **opts):
        return schedule_mxu_stream.run(module, **opts)

    def schedule_vpu(self, module, **opts):
        return schedule_vpu_stream.run(module, **opts)

    def prefetch_order(self, candidates):
        return copy_node.prefetch_order(candidates)


KIND = TpuKind()
