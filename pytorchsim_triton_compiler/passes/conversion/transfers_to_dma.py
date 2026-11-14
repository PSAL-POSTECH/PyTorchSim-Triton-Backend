"""convert-transfers-to-dma: every pytorchsim.wait and pytorchsim.transfer written as the target's DMA
(Target.lower_wait, lower_transfers) where it stands, then dropped; timing=True drops the transfers
unwritten, for the one-tile binary gem5 times. A module global nothing reads after them goes too."""
from ...dialect.pytorchsim import TransferOp, WaitOp, is_load
from ...targets import target
from ..utils.descriptor import TransferFacts
from ..utils.ir_utils import walk_ops
from .. import pass_name

NAME = pass_name(__name__)


def run(module, timing=False, vectorlane=None, subtarget=None, **_):
    """The waits first, then the transfers, each refused when legalize-transfers left it cut or
    wider than the target's descriptor."""
    if vectorlane is None:
        raise ValueError(f"{NAME} needs the lane count: a gather's descriptor names how many lanes hold "
                         f"its index. Pass vectorlane=machine.VECTORLANE_SIZE.")
    tgt = target(subtarget)
    top = module.operation.regions[0].blocks[0]
    for w in [o.operation for o in walk_ops(top) if o.operation.name == WaitOp.OPERATION_NAME]:
        tgt.lower_wait(w)
        w.erase()
    transfers = [o.operation for o in walk_ops(top) if o.operation.name == TransferOp.OPERATION_NAME]
    for op in transfers:
        facts = TransferFacts(op)
        if facts.cut:
            raise RuntimeError(f"a cut transfer reached {NAME}: pytorchsim-legalize-transfers "
                               f"splits every cut into its pieces: {op}")
        if len(facts.tile_shape) > tgt.DMA_DIMS:
            raise RuntimeError(
                f"a transfer of rank {len(facts.tile_shape)} reached {NAME}: "
                f"pytorchsim-legalize-transfers runs before it and leaves every uncut "
                f"transfer at rank <= {tgt.DMA_DIMS}: {op}")
    if not timing:
        tgt.lower_transfers(module, transfers, vectorlane)
    else:
        for op in transfers:
            if not is_load(op):
                _keep_stored(op)
    for op in transfers:
        op.erase()
    _erase_globals_nothing_reads(top)


def _keep_stored(op):
    """A dropped store transfer's buffer handed to an empty asm that reads memory, where the transfer stood: the
    stores into it, and what they store, stay in the timed binary as the DMA kept them in the kernel."""
    from mlir.dialects import arith, llvm, memref
    from mlir.ir import InsertionPoint, IntegerType
    with InsertionPoint(op):
        at = memref.ExtractAlignedPointerAsIndexOp(TransferFacts(op).sram).result
        llvm.InlineAsmOp(None, [arith.IndexCastOp(IntegerType.get_signless(64), at).result], "", "r,~{memory}",
                         has_side_effects=True, asm_dialect=0)


def _erase_globals_nothing_reads(top):
    from mlir.ir import FlatSymbolRefAttr, StringAttr
    for g in [o for o in walk_ops(top)
              if o.operation.name == "memref.get_global"
              and not list(o.operation.results[0].uses)]:
        g.operation.erase()
    read = {FlatSymbolRefAttr(o.operation.attributes["name"]).value for o in walk_ops(top)
            if o.operation.name == "memref.get_global"}
    for g in [o for o in walk_ops(top)
              if o.operation.name == "memref.global"
              and StringAttr(o.operation.attributes["sym_name"]).value not in read]:
        g.operation.erase()
