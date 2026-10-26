"""convert-transfers-to-dma: every pytorchsim.wait and pytorchsim.transfer written as the target kind's DMA
(TargetKind.lower_wait, lower_transfers) where it stands, then dropped; timing=True drops the transfers
unwritten, for the one-tile binary gem5 times. A module global nothing reads after them goes too."""
from ...dialects.pytorchsim import TransferOp, WaitOp
from .target_kind import target_kind
from .lib_descriptor import TransferFacts
from .lib_ir import walk_ops
from . import pass_name

NAME = pass_name(__name__)


def run(module, timing=False, vectorlane=None, target=None, **_):
    """The waits first, then the transfers, each refused when legalize-transfers left it cut or
    wider than the kind's descriptor."""
    if vectorlane is None:
        raise ValueError(f"{NAME} needs the lane count: a gather's descriptor names how many lanes hold "
                         f"its index. Pass vectorlane=machine.VECTORLANE_SIZE.")
    kind = target_kind(target)
    top = module.operation.regions[0].blocks[0]
    for w in [o.operation for o in walk_ops(top) if o.operation.name == WaitOp.OPERATION_NAME]:
        kind.lower_wait(w)
        w.erase()
    transfers = [o.operation for o in walk_ops(top) if o.operation.name == TransferOp.OPERATION_NAME]
    for op in transfers:
        facts = TransferFacts(op)
        if facts.cut:
            raise RuntimeError(f"a cut transfer reached {NAME}: pytorchsim-legalize-transfers "
                               f"splits every cut into its pieces: {op}")
        if len(facts.tile_shape) > kind.DMA_DIMS:
            raise RuntimeError(
                f"a transfer of rank {len(facts.tile_shape)} reached {NAME}: "
                f"pytorchsim-legalize-transfers runs before it and leaves every uncut "
                f"transfer at rank <= {kind.DMA_DIMS}: {op}")
    if not timing:
        kind.lower_transfers(module, transfers, vectorlane)
    for op in transfers:
        op.erase()
    _erase_globals_nothing_reads(top)


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
