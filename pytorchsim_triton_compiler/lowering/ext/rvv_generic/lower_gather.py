"""lower_gather (rvv_generic): a `pytorchsim.gather` along the axis inside each bank, on RVV alone
(lib_gather_in_bank's vrgather). One along the lanes crosses banks and has no carrier here yet: the
shared-scratchpad round trip comes with address space 2."""

from ...passes.lib_gather_in_bank import lower_gather_in_bank
from ...passes.lib_ir import walk
from ...passes.lower import LOWER_GATHER

import mlir.ir as ir  # noqa: E402

NAME = "pytorchsim-lower-gather-generic"

SLOT = LOWER_GATHER.name

FORBIDS = LOWER_GATHER.lowers

MARKERS = FORBIDS


def run(module, *, target, lmul=None, **_):
    """Every gather inside the banks lowered where it stands; one along the lanes is refused."""
    from ....machine import ISA_MAX_LMUL
    lmul = ISA_MAX_LMUL if lmul is None else lmul
    for op in [o for o in walk(module.operation) if o.name == "pytorchsim.gather"]:
        if ir.IntegerAttr(op.attributes["axis"]).value == 0:
            raise NotImplementedError(
                f"a pytorchsim.gather along the lanes crosses banks, and rvv_generic has no carrier for "
                f"that yet (the shared-scratchpad round trip comes with address space 2): {op}")
        lower_gather_in_bank(op, target.vector_lanes, target.vlen_bits, lmul)
    return module
