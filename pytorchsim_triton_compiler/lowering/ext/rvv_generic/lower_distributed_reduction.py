"""lower_distributed_reduction (rvv_generic): a reduction across the distributed axis has no carrier on
RVV alone yet -- the shared-scratchpad round trip that turns it into a reduction within each holder
comes with address space 2. A pack whose LayoutRules fold nothing across the lanes (rvv_generic's)
keeps every reduce's axis off them, so p05 mints none for it."""

from ...passes.lib_bank_layout import DISTRIBUTED_REDUCTION_OP
from ...passes.lib_ir import walk
from ...passes.lower import LOWER_DISTRIBUTED_REDUCTION

NAME = "pytorchsim-lower-distributed-reduction-generic"

SLOT = LOWER_DISTRIBUTED_REDUCTION.name

FORBIDS = LOWER_DISTRIBUTED_REDUCTION.lowers

MARKERS = FORBIDS


def run(module, **_):
    """Refused, naming the first one."""
    for op in walk(module.operation):
        if op.name == DISTRIBUTED_REDUCTION_OP:
            raise NotImplementedError(
                f"a {DISTRIBUTED_REDUCTION_OP} crosses banks, and rvv_generic has no carrier for that yet "
                f"(the shared-scratchpad round trip comes with address space 2): {op}")
    return module
