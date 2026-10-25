"""lower_matmul (rvv_generic): no pytorchsim.dot reaches a pack with rvv_generic's LayoutRules -- they
write every dot out as products and a reduction within each lane, which the common passes lower --
so one here is refused."""

from ...passes.lib_ir import walk
from ...passes.lower import LOWER_MATMUL

NAME = "pytorchsim-lower-matmul-generic"

SLOT = LOWER_MATMUL.name

FORBIDS = LOWER_MATMUL.lowers

MARKERS = FORBIDS


def run(module, **_):
    """Refused, naming the first one."""
    for op in walk(module.operation):
        if op.name in FORBIDS:
            raise NotImplementedError(
                f"a pytorchsim.dot reached rvv_generic's lower-matmul; its LayoutRules write every dot "
                f"out within each lane, so the anchor should have: {op}")
    return module
