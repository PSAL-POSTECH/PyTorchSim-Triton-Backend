"""pytorchsim-verify-supported-math: after the target's LOWER_PASSES, a math op the base ISA has no
instruction for (not RVV_MATH, or RVV_NO_INSTRUCTION) that the vector unit did not take is refused,
naming it -- nothing expands it here, and below here it would become a libm call."""
import sys

from ... import toolchain
from ...contract.stop import Reason, Refusal
from ..utils.ir_utils import walk
from ..utils.math_ops import RVV_MATH, RVV_NO_INSTRUCTION

toolchain.activate_bindings(sys.path)

from mlir.dialects import math as _math  # noqa: E402
from .. import pass_name  # noqa: E402

NAME = pass_name(__name__)


class UnloweredMath(NotImplementedError, Refusal):
    """A math op the base ISA has no instruction for, which the vector unit did not take."""
    reason = Reason.MATH_UNLOWERED

MARKERS = tuple(sorted(({c.OPERATION_NAME for c in vars(_math).values()
                         if isinstance(getattr(c, "OPERATION_NAME", None), str)
                         and c.OPERATION_NAME.startswith("math.")} - set(RVV_MATH))
                        | set(RVV_NO_INSTRUCTION)))

FORBIDS = MARKERS


def run(module, *, subtarget, **_):
    """Refuse the module if a math op of MARKERS is left in it."""
    from ...targets import target
    left = sorted({op.name for op in walk(module.operation) if op.name in MARKERS})
    if not left:
        return module
    tgt = type(target(subtarget)).__name__
    raise UnloweredMath(
        f"{', '.join(left)} reached {NAME} on {subtarget.name} (isa {', '.join(subtarget.isa)}): the base "
        f"ISA has no instruction for it, the target's LOWER_PASSES ({tgt}) do not lower "
        f"it, and no expansion of it exists here")
