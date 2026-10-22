"""tritonpytorchsim-refuse-unlowered-math: after lower-vector-functions, a math op the base ISA has no
instruction for (not RVV_MATH, or RVV_NO_INSTRUCTION) that the vector unit did not take is refused,
naming it -- nothing expands it here, and below here it would become a libm call."""
import sys

from ... import toolchain
from .lib_ir import walk
from .lib_math import RVV_MATH, RVV_NO_INSTRUCTION

toolchain.activate_bindings(sys.path)

from mlir.dialects import math as _math  # noqa: E402

NAME = "tritonpytorchsim-refuse-unlowered-math"

MARKERS = tuple(sorted(({c.OPERATION_NAME for c in vars(_math).values()
                         if isinstance(getattr(c, "OPERATION_NAME", None), str)
                         and c.OPERATION_NAME.startswith("math.")} - set(RVV_MATH))
                        | set(RVV_NO_INSTRUCTION)))

FORBIDS = MARKERS


def run(module, *, target, **_):
    """Refuse the module if a math op of MARKERS is left in it."""
    from .lower import LOWER_VECTOR_FUNCTIONS
    from .slots import active, providers
    left = sorted({op.name for op in walk(module.operation) if op.name in MARKERS})
    if not left:
        return module
    took = [m.NAME for m in providers()
            if m.SLOT == LOWER_VECTOR_FUNCTIONS.name and active(m, target)]
    raise NotImplementedError(
        f"{', '.join(left)} reached {NAME} on {target.name} (isa {', '.join(target.isa)}): the base "
        f"ISA has no instruction for it, {LOWER_VECTOR_FUNCTIONS.name} "
        + (f"({', '.join(took)}) does not lower it" if took else "has no pass on this target")
        + ", and no expansion of it exists here")
