"""tritonpytorchsim-bound-eup-loops-tpuv6e: a vector loop whose f32 exp goes to the SFU's EUP
(lower_vector_functions) at the widest LMUL whose groups the EUP holds while schedule_vpu_stream runs the
pushes DEPTH groups ahead of the pops: an f32 group at LMUL L is L registers, and a pop finds DEPTH + 1 groups
inside (schedule_vpu_stream.eup_registers). The machine's ISA_MAX_LMUL when they all fit."""

from ....machine import ISA_MAX_LMUL
from ...passes.lib_ir import walk
from ...passes.p06_legalize import MAX_LMUL_ATTR
from ...passes.p05_vectorize import VECTOR_LOOP
from ...passes.transform import BOUND_LOOP_WIDTHS
from .lower_vector_functions import SFU_FUNCTIONS, _is_f32_vector

NAME = "tritonpytorchsim-bound-eup-loops-tpuv6e"

REQUIRES = ("vcix_tpuv6e",)

SLOT = BOUND_LOOP_WIDTHS.name

FORBIDS = ()


def run(module, **_):
    """Every vector loop whose own body holds an f32 math.exp, bounded to the widest LMUL the EUP holds DEPTH + 1
    groups of. A loop around it is not: legalize bounds a loop by the loops around it, and those hold loops with
    no exp. Nothing is bounded when the machine does not say what the EUP holds, or when every LMUL fits:
    schedule_vpu_stream then keeps the order and says why, or pipelines at the kernel's LMUL."""
    from .... import machine
    from .schedule_vpu_stream import eup_registers
    return bound(module, eup_registers(machine.target()))


def bound(module, holds):
    """run's work for an EUP that holds `holds` registers (None: not known)."""
    from mlir.ir import IntegerAttr, IntegerType
    from .schedule_vpu_stream import DEPTH, UNROLL
    if "math.exp" not in SFU_FUNCTIONS:
        return module                   # exp is not this lowering's to send to the EUP
    fits = [1 << k for k in range(ISA_MAX_LMUL.bit_length()) if (DEPTH + 1) * UNROLL * (1 << k) <= (holds or 0)]
    if not fits or fits[-1] == ISA_MAX_LMUL:
        return module
    widest = IntegerAttr.get(IntegerType.get_signless(64), fits[-1])
    for lp in [o for o in walk(module.operation) if o.name == "affine.for" and VECTOR_LOOP in o.attributes]:
        body = [o.operation for o in lp.regions[0].blocks[0].operations]
        if any(o.name == "math.exp" and _is_f32_vector(o.results[0].type) for o in body):
            lp.attributes[MAX_LMUL_ATTR] = widest
    return module
