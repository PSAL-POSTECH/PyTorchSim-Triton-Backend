"""The VCIX op forms (the VCIX dialect's vcix.iv, vcix.i, vcix.v.i, vcix.v.iv) by their arity --
vector operands in, vector results out -- and the ISA fields' ranges. Nothing about what any
instruction does. Imports nothing, so the Triton frontend reads it as the compiler does."""

FORMS = {"iv": (1, 0), "i": (0, 0), "v.i": (0, 1), "v.iv": (1, 1)}

OPCODE_RANGE = range(0, 4)

IMM_RANGE = range(-16, 16)
