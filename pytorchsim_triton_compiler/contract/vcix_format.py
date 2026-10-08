"""What an element of a systolic push is, as the push's rs1 field (the vcix.iv imm) names it: one
code per format whatever the element width, as vcix-accelerator's examples/tpu/README.md states
them. Imports nothing, so the Triton frontend reads it as the compiler does."""

NATIVE = 0

BF16 = 3
