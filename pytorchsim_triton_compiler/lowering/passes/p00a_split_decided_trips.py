"""torchsim-split-decided-trips (tensors): a loop whose compares on its induction variable its
bounds decide on a run of its trips -- true on every element of them, or false -- is cut there, and
that run's compares are folded to their value, each select taking its side.

Inductor's looped reduce masks `r0_offset + arange(R0_BLOCK) < r0_numel` on every trip whenever
R0_BLOCK does not divide r0_numel: every trip but the last is full, so it is true there, and on the
full trips the mask's index, compare and select go. The ranges are read through the compare's
producers (lib_ranges); a loop with such a compare whose trips none are decided is stamped with why
(lib_ranges.MASK_KEPT). The same cut runs again on vectors (p06c), after the body is a loop.
"""

from .lib_ranges import split_at_decided_compares

NAME = "torchsim-split-decided-trips"


def run(module, lanes=None, **_opts):
    return split_at_decided_compares(module, lanes)
