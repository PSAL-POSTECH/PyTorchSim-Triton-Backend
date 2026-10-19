"""torchsim-split-decided-vector-trips: p00a's cut (lib_ranges.split_at_decided_compares) once the
body is vectors: a vector loop whose element index -- the loop's induction variable, vector.step
and the lane's own number, below the machine's lanes -- its bounds put below a compare's bound on
a run of its trips, and past it on another, is cut there. The trip a block's last element falls in
keeps the compare; the trips before it run the true side and the trips after the false one, with
no index vector, compare or select."""

from .lib_ranges import split_at_decided_compares

NAME = "torchsim-split-decided-vector-trips"


def run(module, lanes=None, **_opts):
    return split_at_decided_compares(module, lanes)
