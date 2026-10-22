"""Why a compile stops on purpose, as a kernel file declares it and the compiler raises it: a Reason per
decision, not per kernel, and the record a worker process leaves of one. A leaf (stdlib only), so a
kernel file names a stop without loading MLIR into the process that runs Triton."""
import dataclasses
import enum
import json
import os

RECORD = "00-stop.json"


class Reason(enum.Enum):
    """One decision a raise site makes; sites that make the same decision share it."""
    TRITON_PASS_FAILED = "triton-pass-failed"
    ATOMICS = "atomics"
    DTYPE_NOT_RUN = "dtype-not-run"
    MATH_UNLOWERED = "math-unlowered"
    POINTER_BASE_UNTRACED = "pointer-base-untraced"
    FILL_VARIES = "fill-varies"
    POINTER_TILE_UNCONVERTED = "pointer-tile-unconverted"
    DESCRIPTOR_CANNOT_STATE = "descriptor-cannot-state"
    MATRIX_UNIT_ONE_ROW = "matrix-unit-one-row"
    MATRIX_UNIT_NO_FORMAT = "matrix-unit-no-format"
    TOG_UNCOUNTED_LOOP = "tog-uncounted-loop"
    TOG_BRANCH = "tog-branch"
    SPAD_OVERFLOW = "spad-overflow"


class Refusal(Exception):
    """A stop the compiler makes on purpose; a subclass names its `reason` and keeps the exception it
    replaced as its first parent, so every handler that caught it still does."""
    reason = None


class Refused(Refusal):
    """A refusal whose Reason is given where it is raised, for a process that writes it down."""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def record(workdir, e):
    """Leave Refusal `e` in `workdir` for the process that started this one."""
    with open(os.path.join(workdir, RECORD), "w") as f:
        json.dump({"reason": e.reason.value, "message": str(e)}, f)


def recorded(workdir):
    """(Reason, message) of the Refusal a worker left in `workdir`, or None."""
    path = os.path.join(workdir, RECORD)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        got = json.load(f)
    return Reason(got["reason"]), got["message"]


def clear(workdir):
    """Forget a record an earlier run left in `workdir`."""
    try:
        os.unlink(os.path.join(workdir, RECORD))
    except FileNotFoundError:
        pass


@dataclasses.dataclass(frozen=True)
class Stop:
    """Where a refuse, blocked or open kernel stops: the step, and a Reason, or for an open kernel the
    dotted name of the Python exception it dies on."""
    step: str
    reason: object

    def __post_init__(self):
        if not isinstance(self.reason, (Reason, str)):
            raise TypeError(f"Stop.reason is a Reason or a dotted exception name, not {self.reason!r}")
        if isinstance(self.reason, str) and "." not in self.reason:
            raise ValueError(f"Stop.reason {self.reason!r} is not a dotted exception name")
