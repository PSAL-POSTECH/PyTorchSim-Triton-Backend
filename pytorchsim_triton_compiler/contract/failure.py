"""Why a compile stopped, for the caller that chose what it compiled: the file a run a stage stops
leaves in its workdir, written from the type of what the stage raised, never from what it printed."""
import json
import os

FILE = "failure.json"

FORMAT = 1
"""1: {"format", "stage", "kind", "message"}, the facts of that kind, and "cmd" and "artifact" when
the stage had them."""

SPAD_OVERFLOW = "spad-overflow"
"""The tiles take "usage" bytes of each lane's scratchpad, over "budget": shrink by usage / budget."""

SPILL_NOT_ALLOWED = "spill-not-allowed"
"""llc spilled at LMUL 1, where there is no narrower LMUL left to retry at."""

ERROR = "error"
"""Anything no kind above names. A run that exits nonzero and leaves no FILE stopped outside every
stage, and is this kind too."""


class FailureError(RuntimeError):
    pass


def clear(workdir):
    """Drop what an earlier run left, before this one can stop."""
    try:
        os.unlink(os.path.join(workdir, FILE))
    except FileNotFoundError:
        pass


def write(workdir, stage, kind, message, **facts):
    """Leave FILE in `workdir`: the stage the run stopped in, its kind, what it said, its facts."""
    with open(os.path.join(workdir, FILE), "w") as fh:
        json.dump({"format": FORMAT, "stage": stage, "kind": kind, "message": message, **facts},
                  fh, indent=2)
        fh.write("\n")


def read(workdir):
    """What the run in `workdir` left, or None if it left nothing."""
    path = os.path.join(workdir, FILE)
    try:
        with open(path) as fh:
            got = json.load(fh)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        raise FailureError(f"{path}: not valid JSON ({e})")
    if got.get("format") != FORMAT:
        raise FailureError(f"{path}: format {got.get('format')!r}, and this compiler reads and "
                           f"writes format {FORMAT}")
    return got
