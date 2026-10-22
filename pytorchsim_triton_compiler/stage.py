"""What every stage shares: the error it raises, the context it runs in, and how it runs a tool.
Its own module so the stage code under lowering/ and the suite's stages can import it without
importing pipeline, which imports them."""
import os
import subprocess
import time

from . import machine
from .contract.stop import Reason, Refusal
from .metrics import wallclock


class StageError(RuntimeError):
    def __init__(self, stage, message, cmd=None, stderr=None, artifact=None):
        super().__init__(message)
        self.stage = stage
        self.cmd = cmd
        self.stderr = stderr
        self.artifact = artifact


class ExpectIrMismatch(StageError):
    """An artifact's op counts differ from the kernel's expect_ir, or could not be counted: a check
    the harness makes, so it is never the stop a refuse, blocked or open kernel declares."""


class SpillNotAllowed(StageError):
    """llc spilled vector registers where we do not allow a spill -- a narrower LMUL may not spill
    at all, so the stage machine retries at half: in the loops llc names (`loops`), in the matmul
    streams it names (`dots`) or, for spills where it names neither (`outside`), the widest
    streams; or the whole kernel when none of those can narrow."""
    loops = ()
    dots = ()
    outside = 0
    loop_spills = {}
    dot_spills = {}
    spilled_units = {}


class SpadOverflow(StageError, Refusal):
    """The kernel's tiles take `usage` bytes of each lane's scratchpad, over `budget`: the tile is
    bigger than the machine, and the caller that chose it can shrink it by usage / budget."""

    reason = Reason.SPAD_OVERFLOW

    def __init__(self, stage, message, usage, budget, **kw):
        super().__init__(stage, message, **kw)
        self.usage = usage
        self.budget = budget


class Ctx:
    """One compile: the kernel, its workdir and log, and the LMUL the retry narrows."""

    def __init__(self, spec, kernel_path, workdir, verbose=False):
        self.spec = spec
        self.kernel_path = os.path.abspath(kernel_path)
        self.workdir = os.path.abspath(workdir)
        self.verbose = verbose
        self.lmul = machine.ISA_MAX_LMUL
        self.loop_lmul = {}     # loop number (p06_legalize) -> the LMUL a retry narrowed it to
        self.dot_lmul = {}      # matmul number (lib_matmul.DOT_FILE) -> the LMUL a retry narrowed it to
        self.dot_widths = {}    # matmul number -> the LMUL its stream was built at (the lower stage's)
        self.spill_kept = set()  # ("loop" | "dot", number) whose narrowing relieved none of its spills
        self.spill_last = None   # [(place, its width before, its spills before)] the last retry narrowed
        os.makedirs(self.workdir, exist_ok=True)
        self.log_path = os.path.join(self.workdir, "stage.log")

    def path(self, name):
        return os.path.join(self.workdir, name)

    def log(self, msg):
        with open(self.log_path, "a") as f:
            f.write(msg.rstrip() + "\n")
        if self.verbose:
            print(msg.rstrip())

    def warn(self, msg):
        """Log AND always print -- for things that predict a later failure."""
        with open(self.log_path, "a") as f:
            f.write(msg.rstrip() + "\n")
        print(f"          ! {msg.rstrip()}")


class TritonPassFailed(StageError, Refusal):
    """The triton-compile worker exited non-zero: a C++ refusal and a C++ crash are one reason until
    the worker can tell them apart (#260 D8)."""
    reason = Reason.TRITON_PASS_FAILED


class WorkerRefused(StageError, Refusal):
    """A worker process refused on purpose and wrote down why (contract.stop.record)."""

    def __init__(self, reason, stage, message, **kw):
        super().__init__(stage, message, **kw)
        self.reason = reason


def run_tool(ctx, stage, cmd, **kw):
    """Run `cmd`, log what it printed, and raise a StageError naming `stage` if it failed."""
    printable = " ".join(str(c) for c in cmd)
    ctx.log(f"$ {printable}")
    _t0 = time.perf_counter()
    try:
        proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    finally:
        wallclock.PROF.add_tool(cmd[0], time.perf_counter() - _t0)
    if proc.stdout:
        ctx.log(proc.stdout)
    if proc.stderr:
        ctx.log(proc.stderr)
    if proc.returncode != 0:
        said = _last_line(proc.stderr) or _last_line(proc.stdout)
        raise StageError(stage,
                         f"command failed with exit code {proc.returncode}"
                         + (f": {said}" if said else ""),
                         cmd=printable, stderr=proc.stderr or proc.stdout)
    return proc


def _last_line(text):
    for line in reversed((text or "").splitlines()):
        if line.strip():
            return line.strip()
    return ""
