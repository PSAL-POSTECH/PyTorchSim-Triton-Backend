"""The context every stage runs in, the error it raises, and how it runs a tool."""
import os
import subprocess
import time

from . import machine
from .metrics import wallclock


class StageError(RuntimeError):
    def __init__(self, stage, message, cmd=None, stderr=None, artifact=None):
        super().__init__(message)
        self.stage = stage
        self.cmd = cmd
        self.stderr = stderr
        self.artifact = artifact


class SpillNotAllowed(StageError):
    """llc spilled vector registers where we do not allow a spill -- a narrower LMUL may not spill
    at all, so the stage machine retries at half."""


class Ctx:
    def __init__(self, spec, kernel_path, workdir, verbose=False,
                 reuse_inputs=False, runtime_dir=None):
        self.spec = spec
        self.reuse_inputs = reuse_inputs
        self.kernel_path = os.path.abspath(kernel_path)
        self.workdir = os.path.abspath(workdir)
        self.runtime_dir = (os.path.abspath(runtime_dir) if runtime_dir
                            else os.path.join(self.workdir, "runtime"))
        self.verbose = verbose
        self.inputs = None
        self.raw_paths = {}
        os.makedirs(self.workdir, exist_ok=True)
        self.log_path = os.path.join(self.workdir, "stage.log")
        self.lmul = machine.ISA_MAX_LMUL

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


def _said(text):
    for line in reversed((text or "").splitlines()):
        if line.strip():
            return line.strip()
    return ""


def run_tool(ctx, stage, cmd, **kw):
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
        said = _said(proc.stderr) or _said(proc.stdout)
        raise StageError(stage,
                         f"command failed with exit code {proc.returncode}"
                         + (f": {said}" if said else ""),
                         cmd=printable, stderr=proc.stderr or proc.stdout)
    return proc
