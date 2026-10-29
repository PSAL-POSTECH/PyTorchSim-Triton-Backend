"""What every stage shares: the error it raises, the context it runs in, and how it runs a tool.
Its own module so the stage modules and the suite's stages can import it without
importing pipeline, which imports them."""
import hashlib
import os
import subprocess
import time

from . import machine
from ..contract.stop import Reason, Refusal
from . import timer


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

    HEADLINE = "AN expect_ir COUNT DIFFERS"


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


class NotAPrefetchCandidate(StageError, Refusal):
    """KernelSpec.graph_prefetch holds an argument no copy node can hold in the kernel (plan-copy-node):
    the front end's election and the kernel's IR disagree."""

    reason = Reason.NOT_A_PREFETCH_CANDIDATE


DUMP_PASSES_ENV = "PYTORCHSIM_COMPILE_DUMP_PASSES"
"""Set (not "" or "0"), transform and lower write each pass's IR to 08-passes/ and 09-passes/; unset,
they do not -- printing a whole module after every pass is much of a large kernel's compile. A spec
whose expect_ir names a file there turns it on for its own run (compiler._run_stages)."""


def dump_passes_asked():
    return os.environ.get(DUMP_PASSES_ENV, "") not in ("", "0")


class Ctx:
    """One compile: the kernel, its workdir and log, and the LMUL the retry narrows."""

    def __init__(self, spec, kernel_path, workdir, verbose=False):
        self.spec = spec
        self.kernel_path = os.path.abspath(kernel_path)
        self.workdir = os.path.abspath(workdir)
        self.verbose = verbose
        self.lmul = machine.ISA_MAX_LMUL
        self.loop_lmul = {}     # loop number (legalize-vectors') -> the LMUL a retry narrowed it to
        self.dot_lmul = {}      # matmul number (matmul.DOT_FILE) -> the LMUL a retry narrowed it to
        self.dot_widths = {}    # matmul number -> the LMUL its stream was built at (the lower stage's)
        self.spill_kept = set()  # ("loop" | "dot", number) whose narrowing relieved none of its spills
        self.spill_last = None   # [(place, its width before, its spills before)] the last retry narrowed
        self.loops_settled = False  # the retry's widths stand: lower promotes one-trip loops, a spill stays
        self.spills = None       # the build's spills a narrower width may relieve (SpillNotAllowed)
        self.retry_best = None   # ((lmul, loop_lmul, dot_lmul), the tile cycles gem5 timed them at)
        self.retries = 0         # the builds the retry added (compiler.max_retries bounds them)
        self.dump_passes = dump_passes_asked()  # each pass's IR to 08-passes/, 09-passes/
        self.tool_reuse = {}     # _reuse_key -> what that call left (outputs, stdout, stderr): start_tool
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
    return finish_tool(ctx, stage, start_tool(ctx, cmd, **kw))


REUSED_TOOLS = ("llc", "opt", "mlir-translate", "riscv64-unknown-elf-gcc")
"""Tools whose outputs follow from their command line and the contents of the files it names alone:
a retry that rebuilds a stage to the same IR calls them again on the same inputs."""


def _outputs(args):
    """The files a REUSED_TOOLS command line writes: -o's and llc's -pass-remarks-output's."""
    out = [args[i + 1] for i, a in enumerate(args[:-1]) if a == "-o"]
    return out + [a.split("=", 1)[1] for a in args if a.startswith("-pass-remarks-output=")]


def _reuse_key(args, kw):
    """What a REUSED_TOOLS call's outputs follow from: the whole command line (paths and all, as llc
    writes its input's name into its output) and the contents of every file it reads; None for any
    other call. The tool itself is named by its path only: one compile runs one toolchain, and
    reading llc's own binary on every call was most of the binary stage's Python time."""
    if kw or os.path.basename(args[0]) not in REUSED_TOOLS:
        return None
    outs = set(_outputs(args))
    h = hashlib.sha256()
    h.update(args[0].encode() + b"\0")
    for a in args[1:]:
        h.update(a.encode() + b"\0")
        if a not in outs and os.path.isfile(a):
            with open(a, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


class Running:
    """A tool start_tool() left running; wait() reads what it printed once and returns it. A
    REUSED_TOOLS call that succeeds leaves its outputs in `reuse` under `key`."""

    def __init__(self, ctx, cmd, reuse=None, key=None, **kw):
        self.cmd, self.printable = cmd, " ".join(str(c) for c in cmd)
        ctx.log(f"$ {self.printable}")
        self._reuse, self._key = reuse, key
        self._t0, self._done = time.perf_counter(), None
        self._proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True, **kw)

    def wait(self):
        if self._done is None:
            try:
                out, err = self._proc.communicate()
            finally:
                timer.PROF.add_tool(self.cmd[0], time.perf_counter() - self._t0)
            self._done = subprocess.CompletedProcess(self.cmd, self._proc.returncode, out, err)
            if self._key is not None and self._proc.returncode == 0:
                files = _outputs([str(c) for c in self.cmd])
                if all(os.path.isfile(f) for f in files):
                    kept = {}
                    for f in files:
                        with open(f, "rb") as fh:
                            kept[f] = fh.read()
                    self._reuse[self._key] = (kept, out, err)
        return self._done


class Reused:
    """A REUSED_TOOLS call this compile already made with the same command line and inputs: its outputs
    written again from what that call left, as finish_tool reads a Running."""

    def __init__(self, ctx, cmd, kept):
        self.cmd, self.printable = cmd, " ".join(str(c) for c in cmd)
        files, out, err = kept
        ctx.log(f"$ {self.printable}")
        ctx.log("[reused: the same command on the same inputs ran earlier in this compile]")
        for f, data in files.items():
            tmp = f + ".reuse.tmp"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, f)
        timer.PROF.add_tool(os.path.basename(str(cmd[0])) + " (reused)", 0.0)
        self._done = subprocess.CompletedProcess(cmd, 0, out, err)

    def wait(self):
        return self._done


class Finished:
    """A tool another way already ran (frontend_server.served), as finish_tool reads a Running."""

    def __init__(self, ctx, cmd, done, seconds, how):
        self.cmd, self.printable, self._done = cmd, " ".join(str(c) for c in cmd), done
        ctx.log(f"$ {self.printable}")
        ctx.log(f"[{how}]")
        timer.PROF.add_tool(cmd[0], seconds)

    def wait(self):
        return self._done


def start_tool(ctx, cmd, **kw):
    """`cmd` started and left running, for finish_tool; a REUSED_TOOLS call this compile already made on
    the same inputs is not run again (Reused)."""
    reuse = getattr(ctx, "tool_reuse", None)
    key = _reuse_key([str(c) for c in cmd], kw) if reuse is not None else None
    if key is not None and key in reuse:
        return Reused(ctx, cmd, reuse[key])
    return Running(ctx, cmd, reuse=reuse, key=key, **kw)


def finish_tool(ctx, stage, running):
    """Wait for a start_tool() command, log what it printed, and raise a StageError naming `stage` if it
    failed; a CompletedProcess."""
    proc, printable = running.wait(), running.printable
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
