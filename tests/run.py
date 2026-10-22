#!/usr/bin/env python
"""triton-npu -- run the whole kernel suite and print one line each.

    ./tests/run.py                                  every kernel, scored against
                                                  the committed baseline
    ./tests/run.py "tests/cases/mechanism/mask/*.py"     a subset
    ./tests/run.py --update-baseline                record today as the new bar
    ./tests/run.py --baseline other.txt             score against something else
    ./tests/run.py --no-baseline                    just the run, no comparison

WHY THIS EXISTS AND NOT A `for` LOOP. Each kernel is an independent shell-out to
pytorchsim-triton-compiler with its own workdir, sharing nothing but read-only toolchain binaries, so
the suite parallelises perfectly -- on 20 cores that is minutes instead of the
better part of an hour. A sweep nobody wants to wait for is a sweep that stops
being run before every commit, which is exactly when it is worth the most.

WHAT COUNTS AS PASSING. Not every kernel is meant to compile. Half of
tests/cases/ exists to pin down something the backend must REFUSE, and a few
are BLOCKED on something the target does not have yet. Both declare it in their
KernelSpec, pytorchsim-triton-compiler prints REFUSED / BLOCKED AS EXPECTED and exits 0, and it is
pytorchsim-triton-compiler that decides when one of those has gone stale -- so this script only has
to believe what it is told. See pytorchsim_triton_compiler/contract/kernel_spec.py:KernelSpec.expect.

OUTPUT IS SORTED BY INPUT ORDER, not by completion order, so two runs diff
cleanly. That is the whole point of --baseline: the number passing matters much
less than WHICH line moved, and a green sweep that quietly swapped one kernel for
another is the failure mode this catches.

WHAT THE EXIT CODE MEANS, AND WHY IT IS NOT "EVERYTHING PASSED". A large part of
this suite is kernels that are SUPPOSED to fail -- reproducers pinning a bug the
backend still has (tests/cases/mechanism/, and any kernel whose expect_why
records a refusal). Requiring a green sweep would mean the suite could never hold
one, and the number would sit at red forever and stop being read. So the bar is
MOVEMENT, not the count:

    exit 0   nothing that used to pass stopped passing
    exit 1   at least one kernel went PASS -> FAIL, or vanished while passing

A kernel that newly appears failing is news, not a failure: that is what adding a
reproducer looks like. A kernel that starts passing is news too, and it is the
only thing here worth celebrating -- update the baseline so the next run holds
the backend to it.

THE BASELINE IS COMMITTED, at tests/baselines/<target>.txt, which is what makes this
work off main. A branch scores against the baseline IT inherited, so the diff is
exactly what that branch did to the compiler and nothing else -- no need to run
main first to have something to compare with, and two branches can be scored
without either seeing the other. Regenerate it deliberately with
--update-baseline, never as a side effect of a run.
"""
import argparse
import fcntl
import glob
import os
import pathlib
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

#: The repo root -- this file is tests/, and both the kernel paths in the report
#: and the import of the compiler package are resolved against it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: ONCE, AT LOAD: the pool's threads import the package, and one thread putting
#: ROOT on sys.path while another walks it can make that walk miss the package.
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# THE SUITE OWNS ITS MACHINE. There is no shipped default any more: a target
# description states the scratchpad window and the DRAM range too, so the
# compiler supplying half of it was a second place the machine was stated.
import suite_target  # noqa: E402
os.environ.setdefault("TORCHSIM_COMPILE_TARGET", suite_target.NAME)
#: The bar, tracked in git so every branch inherits one. See the module
#: docstring for why this is committed rather than kept per-machine.
def baseline_path():
    """The record for the machine being compiled for, by that target's name.

    KEYED BY TARGET because the answer depends on it: two coverage kernels give
    different results at 128 lanes and at 256, so one committed file cannot be
    the bar for both.
    """
    from pytorchsim_triton_compiler import machine, toolchain
    return os.path.join(HERE, "baselines", machine.target().name + ".txt")
#: RECURSIVE, and it has to be: the suite is two layers deep now
#: (mechanism/ptr/, mechanism/unstructured/) and a fixed depth silently reached
#: 130 of 384 the moment workloads/ and mechanism/ appeared.
DEFAULT_GLOBS = ("tests/cases/**/*.py",)
ERR = re.compile(r"max_abs_err = (\S+)")

def _dma(workdir):
    """(mvin, mvout) this compile recorded, (0, 0) if it did not get that far.

    READ, NOT RECOMPUTED. This used to parse 08-transformed.mlir here, and the torchsim-compile
    rename broke that regex while the sweep still reported "no change".
    """
    from pytorchsim_triton_compiler.metrics import quality
    return quality.read(workdir)


def _uniform(workdir):
    """(UNIFORM buffers, those with an axis that could spread) this compile recorded:
    tiles in one bank, the same in every lane -- information, not a cost; a rank-0
    buffer has no axis to spread."""
    from pytorchsim_triton_compiler.metrics import quality
    return quality.read_uniform(workdir)

# pytorchsim-triton-compiler's failure report has exactly two shapes and these read them, rather
# than fishing for prose that might be anywhere.
#
#   [6/8] binary    FAIL  (0.1s)  mlir-translate -> llc -> ...   <- the STAGE
#   [  extra  ] togcheck FAIL  (0.2s)  ...   <- A BRANCH STAGE IS ALSO ONE.
# `\d+/\d+` matched only the main line, so every StageError raised by `tog` or
# `togcheck` lost its banner and with it the branch that reads a StageError:
# measured, 63 of 77 --tog failures reported "failed with no diagnostic".
STAGE_FAIL = re.compile(r"^\[[^\]]+\] (\w+) +FAIL\b")
#   pytorchsim_triton_compiler.lowering.passes.lib_lane_axis.Fatal: 'linalg.reduce' is ...      <- the REASON
# Python prefixes an exception with its DEFINING MODULE unless it is a builtin,
# so the one this backend raises most arrives fully qualified -- and MLIRError
# arrives as `mlir._mlir_libs._site_initialize.<locals>.MLIRError`, which is why
# the qualifier is matched with \S* and not \w*.
#
# THE NAME IS NOT READ, and that is the point. This matched
# `\w*(?:Error|Fatal|Exception|Overflow)` for one release and it silently lost
# every exception whose class does not happen to end that way -- measured, 31 of
# the 82 failures in a 256-kernel sweep, all of them `NotVectorisable` from
# vectorize, every one reported as `adapt: full log: <path>` at a path whose
# stage.log stops three stages earlier and never held the reason. That is the
# same defect this backend keeps fixing in its passes: recognising a name
# instead of reading the structure. Python's traceback format IS the structure --
# indented frames, then the exception on the first unindented line -- so it is
# read here and the class name is whatever it says.
EXC = re.compile(r"^(?:[\w.<>]+\.)?(\w+)(?::\s?(.*))?$")
TRACEBACK = "Traceback (most recent call last):"
# The old name-shaped one, kept for the ONE case with no traceback to anchor to.
EXC_NAMED = re.compile(r"^(?:\S*\.)?(\w*(?:Error|Fatal|Exception|Overflow))"
                       r"(?::\s?(.*))?$")
# WHICH EXCEPTION IN A CHAIN IS THE DIAGNOSIS -- Python says so in as many words,
# and the two sentences mean opposite things:
#
#   IMPLICIT  something raised while the first was being handled. The second is
#             CLEANUP that also went wrong, and the first is the cause. Measured:
#             five kernels raised `RuntimeError: Invalid attribute ... bad_cast`
#             in vectorize, whose unwinding then left an insertion point
#             open, so `RuntimeError: Unbalanced Location enter/exit` was raised
#             from the `with` in transform.run. Reading the LAST one named the
#             `with` statement in pipeline.py for a bug three frames deep in a
#             pass.
#   EXPLICIT  `raise X from Y` -- a deliberate translation, and the whole point
#             of writing it is that X is the sentence to report.
#
# So neither "first" nor "last" is right; the separator is.
CHAIN_IMPLICIT = "During handling of the above exception, another exception occurred:"
CHAIN_EXPLICIT = "The above exception was the direct cause of the following exception:"
# Python's unraisable-exception header. What follows is a traceback from a
# __del__ or an abandoned generator being finalised, which by construction ran
# AFTER whatever actually failed and never caused it.
UNRAISABLE = "Exception ignored"
# faulthandler's header. pytorchsim-triton-compiler arms it so a signal that kills the interpreter
# still names the Python frame it died in.
FATAL = "Fatal Python error:"
#   File "/long/path/pytorchsim-triton-compiler/pass_elf/widen_broadcast_loads.py", line 51 in _widen
FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+) in (\S+)')


def why_failed(out, rc):
    """The one line that says WHY, out of pytorchsim-triton-compiler's whole transcript.

    NOT THE STAGE BANNER. That is what this fell back to for 34 of the 38
    failures in the 139/177 sweep, and it is wrong twice over:

      it says nothing   `[4/8] adapt FAIL ... the tts-level backend` is the
                        same string for a lane-axis conflict, an IndexError in
                        a pass and an MLIR verifier error. Thirty kernels
                        failing for three unrelated reasons read as one.
      it carries TIME   `(0.4s)` is wall clock. Two sweeps of an UNCHANGED tree
                        produced reports differing on ~15 lines, so a plain
                        `diff before.txt after.txt` -- which the module
                        docstring above promises is clean -- showed kernels
                        "moving" that had not moved. That noise is what makes a
                        no-op edit look like it changed the suite.

    The prefix list this replaces (`startswith("Fatal", ...)`) was written for
    the REFUSED path, where pipeline.py prints `type(e).__name__` and the name
    really is bare. On the FAIL path it prints a raw traceback instead, whose
    last line is `pytorchsim_triton_compiler.lowering.passes.lib_lane_axis.Fatal: ...` -- so the qualifier
    meant the list matched nothing at all, on every failing kernel.
    """
    lines = out.splitlines()

    # A refusal that has stopped refusing fails with NO failing stage: every one
    # of them passed and pytorchsim-triton-compiler rejected the result. Neither branch below finds
    # anything to say about it, and the line it would otherwise land on is the
    # `the file says:` under the headline rather than the headline itself.
    if "EXPECTED A REFUSAL" in out:
        return "EXPECTED A REFUSAL, BUT EVERY STAGE PASSED -- drop `expect`"

    stage, banner = "", -1
    for i, line in enumerate(lines):
        m = STAGE_FAIL.match(line)
        if m:
            stage, banner = m.group(1), i
            break

    # THE ONE THE CHAIN POINTS AT, not the last one printed. See CHAIN_IMPLICIT
    # above: "last" was this file's rule for one release and it reported the
    # unwinding rather than the fault on every kernel that raised inside an
    # MLIR context manager.
    #
    # ANCHORED TO THE TRACEBACK, not scanned over the whole transcript. EXC is
    # deliberately permissive now -- it has to be, to name a class this file has
    # never heard of -- so it is only ever applied where Python guarantees an
    # exception stands: the first UNINDENTED line after the indented frames.
    why, at = "", -1
    adopt = True              # the first exception found is always the diagnosis
    for i, line in enumerate(lines):
        s = line.strip()
        if s == CHAIN_IMPLICIT or s.startswith(UNRAISABLE):
            adopt = False
            continue
        if s == CHAIN_EXPLICIT:
            adopt = True
            continue
        if s != TRACEBACK:
            continue
        for j in range(i + 1, len(lines)):
            if not lines[j] or lines[j][0].isspace():
                continue                      # still inside the frames
            m = EXC.match(lines[j])
            if m and (adopt or not why):
                why, at = f"{m.group(1)}: {(m.group(2) or '').strip()}".strip(), j
            break
    if not why:
        # No traceback at all: the REFUSED path prints `type(e).__name__` bare,
        # with nothing above it to anchor to. Only there is a guess at the name
        # the only thing available, and it stays as narrow as it always was.
        for i, line in enumerate(lines):
            m = EXC_NAMED.match(line.strip())
            if m:
                why, at = f"{m.group(1)}: {(m.group(2) or '').strip()}".strip(), i
    # MLIRError's own message is the header "Failure while executing pass
    # pipeline:" and the diagnosis is the `error:` line printed under it.
    if why.endswith(":"):
        for line in lines[at + 1:at + 4]:
            if line.strip().startswith("error:"):
                why = line.strip()
                break

    if not why and banner >= 0:
        # No traceback, so this is a StageError: pipeline.py prints it as the
        # first indented line under the FAIL banner, and that IS our diagnosis.
        # BELOW THE BANNER and not from the top, because the stage ABOVE the one
        # that failed prints indented lines of its own -- verify's transcript
        # opens with spike's `o  FAIL  shape=(2048,)`, which names the tensor
        # and says nothing about why it is wrong.
        #
        # `banner >= 0` because a run KILLED mid-stage never printed a banner,
        # and `lines[-1 + 1:]` is the whole transcript -- whose first indented
        # line is a traceback frame, i.e. a file and a line number in place of a
        # reason.
        for line in lines[banner + 1:]:
            if line.startswith("  ") and line.strip():
                why = line.strip()
                break

    if not why:
        # A SEGFAULT HAS NO TRACEBACK -- there is no exception, the interpreter
        # is simply gone. pytorchsim-triton-compiler arms faulthandler for exactly this, and what it
        # prints is a header and then the C-level stack of Python frames,
        # innermost FIRST. The innermost frame is the diagnosis: measured on
        # nf4_codebook_dequant, `widen_broadcast_loads.py, line 51` -- a
        # dangling Operation from an op erased earlier in the same walk, which
        # `KILLED by signal 11` said nothing about.
        for i, line in enumerate(lines):
            if not line.startswith(FATAL):
                continue
            frame = next((FRAME.search(f) for f in lines[i + 1:i + 6]
                          if FRAME.search(f)), None)
            where = (f" in {os.path.basename(frame.group(1))}:{frame.group(2)} "
                     f"{frame.group(3)}" if frame else "")
            why = f"{line.strip()}{where}"
            break

    # A FATAL SIGNAL IS NOT A DIAGNOSIS, IT IS AN ENDING -- and for one release
    # this returned it as though it were, before looking at the transcript at
    # all. It is worth spelling out what that cost, because it is the shape of
    # mistake this file keeps making:
    #
    #     measured   6 of 261 kernels reported `KILLED by signal 6 (SIGABRT) --
    #                no diagnostic was printed`. Every one of them had printed a
    #                full traceback naming one line of one pass. The abort came
    #                LATER, at interpreter shutdown, from MLIR handles the failed
    #                pass had left open -- so the process died of the cleanup and
    #                the report threw away the cause to announce the cleanup.
    #
    # The signal still gets said, because a driver that dies after reporting is
    # itself a defect and hiding it would be the opposite error. It is a PREFIX.
    if rc is not None and rc < 0:
        try:
            import signal
            name = signal.Signals(-rc).name
        except (ValueError, ImportError):
            name = "?"
        killed = f"KILLED by signal {-rc} ({name})"
        if not why:
            return f"{killed} -- no diagnostic was printed"
        # `in <stage>` only when a stage banner actually named one. A run killed
        # mid-stage never printed a banner, and there is nothing to name.
        return f"{killed}{f' in {stage}' if stage else ''}: {why}"

    if not why:
        why = "failed with no diagnostic"

    # The numbers ARE the reason a verify failed, and without them a kernel
    # drifting from 1e-3 to 6.0 looks identical in the report to one that never
    # worked.
    if stage == "verify":
        why = " ".join([why] + ERR.findall(out))
    return f"{stage}: {why}" if stage else why


#: Where sweeps put artifacts when nobody said. Kept stable so a single sweep
#: reuses it (see workdir_for) -- claim_workroot is what keeps a SECOND sweep off it.
DEFAULT_WORKROOT = "/tmp/torchsim-regress"


def default_jobs():
    """How many kernels to have in flight, from what this process may actually run.

    NOT `os.cpu_count()`. That is the machine's core count, not ours: under a cpuset
    or a container CPU limit it is simply a bigger number, and the workers it sizes
    are real processes holding real memory whether or not there is a core for them.
    Measured here: cpu_count() 36, sched_getaffinity 30 -- and one sweep at -j 34
    peaks at 47 processes and 8.4 GiB, roughly 250 MiB per worker, so the default
    picks the memory bill off the host's spec sheet. Sessions do not know about each
    other either; N of them is N times this, linearly (3 measured at 24.5 GiB).
    """
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:                      # not Linux
        n = os.cpu_count() or 4
    return max(1, n - 2)


def claim_workroot(workroot):
    """Take `workroot` exclusively, or hand back a private one and say so.

    TWO SWEEPS IN ONE WORKROOT MIX THEIR ARTIFACTS, and the mixing is silent. The
    per-kernel directory name is the kernel PATH (see workdir_for), which makes
    the 34 workers of ONE sweep collision-free -- each kernel is visited once --
    and does nothing at all about a second sweep, which visits every one of them
    again. Both then run pytorchsim-triton-compiler into the same directory and overwrite each
    other's stage dumps, .o and .raw mid-build.

    That is not a crash, which is the problem. Measured: three concurrent sweeps
    on this default, and the one that survived reported 275/308 where a lone
    sweep reports 281/308 -- 5 `binary: ld returned 1` and 1 `spike: No such
    file`, six failures belonging to no compiler. `tests/run.py` exists to say
    WHICH line moved; six invented lines is the one output worse than none.

    So: flock the workroot. Uncontended (the normal case, one sweep) it is the
    stable path and nothing changes. Contended, fall back to a private root and
    LOG it -- a sweep that quietly relocated is its own confusing bug report.
    An explicit --workdir is never touched: that caller took responsibility.
    """
    os.makedirs(workroot, exist_ok=True)
    lock = open(os.path.join(workroot, ".sweep.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        private = f"{workroot}-{os.getpid()}"
        print(f"note: {workroot} is held by another sweep -- using {private}",
              file=sys.stderr, flush=True)
        os.makedirs(private, exist_ok=True)
        return private, None
    return workroot, lock       # keep `lock` alive: closing it drops the flock


def workdir_for(workroot, path):
    """Where this kernel's artifacts go, and the STALE one taken out first.

    THE STAGE-3 ARTIFACT IS DELETED, and it is the round-trip count that needs
    it. The workdir is REUSED across sweeps -- that is what makes a rerun cheap
    -- so a kernel that dies inside stage 4 leaves the PREVIOUS sweep's
    `08-transformed.mlir` sitting there, and the counter reads it and reports
    numbers for a compiler that is no longer in the tree.

        measured   dl/flux_dual_stream_attn and dl/mel_logspec were refused by
                   vectorize in both the before and the after tree, and the
                   after sweep reported 6 and 2 trips against 0 -- 8 of an
                   apparent 53 -> 61. The file was 13 minutes old and came from
                   an intermediate tree that had let them through.

    A number that silently describes an older compiler is worse than no number,
    which is the same defect this file's own trip-note comment records from the
    other side (counting only the kernels that already work).

    THE NAME IS THE PATH, NOT THE BASENAME, and that is not tidiness. The suite
    is 275 kernels over 13 directories and the basename is not unique by
    anything -- `tests/cases/smoke/reduce.py` and a `tests/cases/mechanism/reduce/reduce.py`
    would land in ONE directory, and the sweep runs 18 of these at a time, so
    the two would interleave their artifacts and each would be scored partly on
    the other's. The failure that produces is a kernel whose line changes
    between two runs of an unchanged tree: exactly the "the suite is
    nondeterministic" claim rule 1 records as measured false, resurrected in a
    form nobody would find. Deriving the name from the path makes the collision
    unrepresentable rather than merely absent today.
    """
    wd = _recorded_in(workroot, path)
    stale = os.path.join(wd, "08-transformed.mlir")
    # UNLINK, DO NOT check-then-remove. What is wanted is the state "that file is
    # not there", so someone else having got there first IS the goal met, not an
    # error. `if exists: remove` reads the same but is two steps, and another
    # sweep sharing this workroot fits between them: both see it, one removes it,
    # the other raises FileNotFoundError out of a worker -- and because this call
    # sits outside one()'s try, that escapes pool.map and takes the WHOLE sweep
    # down, not the one kernel. Measured: three concurrent default-workdir sweeps,
    # two died on tests/cases/smoke/add.py (alphabetically first, so within seconds) with
    # zero lines of output. See also the per-session default workroot below --
    # this stops the crash, that stops the artifact mixing.
    pathlib.Path(stale).unlink(missing_ok=True)
    return wd


def _moves_between_lanes(workroot, files):
    """One line: the sweep's moves between lanes, the XLU's by kind, summed off each compile's
    record."""
    from pytorchsim_triton_compiler.metrics import quality
    xlu = {}
    for path in files:
        got = quality.read_all(_recorded_in(workroot, path))
        for kind, n in got.get("xlu", {}).items():
            xlu[kind] = xlu.get(kind, 0) + n
    kinds = ", ".join(f"{k} {n}" for k, n in sorted(xlu.items()))
    return (f"moves between lanes: {sum(xlu.values())} through the XLU"
            + (f" ({kinds})" if kinds else ""))


def _recorded_in(workroot, path):
    """The workdir a kernel's compile recorded itself in, without making it."""
    return os.path.join(workroot, os.path.relpath(path, ROOT)[:-3].replace(os.sep, "__"))


def _uniform_tiles(workroot, files):
    """The sweep's UNIFORM tiles that have an axis to spread, summed off each compile's record."""
    return sum(_uniform(_recorded_in(workroot, path))[1] for path in files)


def pass_cases(path):
    """One case script (tests/passes: a pass on IR no kernel reaches; tests/runtime: the target and
    what the devices hand the simulators): PASS iff it exits 0."""
    rel = os.path.relpath(path, ROOT)
    r = subprocess.run([sys.executable, path], cwd=ROOT, capture_output=True, text=True)
    bad = [l for l in r.stdout.splitlines() if l.startswith("FAIL")]
    if r.returncode == 0:
        return f"PASS  {rel:<62} pass cases"
    return f"FAIL  {rel:<62} {(bad or [r.stderr.strip()[-120:]])[0][:120]}"


def one(path, workroot, timeout, tog=False, timing=False):
    """Run one kernel to completion and reduce it to a single line.

    NOTHING RAISES OUT OF HERE. This runs inside pool.map, and pool.map re-raises
    the first worker exception from the ITERATOR -- so one kernel throwing does
    not cost one line, it costs the sweep: 308 kernels replaced by a traceback.
    A sweep that reports `FAIL <kernel> <error>` is readable and diffable against
    the baseline; a sweep that reports nothing is not news, it is a lost run. The
    concrete one that taught this was a FileNotFoundError from workdir_for (see
    there), but the guard is not about that bug -- it is about the shape.
    """
    rel = os.path.relpath(path, ROOT)
    try:
        wd = workdir_for(workroot, path)
    except OSError as e:
        return f"FAIL  {rel:<62} workdir: {type(e).__name__}: {e}"
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "torchsim-compile"),
                            rel, "--workdir", wd] + (["--tog"] if tog else [])
                           + (["--timing"] if timing else []),
                           cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return f"FAIL  {rel:<62} TIMEOUT after {timeout}s"
    except Exception as e:                      # noqa: BLE001 -- see the docstring
        return f"FAIL  {rel:<62} runner: {type(e).__name__}: {e}"

    trip_note = ""
    cyc = re.search(r"^TOGSIM CYCLES: (\d+)", out, re.M)
    if cyc:
        trip_note += f"  [togsim {cyc.group(1)} cyc]"
    _, uniform = _uniform(wd)
    if uniform:
        trip_note += f"  [{uniform} UNIFORM]"

    for word, how in (("REFUSED", "refused by design"),
                      ("BLOCKED", "blocked on the target"),
                      ("OPEN", "open, ours to fix")):
        key = f"{word} AS EXPECTED:"
        if key in out:
            why = out.split(key, 1)[1].strip().splitlines()[0]
            # THE TRIP NOTE GOES HERE TOO. A kernel that stops early still HAS
            # the round trips fit_layout_to_lanes gave it, and rule 8b makes that
            # a tracked number -- dropping it hid 12 of them the day `open`
            # moved 22 kernels onto this branch.
            return f"PASS  {rel:<62} {how} -- {why[:80]}{trip_note}"
    # THE TRIP NOTE GOES HERE TOO, for the reason it goes on the branch above:
    # a kernel that just went green still HAS its round trips, and leaving them
    # off made the p15 bank-bound fix read as 95 -> 94 when it was 95 -> 95.
    if "NOT EXERCISED:" in out:
        why = out.split("NOT EXERCISED:", 1)[1].strip().splitlines()[0]
        return f"PASS  {rel:<62} not exercised -- {why[:80]}{trip_note}"
    if "THE BLOCK HAS LIFTED" in out:
        return f"FAIL  {rel:<62} THE BLOCK HAS LIFTED -- drop `expect`{trip_note}"
    if "THE GAP HAS CLOSED" in out:
        return f"FAIL  {rel:<62} THE GAP HAS CLOSED -- drop `expect`{trip_note}"
    if "ALL STAGES PASSED" in out:
        # The errors are worth carrying even on a pass: a kernel that starts
        # answering 1e-3 where it used to answer 1e-7 is still green and still
        # news, and the diff is the only place that shows up.
        return f"PASS  {rel:<62} {' '.join(ERR.findall(out))}{trip_note}"

    # Failed. The stage it died in and the reason it gave, and nothing that is
    # not reproducible -- see why_failed.
    #
    # THE TRIP NOTE GOES ON A FAILURE TOO, and it did not for one release. A
    # kernel that dies below stage 3 still HAS the round trips fit_layout_to_lanes
    # gave it, and 룰 8b makes their count a tracked number -- so leaving them
    # off meant the suite total measured only the kernels that already work.
    #
    #     measured   the sweep reported 15 trips, 2 pointless. Counting every
    #                kernel that reached 08-transformed.mlir: 51 and 12. The worst
    #                offender in the suite, gated_delta_rule at 24 trips and 8
    #                pointless, was invisible because it dies at `lower`.
    #
    # Which is the shape of the mistake this file keeps making: a metric that
    # quietly excludes is worse than no metric, because it reads as coverage.
    return f"FAIL  {rel:<62} {why_failed(out, r.returncode)[:120]}{trip_note}"


def sweep_breakdown(workroot, files):
    """The suite's wall clock, off every kernel's record.

    THIS ONLY SAYS WHICH WORKDIR IS WHOSE. Reading and rendering the record is
    prof's, where the format is -- a kernel that died in `adapt` still
    contributes the time it spent there, because the record is written either way.
    """
    from pytorchsim_triton_compiler.metrics import wallclock

    entries = []
    for path in files:
        try:
            entries.append((os.path.relpath(path, ROOT), workdir_for(workroot, path)))
        except OSError:
            continue
    return wallclock.sweep(entries) or (
        f"no {wallclock.FILENAME} under {workroot} -- nothing to break down "
        f"(every kernel failed before the compiler started?)")


#: Checked once per --tog sweep before any kernel: the IR refusals of a wait or a barrier, the
#: loop bounds a wait's size and the TOG's loops read, and what the timing devices hand TOGSim
#: (overlap 0, the Issue row).
SEMAPHORE_CHECKS = ("runtime/wait_refusals.py", "runtime/loop_bounds_cases.py",
                    "runtime/timing_cases.py")


def check_semaphores():
    """Run SEMAPHORE_CHECKS; exit naming the first that fails."""
    for rel in SEMAPHORE_CHECKS:
        proc = subprocess.run([sys.executable, os.path.join(HERE, rel)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"tests/{rel} failed (rc={proc.returncode}); no kernel ran:\n"
                     f"{proc.stdout}{proc.stderr}")
        print(f"tests/{rel}: every case gets what it names", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="*", default=[],
                    help="kernel globs, relative to the repo root")
    ap.add_argument("-j", "--jobs", type=int, default=default_jobs())
    ap.add_argument("-o", "--out", help="also write the report here")
    ap.add_argument("--baseline", default=None,
                    help="report to score against (default: the committed one)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="run without scoring")
    ap.add_argument("--update-baseline", action="store_true",
                    help="rewrite the committed baseline from this run")
    # `None` and not the path, so main() can tell "did not say" (claim it, and
    # step aside if another sweep holds it) from "said this one" (obey exactly).
    ap.add_argument("--workdir", default=None,
                    help=f"artifact root (default: {DEFAULT_WORKROOT}, or a "
                         f"private one if that is already held by a sweep)")
    ap.add_argument("--timeout", type=int, default=1800)
    # OFF BY DEFAULT ONLY UNTIL FOUR KERNELS ARE FIXED. Nothing in this sweep
    # builds a TOG, so the 3,383 lines that make one are gated by
    # nothing -- measured within the hour: moving two constants broke every
    # --tog build and 384 green kernels said nothing.
    ap.add_argument("--tog", action="store_true",
                    help="also emit the tile-operation-graph (TOG) for every kernel")
    ap.add_argument("--timing", action="store_true",
                    help="--tog, then gem5 tile cycles and a TOGSim run per kernel")
    ap.add_argument("--breakdown", action="store_true",
                    help="also print where the sweep's wall clock went, summed "
                         "over every kernel (stage, pass, tool)")
    args = ap.parse_args()

    files = []
    for g in (args.globs or DEFAULT_GLOBS):
        files += sorted(glob.glob(os.path.join(ROOT, g), recursive=True))
    files = sorted(set(files))
    if not files:
        sys.exit("no kernels matched")

    if args.tog or args.timing:
        check_semaphores()

    if args.workdir is None:
        workroot, _sweep_lock = claim_workroot(DEFAULT_WORKROOT)
    else:
        workroot, _sweep_lock = args.workdir, None   # caller's call, caller's problem

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        lines = list(pool.map(
            lambda p: one(p, workroot, args.timeout, args.tog, args.timing), files))
    if not args.globs:
        lines += [pass_cases(p) for d, pat in (("passes", "*.py"), ("runtime", "*_cases.py"))
                  for p in sorted(glob.glob(os.path.join(HERE, d, pat)))]
    for line in lines:
        print(line, flush=True)

    npass = sum(l.startswith("PASS") for l in lines)
    print(f"\n{npass} of {len(lines)} passing")
    #: WHAT A LAYOUT COSTS: lanes, and the moves between them the cross-lane unit makes, by kind.
    print(f"{_uniform_tiles(workroot, files)} UNIFORM tiles (one bank, the same in every lane) that "
          f"have an axis that could spread -- information, not a cost")
    print(_moves_between_lanes(workroot, files))
    if args.breakdown:
        print()
        print(sweep_breakdown(workroot, files))
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")

    baseline = args.baseline or baseline_path()
    if args.update_baseline:
        os.makedirs(os.path.dirname(baseline), exist_ok=True)
        with open(baseline, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nbaseline updated: {os.path.relpath(baseline, ROOT)}")
        sys.exit(0)

    if args.no_baseline or not os.path.exists(baseline):
        if not args.no_baseline:
            print(f"\nno baseline at {os.path.relpath(baseline, ROOT)} -- "
                  f"nothing to score against. --update-baseline writes one.")
        sys.exit(0)

    # Verdict and path only. The error column moves for reasons that are not
    # regressions (a kernel that now runs prints numbers it did not before),
    # and a diff that cries wolf gets ignored.
    def verdicts(text):
        out = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] in ("PASS", "FAIL"):
                out[parts[1]] = parts[0]
        return out

    with open(baseline) as f:
        was = verdicts(f.read())
    now = verdicts("\n".join(lines))

    broke = sorted(k for k in now if was.get(k) == "PASS" and now[k] == "FAIL")
    fixed = sorted(k for k in now if was.get(k) == "FAIL" and now[k] == "PASS")
    added = sorted(k for k in now if k not in was)
    # A baseline entry missing from this run means "deleted" only when the run
    # covered everything. On a subset it means "not asked for", and reporting
    # those as lost would make every targeted run red -- which is how a scoring
    # tool stops being run at all.
    subset = bool(args.globs)
    gone = [] if subset else sorted(k for k in was if k not in now)

    print()
    for k in broke:
        print(f"  REGRESSED  PASS -> FAIL   {k}")
    for k in gone:
        print(f"  {'REGRESSED' if was[k] == 'PASS' else 'removed  '}  "
              f"{was[k]} -> absent   {k}")
    for k in fixed:
        print(f"  fixed      FAIL -> PASS   {k}")
    for k in added:
        print(f"  new        {now[k]:<12} {k}")
    if not (broke or gone or fixed or added):
        print(f"no change against {os.path.relpath(baseline, ROOT)}"
              + (" (subset)" if subset else ""))

    lost = broke + [k for k in gone if was[k] == "PASS"]
    if lost:
        print(f"\n{len(lost)} kernel(s) stopped passing")
    elif fixed:
        print(f"\n{len(fixed)} kernel(s) started passing -- "
              f"rerun with --update-baseline to hold the backend to it")
    sys.exit(1 if lost else 0)


if __name__ == "__main__":
    main()
