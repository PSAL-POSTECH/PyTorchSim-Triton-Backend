#!/usr/bin/env python
"""triton-npu -- run the whole kernel suite and print one line each.

    ./tests/run.py                                  every kernel, scored against
                                                  the committed baseline
    ./tests/run.py "tests/cases/mechanism/mask/*.py"     a subset
    ./tests/run.py --update-baseline                record today as the new bar
    ./tests/run.py --baseline other.txt             score against something else
    ./tests/run.py --no-baseline                    just the run, no comparison

Each kernel is its own pytorchsim-triton-compiler process and workdir, so the suite runs in parallel.
A kernel that declares a refusal or a block in its KernelSpec (expect) passes when the compiler says
so. Lines come out in input order, so two reports diff cleanly, and the exit code is movement, not
the count:

    exit 0   nothing that used to pass stopped passing
    exit 1   at least one kernel went PASS -> FAIL, or vanished while passing

The baseline is committed at tests/baselines/<target>.txt, so a branch scores against the one it
inherited; rewrite it only with --update-baseline.
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

#: The repo root: kernel paths in the report and the compiler package resolve against it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Once, at load: a pool thread putting ROOT on sys.path while another walks it can miss the package.
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# The suite names its machine; the compiler has no default one.
import suite_target  # noqa: E402
os.environ.setdefault("TORCHSIM_COMPILE_TARGET", suite_target.NAME)
from pytorchsim_triton_compiler.stage import ExpectIrMismatch  # noqa: E402
def baseline_path():
    """The committed record for the machine being compiled for, by its name: results differ by
    machine (128 lanes against 256), so each target has its own."""
    from pytorchsim_triton_compiler import machine
    return os.path.join(HERE, "baselines", machine.target().name + ".txt")
#: Recursive: the suite is nested, and a fixed depth silently misses kernels.
DEFAULT_GLOBS = ("tests/cases/**/*.py",)
ERR = re.compile(r"max_abs_err = (\S+)")

def _uniform(workdir):
    """(UNIFORM buffers, those with an axis that could spread) this compile recorded:
    tiles in one bank, the same in every lane -- information, not a cost; a rank-0
    buffer has no axis to spread."""
    from pytorchsim_triton_compiler.metrics import quality
    return quality.read_uniform(workdir)

# The stage banner `[6/8] binary FAIL (0.1s) ...`; a branch stage (`[  extra  ] togcheck FAIL`) is one too.
# WHY_WIDTH: the characters of a FAIL line that say why; past them the line is cut.
WHY_WIDTH = 120
STAGE_FAIL = re.compile(r"^\[[^\]]+\] (\S+) +FAIL\b")
# The exception on the first unindented line after a traceback's frames, whatever its class is named;
# the qualifier may be a module path or `mlir._mlir_libs._site_initialize.<locals>`.
EXC = re.compile(r"^(?:[\w.<>]+\.)?(\w+)(?::\s?(.*))?$")
TRACEBACK = "Traceback (most recent call last):"
# By name, only for a refusal, which prints `type(e).__name__` with no traceback above it.
EXC_NAMED = re.compile(r"^(?:\S*\.)?(\w*(?:Error|Fatal|Exception|Overflow))"
                       r"(?::\s?(.*))?$")
# In a chain, IMPLICIT means the later exception is cleanup and the first is the cause; EXPLICIT
# (`raise X from Y`) means the later one is the sentence to report.
CHAIN_IMPLICIT = "During handling of the above exception, another exception occurred:"
CHAIN_EXPLICIT = "The above exception was the direct cause of the following exception:"
# Python's unraisable-exception header: a finaliser's traceback, which ran after the failure.
UNRAISABLE = "Exception ignored"
# faulthandler's header. pytorchsim-triton-compiler arms it so a signal that kills the interpreter
# still names the Python frame it died in.
FATAL = "Fatal Python error:"
#   File "/long/path/pytorchsim-triton-compiler/pass_elf/widen_broadcast_loads.py", line 51 in _widen
FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+) in (\S+)')


def why_failed(out, rc):
    """The one line that says why, out of the compiler's whole transcript: never the stage banner,
    which reads the same for unrelated failures and carries wall-clock time that makes reports of
    an unchanged tree differ."""
    lines = out.splitlines()

    # A refusal that stopped refusing fails with no failing stage: say the headline.
    if "EXPECTED A REFUSAL" in out:
        return "EXPECTED A REFUSAL, BUT EVERY STAGE PASSED -- drop `expect`"

    stage, banner = "", -1
    for i, line in enumerate(lines):
        m = STAGE_FAIL.match(line)
        if m:
            stage, banner = m.group(1), i
            break

    # The exception the chain points at (CHAIN_IMPLICIT), read only where a traceback ends, since EXC
    # matches any class name.
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
        # No traceback: a refusal printed `type(e).__name__` bare (EXC_NAMED).
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
        # A StageError: the first indented line below the FAIL banner (above it, the stage before
        # prints its own). A run killed mid-stage has no banner, hence `banner >= 0`.
        said = [l.strip() for l in lines[banner + 1:] if l.startswith("  ") and l.strip()]
        if said and said[0].startswith(ExpectIrMismatch.HEADLINE):
            said = said[1:]
        why = said[0] if said else ""

    if not why:
        # A segfault has no traceback: faulthandler's stack, innermost frame first, names where.
        for i, line in enumerate(lines):
            if not line.startswith(FATAL):
                continue
            frame = next((FRAME.search(f) for f in lines[i + 1:i + 6]
                          if FRAME.search(f)), None)
            where = (f" in {os.path.basename(frame.group(1))}:{frame.group(2)} "
                     f"{frame.group(3)}" if frame else "")
            why = f"{line.strip()}{where}"
            break

    # A fatal signal is a prefix, not the diagnosis: an abort at shutdown often follows a traceback
    # that already named the fault.
    if rc is not None and rc < 0:
        try:
            import signal
            name = signal.Signals(-rc).name
        except (ValueError, ImportError):
            name = "?"
        killed = f"KILLED by signal {-rc} ({name})"
        if not why:
            return f"{killed} -- no diagnostic was printed"
        # `in <stage>` only when a banner named one.
        return f"{killed}{f' in {stage}' if stage else ''}: {why}"

    if not why:
        why = "failed with no diagnostic"

    # A verify failure carries its errors, so drift shows in the report.
    if stage == "verify":
        why = " ".join([why] + ERR.findall(out))
    return f"{stage}: {why}" if stage else why


#: Where sweeps put artifacts when nobody said. Kept stable so a single sweep
#: reuses it (see workdir_for) -- claim_workroot is what keeps a SECOND sweep off it.
DEFAULT_WORKROOT = "/tmp/torchsim-regress"


def default_jobs():
    """How many kernels to have in flight: the CPUs this process may run on (sched_getaffinity,
    not cpu_count, which ignores cpusets), less two. A worker holds about 250 MiB."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:                      # not Linux
        n = os.cpu_count() or 4
    return max(1, n - 2)


def claim_workroot(workroot):
    """Take `workroot` with a flock, or say so and hand back a private one: two sweeps in one
    workroot overwrite each other's artifacts mid-build and report failures no compiler made."""
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
    """Where this kernel's artifacts go: named by its path (basenames repeat across directories),
    reused across sweeps, with the last sweep's 08-transformed.mlir removed so nothing counts it."""
    wd = _recorded_in(workroot, path)
    stale = os.path.join(wd, "08-transformed.mlir")
    # Unlink with missing_ok, not check-then-remove: another sweep removing it between the two would
    # raise out of a worker and end the whole sweep.
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
    return f"FAIL  {rel:<62} {(bad or [r.stderr.strip()[-WHY_WIDTH:]])[0][:WHY_WIDTH]}"


def one(path, workroot, timeout, tog=False, timing=False):
    """Run one kernel to completion and reduce it to a single line. Nothing raises out of here:
    pool.map would re-raise it and lose the whole sweep."""
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
            # Every line carries the trip note, whatever the verdict.
            return f"PASS  {rel:<62} {how} -- {why[:80]}{trip_note}"
    if "NOT EXERCISED:" in out:
        why = out.split("NOT EXERCISED:", 1)[1].strip().splitlines()[0]
        return f"PASS  {rel:<62} not exercised -- {why[:80]}{trip_note}"
    if "THE BLOCK HAS LIFTED" in out:
        return f"FAIL  {rel:<62} THE BLOCK HAS LIFTED -- drop `expect`{trip_note}"
    if "THE GAP HAS CLOSED" in out:
        return f"FAIL  {rel:<62} THE GAP HAS CLOSED -- drop `expect`{trip_note}"
    if "ALL STAGES PASSED" in out:
        # A pass carries its errors too: drift from 1e-7 to 1e-3 shows only in the diff.
        return f"PASS  {rel:<62} {' '.join(ERR.findall(out))}{trip_note}"

    # Failed: the stage it died in and the reason it gave (why_failed), with the trip note.
    return f"FAIL  {rel:<62} {why_failed(out, r.returncode)[:WHY_WIDTH]}{trip_note}"


def sweep_breakdown(workroot, files):
    """The suite's wall clock, off every kernel's record (written whether or not it passed);
    rendering it is wallclock's."""
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
    # None, so "not given" (claim the default, or step aside) differs from a given path (obeyed).
    ap.add_argument("--workdir", default=None,
                    help=f"artifact root (default: {DEFAULT_WORKROOT}, or a "
                         f"private one if that is already held by a sweep)")
    ap.add_argument("--timeout", type=int, default=1800)
    # Off by default, so a default sweep does not check the TOG build at all.
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

    # Verdict and path only: the error column moves for reasons that are not regressions.
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
    # A missing entry means "deleted" only on a full run; on a subset it was not asked for.
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
