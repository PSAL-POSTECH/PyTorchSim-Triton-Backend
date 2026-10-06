#!/usr/bin/env python
"""Which branches in pytorchsim_triton_compiler/elf/passes are EVALUATED but never TAKEN, over the suite.

    ./tools/branch_firings.py                     trace, then report
    ./tools/branch_firings.py --report-only       reuse the traces on disk
    ./tools/branch_firings.py --min 500           only tests evaluated this often
    ./tools/branch_firings.py --where p07         one pass

WHAT IT IS LOOKING FOR, AND WHY THAT IS NOT COVERAGE. Dead code is evaluated
zero times and any linter finds it. This finds the other thing, and it is the
dominant defect in this backend's history:

    a pass MOVED, and an assumption that was true where it used to sit was
    never asked again -- so a branch runs thousands of times, is false every
    single time, and the code it guards silently does nothing.

Sixteen fixed defects had exactly that shape. `_splits_the_outer` excluded
expand/collapse by name and was evaluated 21929 times over 240 kernels and
passed 0. `_reshape_to_relayout` matched the op name "tensor.reshape".
`_flip_matmul` wanted a `tts.load` from a `tts.make_tptr`, five passes after the
last tts op stopped surviving. `widen_broadcast_loads` decided banking five
passes before the axis was known. None of them is visible by reading, because
each carries a plausible present-tense argument for itself. Only the count says
so.

    evaluated 0, taken 0        dead code, or a shape the suite never has
    evaluated N, taken 0        THIS. a question whose answer stopped changing
    evaluated N, else taken 0   the same thing from the other side

HOW IT MEASURES. `sys.settrace` with a global hook that returns a line tracer
only for frames whose file is under pytorchsim_triton_compiler/elf/passes, so everything else costs one
comparison per call. Each kernel runs in its own process exactly as tests/run.py
runs it and dumps its line counts at exit; the counts are summed. A branch's
test line and the first line of its body are both statements, so "how often was
this asked" and "how often was the answer yes" are two entries in the same
table.

WHAT IT CANNOT SEE. A short-circuit inside one `and` chain is one line, so a
sub-condition that is always false hides behind a test that sometimes passes.
Split the chain if a branch looks suspicious and the count says it fires.

READING THE OUTPUT. Bodies are classified, because most never-taken branches are
supposed to be never-taken: a `raise` that has not fired is a guard doing its
job, and a `continue` that never runs is a filter matching everything. The ones
worth reading are the ones classified WORK -- a branch that would have rewritten
something and never did.
"""
import argparse
import ast
import glob
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASSES = os.path.join(ROOT, "pytorchsim-triton-compiler", "passes")
DEFAULT_GLOBS = ("tests/cases/**/*.py",)

# The child half. Written out and run rather than imported, because it has to be
# the FIRST thing in a fresh interpreter -- sys.settrace only sees frames pushed
# after it is installed.
CHILD = '''
import atexit, json, os, runpy, sys, threading
ROOT, TARGET, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
counts = {}
def _line(frame, event, arg):
    if event == "line":
        k = frame.f_code.co_filename + ":" + str(frame.f_lineno)
        counts[k] = counts.get(k, 0) + 1
    return _line
def _call(frame, event, arg):
    return _line if frame.f_code.co_filename.startswith(TARGET) else None
def _dump():
    sys.settrace(None)
    try:
        with open(OUT, "w") as f:
            json.dump(counts, f)
    except Exception:
        pass
atexit.register(_dump)
sys.argv = ["pytorchsim-triton-compiler"] + sys.argv[4:]
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.settrace(_call); threading.settrace(_call)
try:
    runpy.run_path(os.path.join(ROOT, "pytorchsim-triton-compiler"), run_name="__main__")
except BaseException:
    pass
_dump()
'''


def trace_one(kernel, child, tracedir, workroot, timeout):
    tag = os.path.relpath(kernel, ROOT)[:-3].replace(os.sep, "__")
    out = os.path.join(tracedir, tag + ".json")
    try:
        subprocess.run([sys.executable, child, ROOT, PASSES, out,
                        os.path.relpath(kernel, ROOT),
                        "--workdir", os.path.join(workroot, tag)],
                       cwd=ROOT, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def merge(tracedir):
    counts, seen = {}, {}
    for p in sorted(glob.glob(os.path.join(tracedir, "*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        who = os.path.basename(p)[:-5]
        for k, v in d.items():
            counts[k] = counts.get(k, 0) + v
            seen.setdefault(k, []).append(who)
    return counts, seen


def body_kind(stmts):
    """WORK, or the name of the reason a never-taken body is fine."""
    flat = {type(s).__name__ for s in stmts}
    if flat <= {"Raise"}:
        return "raise"
    if flat <= {"Continue", "Pass", "Break"}:
        return "skip"
    if flat <= {"Return"} and all(
            s.value is None or isinstance(s.value, ast.Constant)
            or (isinstance(s.value, ast.Tuple)
                and all(isinstance(e, ast.Constant) for e in s.value.elts))
            for s in stmts):
        return "return-const"
    if len(stmts) <= 3 and any(isinstance(n, ast.Raise)
                               for s in stmts for n in ast.walk(s)):
        return "raise"
    return "WORK"


def report(counts, seen, minimum, where):
    def hits(f, ln):
        return counts.get(f + ":" + str(ln), 0)

    rows, never = [], []
    for path in sorted(glob.glob(os.path.join(PASSES, "*.py"))):
        rel = os.path.relpath(path, ROOT)
        if where and where not in rel:
            continue
        src = open(path).read()
        lines = src.splitlines()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                ev = hits(path, node.test.lineno)
                if ev >= minimum and hits(path, node.body[0].lineno) == 0:
                    rows.append((ev, body_kind(node.body), rel, node.test.lineno,
                                 lines[node.lineno - 1].strip()[:88]))
            elif isinstance(node, ast.FunctionDef):
                real = [s for s in node.body
                        if not (isinstance(s, ast.Expr)
                                and isinstance(s.value, ast.Constant))]
                if real and hits(path, real[0].lineno) == 0:
                    never.append((rel, node.lineno, node.name))

    print("=== EVALUATED, NEVER TAKEN, AND THE BODY DOES WORK "
          f"(>= {minimum} evaluations) ===")
    for ev, k, p, ln, txt in sorted(rows, key=lambda r: -r[0]):
        if k == "WORK":
            print(f"{ev:>9} eval / 0 taken   {p}:{ln}\n{'':>18}{txt}")
    print("\n=== SAME, BUT THE BODY ONLY RAISES / SKIPS / RETURNS A CONSTANT ===")
    print("    (a guard that has never fired is a guard doing its job)")
    for ev, k, p, ln, txt in sorted(rows, key=lambda r: -r[0]):
        if k != "WORK":
            print(f"{ev:>9} eval / 0 taken   {k:<12} {p}:{ln}  {txt[:60]}")
    print("\n=== FUNCTIONS NEVER ENTERED ===")
    for p, ln, name in never:
        print(f"    {p}:{ln}  {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="*", default=[])
    ap.add_argument("-j", "--jobs", type=int,
                    default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--min", type=int, default=100,
                    help="ignore tests evaluated fewer times than this")
    ap.add_argument("--where", help="substring of the pass file to restrict to")
    ap.add_argument("--tracedir", default="/tmp/torchsim-firings")
    ap.add_argument("--workdir", default="/tmp/torchsim-firings-work")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--report-only", action="store_true",
                    help="reuse whatever traces are already in --tracedir")
    args = ap.parse_args()

    if not args.report_only:
        files = []
        for g in (args.globs or DEFAULT_GLOBS):
            files += sorted(glob.glob(os.path.join(ROOT, g), recursive=True))
        files = sorted(set(files))
        if not files:
            sys.exit("no kernels matched")
        os.makedirs(args.tracedir, exist_ok=True)
        for stale in glob.glob(os.path.join(args.tracedir, "*.json")):
            os.remove(stale)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(CHILD)
            child = f.name
        print(f"tracing {len(files)} kernels ...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda p: trace_one(p, child, args.tracedir,
                                              args.workdir, args.timeout),
                          files))
        os.unlink(child)

    counts, seen = merge(args.tracedir)
    n = len(glob.glob(os.path.join(args.tracedir, "*.json")))
    print(f"{n} traces, {len(counts)} distinct lines in pytorchsim_triton_compiler/elf/passes reached\n",
          file=sys.stderr)
    report(counts, seen, args.min, args.where)


if __name__ == "__main__":
    main()
