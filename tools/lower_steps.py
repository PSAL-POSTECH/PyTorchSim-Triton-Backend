"""Lower a kernel through triton-shared ONE PASS AT A TIME.

Stage 3 runs seven steps in one invocation, which shows the input and the output
and nothing in between -- and in between is where every interesting decision is
made: whether a pointer stays structured, whether MaskAnalysis can parse a mask,
which converter claims the op. `pytorchsim-triton-compiler --to triton-shared` gives the
whole of it; this gives you each step.

Writes one file per pass under <workdir>/<kernel>/triton-shared-detail/, so a
diff between adjacent files is that pass's entire effect. Kept in its own
subdirectory so it never collides with the numbered stage outputs run.py writes
next to it (01-ttir.mlir, 03-ttshared.mlir, ...).

    python tools/lower_steps.py tests/cases/ptr_gathered.py
    python tools/lower_steps.py tests/cases/*.py --summary
    python tools/lower_steps.py tests/cases/smoke/gemm.py --debug     # PtrAnalysis remarks

To run a pipeline that is NOT a prefix of the standard one -- to see what an
individual pass is responsible for by taking it away -- use --skip or --passes:

    # what does this pass actually do? take it away and see
    python tools/lower_steps.py tests/cases/smoke/add.py --skip triton-to-ptr

    # an exact list, in this order
    python tools/lower_steps.py tests/cases/smoke/add.py \\
        --passes triton-to-structured,triton-arith-to-linalg

Both write a single `custom.mlir` instead of the numbered series, and print
which tt/tts ops survived. Passes are just flags to triton-shared-opt and it
applies them in the order given, so any subset is legal to *run*; whether the
result is meaningful is a different question -- dropping triton-to-structured
leaves raw tt.addptr chains that nothing downstream consumes.

Doing it by hand is two commands -- stages 1-2 for the IR triton-shared is
actually handed, then triton-shared-opt with a prefix of the flag list:

    pytorchsim-triton-compiler tests/cases/ptr_gathered.py --to triton-shared
    $TORCHSIM_COMPILE_TRITON_SHARED_OPT --triton-to-structured \\
        work/ptr_gathered_kernel/02-ttclean.mlir -o /tmp/01.mlir

Use `02-ttclean.mlir`, not `01-ttir.mlir`: the triton-shared stage first takes
out what triton-shared cannot read at all (`ttg.barrier`) or has no use for
(`llvm.intr.assume`), so `01-ttir.mlir` is an input the pipeline never gives it
and a kernel with a `tl.assume` dies on the parse.

Add `--debug-only=triton-ptr-analysis` to that second command to see WHY a
pointer or a mask was rejected -- that is how the remarks quoted in the
tests/cases/ docstrings were obtained.

See the README's "triton-shared 내부" section for what each step does and
which source file implements it.
"""
import argparse
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402
from pytorchsim_triton_compiler.pipeline import TTSHARED_PIPELINE  # noqa: E402
from pytorchsim_triton_compiler.contract.kernel_spec import load_spec  # noqa: E402

# IMPORTED, NOT RESTATED. pipeline.py's stage 3 is the authority, so a step
# added or an option changed there reaches this tool with nothing to keep in
# sync -- the drift a second copy causes is silent, which is why there is one.
PIPELINE = TTSHARED_PIPELINE

# Which steps get a numbered snapshot: the ones that decide POINTERS. The rest
# are type cleanup and casts, so they run without a file of their own.
SNAPSHOTS = {
    "triton-to-structured":   "01-structured",
    "triton-to-unstructured": "02-unstructured",
    "triton-arith-to-linalg": "03-arith-linalg",
    "triton-ptr-to-memref":   "04-ptr-memref",
}

ALL_PASSES = [name for name, _ in PIPELINE]

# What to count in --summary. These are the ops whose presence or absence says
# which lowering path a kernel took.
OPS = ("tt.load", "tt.store",
       "tts.make_tptr", "tts.make_gather_scatter_tptr",
       "tts.load", "tts.store", "tts.gather", "tts.scatter")

# Sub-directory of the kernel's workdir. Separate from run.py's own numbered
# outputs so the two never overwrite each other.
DETAIL = "triton-shared-detail"


def find_kernels(root):
    """Every kernel file under tests/cases/, however deeply nested.

    Recursive on purpose: tests/cases/ is organised into topic
    subdirectories and gets reorganised as the set grows, and a hardcoded list
    of globs silently stops finding whole directories when that happens. A file
    counts as a kernel if it defines a module-level SPEC.
    """
    out = []
    for p in sorted(glob.glob(os.path.join(root, "tests", "cases", "**", "*.py"),
                              recursive=True)):
        if "__pycache__" in p:
            continue
        try:
            load_spec(p)
        except Exception:
            continue          # helper module, not a kernel
        out.append(p)
    return out


def emit_ttclean(kernel, workdir, verbose):
    """`02-ttclean.mlir`, the exact input triton-shared-opt is handed. The stage
    writes it BEFORE running triton-shared, so a kernel that crashes there -- the
    one this tool is for -- still leaves it; its absence is the only failure."""
    kdir = os.path.join(workdir, load_spec(kernel).name)
    got = os.path.join(kdir, "02-ttclean.mlir")
    if os.path.exists(got):
        os.remove(got)
    cmd = [sys.executable, os.path.join(ROOT, "pytorchsim-triton-compiler"), kernel,
           "--to", "triton-shared", "--workdir", kdir]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if not os.path.exists(got):
        print(f"  no triton-shared input written\n{p.stdout}{p.stderr}")
        return None
    return got


def run_stages(ttclean, outdir, debug):
    """Walk the whole pipeline, snapshotting the passes that decide pointers."""
    os.makedirs(outdir, exist_ok=True)
    flags = []
    final = None
    for name, add in PIPELINE:
        flags = flags + add
        stem = SNAPSHOTS.get(name)
        # Every step runs; only the interesting ones get their own file. The
        # last is written too -- that file IS stage 3's 03-ttshared.mlir, up to
        # the generic printer stage 3 adds and this does not.
        is_last = name == PIPELINE[-1][0]
        if stem is None and not is_last:
            continue
        out = os.path.join(outdir, f"{stem}.mlir" if stem else "98-full.mlir")
        cmd = [toolchain.TRITON_SHARED_OPT, *flags]
        if debug:
            cmd.append("--debug-only=triton-ptr-analysis")
        cmd += [ttclean, "-o", out]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if p.returncode != 0:
            with open(out, "w") as f:
                f.write(f"// PASS FAILED at {name}\n// {p.stderr}\n")
            print(f"  {name:<26} FAILED")
        elif debug and p.stderr.strip():
            for line in p.stderr.splitlines():
                if "remark" in line or "Failed" in line:
                    print(f"  {name:<26} {line.strip()[:100]}")
        if is_last:
            final = out
    return final


def run_custom(ttclean, outdir, passes, debug):
    """One run with an arbitrary pass list. Writes custom.mlir."""
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "custom.mlir")
    flags = [f for name in passes for f in dict(PIPELINE)[name]]
    cmd = [toolchain.TRITON_SHARED_OPT] + flags
    if debug:
        cmd.append("--debug-only=triton-ptr-analysis")
    cmd += [ttclean, "-o", out]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    print("  " + " ".join(flags))
    if p.returncode != 0:
        with open(out, "w") as f:
            f.write(f"// PASS PIPELINE FAILED\n// {p.stderr}\n")
        print(f"  FAILED: {p.stderr.strip().splitlines()[0][:100]}")
        return out
    if debug:
        for line in p.stderr.splitlines():
            if "remark" in line or "Failed" in line:
                print(f"    {line.strip()[:110]}")
    # What survived is the interesting part of a custom run.
    left = {}
    with open(out) as f:
        text = f.read()
    for o in OPS:
        n = text.count(o)
        if n:
            left[o] = n
    print("  survives: " + (", ".join(f"{k}={v}" for k, v in left.items()) or "(no tt/tts ops)"))
    return out


def count(path, needle):
    try:
        with open(path) as f:
            return f.read().count(needle)
    except OSError:
        return 0


def summary(done):
    print()
    print("op counts after each pass (pipeline is cumulative)")
    print("=" * 112)
    for name, outdir, ttclean in done:
        print(f"\n{name}")
        print("  " + f"{'stage':<22}" + "".join(f"{o.replace('tts.', 'ts.'):>13}" for o in OPS))
        rows = [("00-ttclean", ttclean)] + [
            (stem, os.path.join(outdir, f"{stem}.mlir"))
            for name, _ in PIPELINE if (stem := SNAPSHOTS.get(name))]
        for stem, path in rows:
            row = "".join(f"{count(path, o):>13}" for o in OPS)
            print("  " + f"{stem:<22}" + row)


def main():
    ap = argparse.ArgumentParser(
        description="Lower kernels through triton-shared one pass at a time.")
    ap.add_argument("kernels", nargs="*",
                    help="kernel files (default: every kernel under tests/cases/)")
    ap.add_argument("--workdir", default=os.path.join(ROOT, "work"))
    ap.add_argument("--summary", action="store_true",
                    help="print the op-count table")
    ap.add_argument("--debug", action="store_true",
                    help="pass --debug-only=triton-ptr-analysis and echo remarks")
    ap.add_argument("--skip", default="",
                    help="comma-separated step names to drop from the full "
                         "pipeline, e.g. --skip triton-to-ptr")
    ap.add_argument("--passes", default="",
                    help="comma-separated pass names to run, in this order "
                         "(overrides --skip)")
    ap.add_argument("--list-passes", action="store_true",
                    help="print the full pipeline and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_passes:
        for p in ALL_PASSES:
            print(p)
        return

    custom = None
    if args.passes:
        custom = [p.strip().lstrip("-") for p in args.passes.split(",") if p.strip()]
        bad = [p for p in custom if p not in ALL_PASSES]
        if bad:
            raise SystemExit(f"unknown pass(es): {chr(44).join(bad)}\nknown: {chr(44).join(ALL_PASSES)}")
    elif args.skip:
        drop = {p.strip().lstrip("-") for p in args.skip.split(",") if p.strip()}
        unknown = drop - set(ALL_PASSES)
        if unknown:
            raise SystemExit(f"unknown pass(es): {', '.join(sorted(unknown))}\n"
                             f"known: {', '.join(ALL_PASSES)}")
        custom = [p for p in ALL_PASSES if p not in drop]

    kernels = args.kernels or find_kernels(ROOT)
    done = []
    for k in kernels:
        name = os.path.basename(k)[:-3]
        print(f"{name}")
        ttclean = emit_ttclean(k, args.workdir, args.verbose)
        if ttclean is None:
            continue
        outdir = os.path.join(os.path.dirname(ttclean), DETAIL)
        if custom is not None:
            out = run_custom(ttclean, outdir, custom, args.debug)
            print(f"  -> {os.path.relpath(out, ROOT)}")
            continue
        run_stages(ttclean, outdir, args.debug)
        done.append((name, outdir, ttclean))
        print(f"  -> {os.path.relpath(outdir, ROOT)}/")

    if args.summary:
        summary(done)


if __name__ == "__main__":
    main()
