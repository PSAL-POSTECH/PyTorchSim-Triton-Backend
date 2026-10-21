"""The stage machine: which stages there are and in what order, and the loop that runs them and says
how the run ended. What each stage does is in the module its STAGE_FN entry names:

  triton-compile   lowering/triton_compile.py        kernel -> 01..07-regularized (a worker process)
  torchsim-compile
    transform      lowering/passes/transform.py      p00..p07, lower-copies slot -> 08-transformed
    lower          lowering/passes/lower.py          the accelerator's slots, p11 -> 09-custom,
                                                       lower-dma slot, p13 -> 10-lowered
    binary         lowering/elf                      -> 11-<kernel>.elf, kernel.json
  tog (--tog)      lowering/tog                      09-custom -> tog.so, tile types, cycle ELF

Each is imported when its stage runs, so importing this module loads no MLIR -- `doctor` must work
on a toolchain that cannot.

A run a stage stops leaves contract/failure's FILE in the workdir, saying why by the type of what it
raised; any other run leaves none."""
import glob
import importlib
import os
import pathlib
import re
import time

from . import machine
from .contract import failure
from .metrics import quality, wallclock
from .stage import Ctx, SpadOverflow, SpillNotAllowed, StageError


# --------------------------------------------------------------------------
# The stages and their order
# --------------------------------------------------------------------------
STAGES = ["triton-compile", "torchsim-compile"]

STEPS_OF = {"torchsim-compile": ["transform", "lower", "binary"]}

TOG_STAGES = ["tog"]

LAST_STAGE = "binary"


def steps():
    """Every step in order -- a stage's own steps, or the stage itself."""
    return [s for st in STAGES for s in STEPS_OF.get(st, [st])]


def stage_of(step):
    """The stage `step` belongs to, or None for one on the --tog branch."""
    return next((st for st in STAGES if step in STEPS_OF.get(st, [st])), None)


def register_stage(name, fn, desc, last=False, tog=False, after=None):
    """Add a stage after the compiler's own -- right after `after` when one is named -- or, with
    `tog`, on the branch --tog adds after `binary`."""
    global LAST_STAGE
    if tog:
        if name not in TOG_STAGES:
            TOG_STAGES.append(name)
    elif name not in STAGES:
        STAGES.insert(STAGES.index(after) + 1 if after else len(STAGES), name)
    STAGE_FN[name] = fn
    STAGE_DESC[name] = desc
    if last:
        LAST_STAGE = name


# --------------------------------------------------------------------------
# What runs each stage
# --------------------------------------------------------------------------
STAGE_MODULE = {
    "triton-compile": ".lowering.triton_compile",
    "transform": ".lowering.passes.transform",
    "lower": ".lowering.passes.lower",
    "binary": ".lowering.elf",
    "tog": ".lowering.tog",
}


def stage_module(name):
    """The module that does stage `name`: its run(ctx), and recipe() for --list-stages."""
    return importlib.import_module(STAGE_MODULE[name], __package__)


def _run_of(name):
    def run(ctx):
        return stage_module(name).run(ctx)
    return run


STAGE_FN = {name: _run_of(name) for name in STAGE_MODULE}

STAGE_DESC = {
    "triton-compile": "triton kernel -> TTGIR -> linalg/memref/scf.for",
    "torchsim-compile": "tts/linalg -> RISC-V ELF: transform, lower, binary",
    "transform": "the tts-level backend: p00..p07",
    "lower": "matmul, vcix, xlu, descriptors -> DMA -> LLVM",
    "binary": "mlir-translate -> llc -> link RISC-V ELF",
    "tog": "the tile-operation-graph (TOG), the tile types and the cycle ELF (--tog)",
}


# --------------------------------------------------------------------------
# expect_ir: the counts a kernel file declares its artifacts hold. A dotted name is counted in
# the parsed artifact: the ops of that name plus the ops carrying an attribute of that name. Any
# other key (a type or an attribute as printed, an assembly mnemonic) is a token of the text.
# --------------------------------------------------------------------------
_WRITTEN_BY = {**{n: "triton-compile" for n in ("01", "02", "03", "04", "05", "06", "07")},
               "08": "transform", "09": "lower", "10": "lower", "11": "binary"}


def _expect_ir(spec):
    return dict(spec.expect_ir or {})


def _clear_expect_ir(ctx, todo):
    """Remove the declared artifacts a step in `todo` will write again, so a step that dies before
    writing one is not checked against the previous run's."""
    for fname in _expect_ir(ctx.spec):
        if _WRITTEN_BY.get(fname[:2], todo[0]) in todo:
            pathlib.Path(ctx.path(fname)).unlink(missing_ok=True)


def _the_artifact_named(workdir, fname):
    path = os.path.join(workdir, fname)
    if os.path.isfile(path):
        return path
    m = re.match(r"(.*/)?\d+-(.+)$", fname)
    if m is None:
        return None
    got = glob.glob(os.path.join(workdir, m.group(1) or "", "*-" + m.group(2)))
    return got[0] if len(got) == 1 else None


_OP_NAME = re.compile(r"[a-z_]\w*(\.[a-z_]\w*)+")


def _count_token(text, key):
    spellings = key if isinstance(key, tuple) else (key,)
    return sum(len(re.findall(r"(?<![\w.])" + re.escape(s) + r"(?![\w.])", text))
               for s in spellings)


def _op_counts_of(path, stage):
    """Name -> count of the ops, then of the attributes, in the `.mlir` artifact at `path`: parsed
    here when this process's MLIR reads it (every artifact but the Triton frontend's own 01 and
    02), else by that Triton, which counts the ops."""
    from . import toolchain
    from .lowering.passes.lib_ir import attr_counts, op_counts
    toolchain.ensure_bindings()
    from mlir.ir import Context, MLIRError, Module
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, open(path) as f:
        try:
            module = Module.parse(f.read())
            return dict(op_counts(module.operation)), dict(attr_counts(module.operation))
        except MLIRError:
            pass
    import json
    import subprocess
    import sys
    from .lowering import triton_compile
    got = subprocess.run([sys.executable, triton_compile.__file__, "--count-ops", path],
                         env=toolchain.triton_env(), capture_output=True, text=True)
    if got.returncode != 0:
        raise StageError(stage, f"{path} could not be parsed to count its ops",
                         stderr=got.stderr, artifact=path)
    return json.loads(got.stdout)[path], {}


def _count(text, parsed, key):
    spellings = key if isinstance(key, tuple) else (key,)
    if parsed is not None and all(_OP_NAME.fullmatch(s) for s in spellings):
        ops, attrs = parsed
        return sum(ops.get(s, 0) + attrs.get(s, 0) for s in spellings)
    return _count_token(text, key)


def _spell(op):
    return " or ".join(op) if isinstance(op, tuple) else op


def _check_expect_ir(ctx, stage, done):
    if os.environ.get("TORCHSIM_COMPILE_IGNORE_EXPECT_IR"):
        return
    for fname, counts in _expect_ir(ctx.spec).items():
        if fname in done:
            continue
        path = _the_artifact_named(ctx.workdir, fname)
        if path is None:
            continue
        done.add(fname)
        with open(path) as f:
            text = f.read()
        parsed = _op_counts_of(path, stage) if path.endswith(".mlir") else None
        wrong = [(key, want, got) for key, want in counts.items()
                 for got in [_count(text, parsed, key)] if got != want]
        if wrong:
            raise StageError(stage, "%s does not say what the kernel declares: %s"
                             % (fname, ", ".join(
                                 f"{_spell(op)} is {got}, expect_ir says {want}"
                                 for op, want, got in wrong)),
                             artifact=path)
        ctx.log(f"[expect_ir] {fname}: {len(counts)} count(s) as declared")


# --------------------------------------------------------------------------
# failure.json: why a run stopped, for a caller to act on
# --------------------------------------------------------------------------
FAILURE_KINDS = [
    (SpadOverflow, failure.SPAD_OVERFLOW, ("usage", "budget")),
    (SpillNotAllowed, failure.SPILL_NOT_ALLOWED, ()),
]


def _write_failure(ctx, name, e):
    kind, facts = next(((k, f) for t, k, f in FAILURE_KINDS if isinstance(e, t)),
                       (failure.ERROR, ()))
    said = str(e) if isinstance(e, StageError) else f"{type(e).__name__}: {e}"
    held = {fact: getattr(e, fact, None) for fact in ("cmd", "artifact")}
    failure.write(ctx.workdir, getattr(e, "stage", name), kind, said,
                  **{fact: getattr(e, fact) for fact in facts},
                  **{fact: str(v) for fact, v in held.items() if v is not None})


# --------------------------------------------------------------------------
# Driving the stages
# --------------------------------------------------------------------------
def run(spec, kernel_path, workdir, from_stage="triton-compile", to_stage=None,
        verbose=False, breakdown=False, tog=False):
    """Drive the stages, then leave `timing.json` in the workdir either way."""
    wallclock.PROF.label = spec.name
    try:
        return _run_stages(spec, kernel_path, workdir, from_stage=from_stage,
                           to_stage=to_stage, verbose=verbose, tog=tog)
    finally:
        wallclock.PROF.write(os.path.join(os.path.abspath(workdir), wallclock.FILENAME))
        if breakdown or wallclock.want_report():
            print()
            print(wallclock.PROF.report())


def _plan(from_stage, to_stage, tog):
    """The steps this run takes, and whether it is the whole chain."""
    chain = steps()
    first = STEPS_OF.get(from_stage, [from_stage])[0]
    last = STEPS_OF.get(to_stage, [to_stage])[-1] if to_stage else chain[-1]
    todo = chain[chain.index(first):chain.index(last) + 1]
    if tog and "binary" in todo:
        todo = todo[:todo.index("binary") + 1] + TOG_STAGES + todo[todo.index("binary") + 1:]
    return todo, first == chain[0] and to_stage is None


def _run_stages(spec, kernel_path, workdir, from_stage="triton-compile", to_stage=None,
                verbose=False, tog=False):
    ctx = Ctx(spec, kernel_path, workdir, verbose=verbose)
    open(ctx.log_path, "w").close()
    failure.clear(ctx.workdir)

    todo, whole = _plan(from_stage, to_stage, tog)
    _clear_expect_ir(ctx, todo)
    if "transform" in todo:
        pathlib.Path(ctx.path(quality.FILENAME)).unlink(missing_ok=True)

    print(f"kernel   : {spec.name}   ({kernel_path})")
    print(f"workdir  : {ctx.workdir}")
    print(f"stages   : {' -> '.join(todo)}")
    print()

    checked_ir = set()
    # A retry from binary builds transform and lower again, and expect_ir pins the build kept: a
    # count that differs before binary has passed waits for it (stage, label, error, seconds).
    held = None
    i = 0
    while i < len(todo):
        name = todo[i]
        i += 1
        st = stage_of(name)
        shown_name = name if st in (None, name) else f"{st}:{name}"
        label = (f"[{STAGES.index(st) + 1}/{len(STAGES)}] {shown_name:<23}"
                 if st else f"[  extra  ] {shown_name:<23}")
        t0 = time.time()
        try:
            with wallclock.PROF.stage(name):
                artifact = STAGE_FN[name](ctx)
                try:
                    _check_expect_ir(ctx, name, checked_ir)
                except StageError as e:
                    if name not in ("transform", "lower") or "binary" not in todo[i:]:
                        raise
                    held = (name, label, e, time.time() - t0)
        except StageError as e:
            dt = time.time() - t0
            narrowed = _narrow_for(ctx, e) if isinstance(e, SpillNotAllowed) else None
            if narrowed:
                again, narrowed = narrowed
                print(f"{label} RETRY ({dt:.1f}s)  {e}, which we do not allow; "
                      f"again from {again} with {narrowed}")
                ctx.log(f"[retry] {e}; again from {again} with {narrowed}")
                todo, i = _rewind_to(todo, i - 1, again)
                _clear_expect_ir(ctx, todo[i:])
                checked_ir.clear()
                held = None
                continue
            if held is not None:
                return _report_held(ctx, spec, kernel_path, held)
            _write_failure(ctx, name, e)
            if _stops_as_declared(spec, name):
                return _report_stopped(spec, label, name, dt, str(e))
            _report_failure(ctx, spec, kernel_path, label, name, dt, e)
            return False
        except Exception as e:  # noqa: BLE001
            dt = time.time() - t0
            if held is not None:
                return _report_held(ctx, spec, kernel_path, held)
            _write_failure(ctx, name, e)
            if _stops_as_declared(spec, name):
                return _report_stopped(spec, label, name, dt,
                                       f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
            _report_failure(ctx, spec, kernel_path, label, name, dt, None)
            return False

        dt = time.time() - t0
        shown = os.path.relpath(str(artifact), ctx.workdir) if artifact else ""
        print(f"{label} ok    ({dt:.1f}s)  {STAGE_DESC[name]:<45} {shown}")
        if name == "binary" and held is not None:
            return _report_held(ctx, spec, kernel_path, held)

    return _report_finished(ctx, spec, todo, whole, checked_ir)


def _report_held(ctx, spec, kernel_path, held):
    """The expect_ir count that differed in the build binary kept, reported as its stage's failure."""
    name, label, e, dt = held
    _write_failure(ctx, name, e)
    if _stops_as_declared(spec, name):
        return _report_stopped(spec, label, name, dt, str(e))
    _report_failure(ctx, spec, kernel_path, label, name, dt, e)
    return False


def _width_of(ctx, place):
    kind, n = place
    return ctx.loop_lmul.get(n, ctx.lmul) if kind == "loop" else ctx.dot_widths.get(n, 1)


def _set_width(ctx, place, width):
    kind, n = place
    (ctx.loop_lmul if kind == "loop" else ctx.dot_lmul)[n] = width


def _narrow_for(ctx, e):
    """One step of the retry, on the places llc named for a spill: (the stage to build again from,
    what was done), or None when nothing can be. The place narrowed last is judged first: if its own
    spills did not drop, its width goes back and it is kept (spill_kept). Then the place with the
    most spills that a narrower width may relieve is halved, alone. With none, a kernel the stack
    refuses (no window, no room) narrows its widest streams, then the kernel's LMUL. A loop's width
    is legalize's (transform), a stream's the matmul lowering's (lower)."""
    from .lowering.elf.build import narrowable
    said, stages = [], set()
    places = {("loop", n): k for n, k in e.loop_spills.items()}
    places.update({("dot", n): k for n, k in e.dot_spills.items()})
    if ctx.spill_last is not None:
        place, width, before = ctx.spill_last
        ctx.spill_last = None
        if places.get(place, 0) >= before:
            _set_width(ctx, place, width)
            ctx.spill_kept.add(place)
            stages.add("transform" if place[0] == "loop" else "lower")
            said.append(f"LMUL {width} again in {place[0]} {place[1]} (narrower relieved none of its "
                        f"{before})")
    relieve = narrowable(ctx, e.loop_spills, e.dot_spills)
    if relieve:
        place = max(relieve, key=lambda p: (relieve[p], p))
        width = _width_of(ctx, place)
        _set_width(ctx, place, width // 2)
        ctx.spill_last = (place, width, relieve[place])
        stages.add("transform" if place[0] == "loop" else "lower")
        said.append(f"LMUL {width // 2} in {place[0]} {place[1]} ({relieve[place]} spill(s) and reload(s))")
    if said:
        return ("transform" if "transform" in stages else "lower"), ", ".join(said)
    widest = max(ctx.dot_widths.values(), default=1)
    if widest > 1:
        dots = [n for n, w in ctx.dot_widths.items() if w == widest]
        for n in dots:
            ctx.dot_lmul[n] = widest // 2
        return "lower", f"the stack refuses them: LMUL {widest // 2} in the widest streams {dots}"
    if ctx.lmul > 1:
        ctx.lmul //= 2
        return "transform", f"LMUL {ctx.lmul}"
    return None


def _rewind_to(todo, at, stage):
    """`todo` and the position to go on from: the last `stage` before `at`, or `stage` and the
    stages after it inserted at `at` when it did not run."""
    if stage in todo[:at]:
        return todo, todo.index(stage)
    after = ["transform", "lower"]
    return todo[:at] + after[after.index(stage):] + todo[at:], at


def _stops_as_declared(spec, stage):
    """A file that says it is refused or blocked may stop at any stage short of the last; one that
    says it is open may stop anywhere."""
    return (spec.expect in ("refuse", "blocked") and stage != LAST_STAGE) or spec.expect == "open"


def _report_stopped(spec, label, name, dt, said):
    word = ("REFUSED" if spec.expect == "refuse"
            else "BLOCKED" if spec.expect == "blocked" else "OPEN")
    print(f"{label} {word} ({dt:.1f}s)  {STAGE_DESC[name]}")
    print()
    print(f"  {said}")
    print()
    print(f"{word} AS EXPECTED: {spec.expect_why}")
    return True


def _report_failure(ctx, spec, kernel_path, label, name, dt, e):
    """`e` is the StageError, or None for any other exception, whose traceback is printed."""
    print(f"{label} FAIL  ({dt:.1f}s)  {STAGE_DESC[name]}")
    print()
    if e is None:
        import traceback
        traceback.print_exc()
    else:
        if spec.expect in ("refuse", "blocked"):
            print(f"  THIS FILE SAYS {spec.expect}, AND NOTHING STOPPED: "
                  f"{spec.expect_why}")
            print("  every stage produced a program and the ANSWER is what "
                  "failed, which is a wrong number, not a refusal.")
            print()
        print(f"  {e}")
        if e.cmd:
            print(f"  command : {e.cmd}")
        if e.artifact:
            print(f"  input   : {e.artifact}")
        if e.stderr:
            print("  stderr  :")
            for line in e.stderr.strip().splitlines()[:20]:
                print(f"    {line}")
    print()
    print(f"  full log: {ctx.log_path}")
    resume = ("transform" if ctx.lmul < machine.ISA_MAX_LMUL
              and name in STEPS_OF["torchsim-compile"] else name)
    print(f"  resume  : pytorchsim-triton-compiler {kernel_path} --from {resume}")


def _report_finished(ctx, spec, todo, whole, checked_ir):
    """Every step passed: say whether that is what the file declares."""
    missing = [f for f in _expect_ir(spec) if f not in checked_ir] if whole else []
    missing = [f for f in missing if _the_artifact_named(ctx.workdir, f) is None]
    if missing:
        print()
        print(f"  expect_ir names {', '.join(missing)}, which this run never "
              f"wrote. Name the artifact the stage prints beside itself.")
        return False

    print()
    if whole and spec.expect_stage and spec.expect_stage not in todo:
        print(f"NOT EXERCISED: this file pins the {spec.expect_stage} stage, which this run "
              f"did not reach (--{'timing' if spec.expect_stage == 'timing' else 'tog'})")
        print(f"  every stage it ran passed; expect={spec.expect!r} was not tested")
        return True
    if spec.expect == "open":
        print("THE GAP HAS CLOSED, AND EVERY STAGE PASSED")
        print(f"  the file says: {spec.expect_why}")
        print("  drop `expect` -- this is an ordinary kernel now.")
        print("ALL STAGES PASSED")
        return True
    if spec.expect == "blocked":
        print("THE BLOCK HAS LIFTED, AND EVERY STAGE PASSED")
        print(f"  the file says: {spec.expect_why}")
        print("  drop `expect` -- this is an ordinary kernel now.")
        print("ALL STAGES PASSED")
        return True
    if spec.expect == "refuse":
        print("EXPECTED A REFUSAL, BUT EVERY STAGE PASSED")
        print(f"  the file says: {spec.expect_why}")
        print("  if that is no longer true, drop `expect` and give it a real "
              "reference.")
        return False
    print("ALL STAGES PASSED")
    return True
