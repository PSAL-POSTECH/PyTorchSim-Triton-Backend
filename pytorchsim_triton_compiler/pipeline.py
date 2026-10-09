"""The stage machine: which stages there are and in what order, what each one does, the checks it
holds its own output to, and the loop that runs them and says how the run ended."""
import glob
import os
import pathlib
import re
import sys
import time

from . import machine
from . import toolchain
from .metrics import wallclock
from .metrics import quality
from .lowering.elf import build as elf_build
from .stage import Ctx, SpillNotAllowed, StageError, run_tool


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
# Stage triton-compile
# --------------------------------------------------------------------------
def stage_triton_compile(ctx):
    """The only stage that imports triton, and it does so in its OWN process: libtriton carries an
    MLIR of its own, and beside the bindings' one the dialect registries cross. The worker takes the
    kernel from its SPEC to 07-regularized.mlir: Triton IR (01), the no-ops nothing below parses
    dropped (02), TTGIR with every layout this machine fixes anchored in one pass and propagated once
    (03, 04, 05), lowered to linalg/memref with every encoding carried down (06), then its layouts
    regularized -- every encoding the LinearLayout it is, unit extents folded (07)."""
    names = ("01-ttir.mlir", "02-ttclean.mlir", "03-ttgir.mlir", "04-anchored.mlir",
             "05-layouts.mlir", "06-linalg.mlir", "07-regularized.mlir")
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lowering", "triton_compile.py")
    run_tool(ctx, "triton-compile", [sys.executable, worker, ctx.kernel_path, machine.VECTORLANE_SIZE,
                                     *(ctx.path(n) for n in names)], env=toolchain.triton_env())
    _check_layouts_carried(ctx, ctx.path("06-linalg.mlir"))
    return ctx.path("07-regularized.mlir")


def _check_layouts_carried(ctx, path):
    _ensure_bindings()
    from mlir.ir import Context, Location, Module, RankedTensorType
    from .lowering.passes.lib_ir import is_reshape_shape, name_of, op_of, walk
    with open(path) as f:
        text = f.read()
    ctx_ = Context()
    ctx_.allow_unregistered_dialects = True
    bare = []
    with ctx_, Location.unknown():
        module = Module.parse(text)
        for raw in walk(module.operation):
            op = op_of(raw)
            values = list(op.results) + [a for r in raw.regions for b in r.blocks for a in b.arguments]
            for v in values:
                if not isinstance(v.type, RankedTensorType):
                    continue
                ty = RankedTensorType(v.type)
                elem = ty.element_type
                if (ty.rank == 0 or ty.encoding is not None or "ptr" in str(elem)
                        or is_reshape_shape(v)):
                    continue
                bare.append("%s : %s" % (name_of(op), ty))
    if bare:
        raise StageError("triton-compile", "%s: %d tile(s) without a layout encoding, first %s"
                         % (os.path.basename(path), len(bare), "; ".join(bare[:3])),
                         artifact=path)


# --------------------------------------------------------------------------
# Stage torchsim-compile, step transform: p00..p07 -> 08-transformed.mlir
# --------------------------------------------------------------------------
SUPPORTED_DTYPES = frozenset((
    "i1", "i8", "i16", "i32", "i64",
    "f16", "f32", "f64", "bf16",
))

_ZERO_STRIDE = re.compile(r"strided<\[([^\]]*)\]")


def stage_transform(ctx):
    src, out = ctx.path("07-regularized.mlir"), ctx.path("08-transformed.mlir")
    _ensure_bindings()
    from mlir.ir import Context, Module, Location
    with wallclock.PROF.step("import pytorchsim_triton_compiler.lowering.passes"):
        from .lowering import passes  # noqa: F401
    from .lowering.passes import (PHASES, PIPELINE, pass_name,
                                  p00_convert_unsupported_ops,
                                  p01_convert_accesses_to_transfer,
                                  p02_materialize_layouts, p03_fuse, p04_bufferize,
                                  p05_vectorize, p06_legalize, p07_lower_copies)

    with open(src) as f:
        text = f.read()

    arg_numels = {i: a.numel for i, a in enumerate(ctx.spec.args)}
    mlir_ctx = Context()
    mlir_ctx.allow_unregistered_dialects = True
    pass_dir = _fresh_dir(ctx.path("08-passes"))
    ran = []
    mark = [time.perf_counter()]

    def dump(name, module):
        wallclock.PROF.add_pass(name, time.perf_counter() - mark[0])
        ran.append(name)
        path = os.path.join(pass_dir, f"{len(ran):02d}-{name}.mlir")
        with open(path, "w") as f:
            f.write(str(module))
        _check_no_zero_stride(name, module, path)
        _check_forbids(ctx, name, module, path)
        mark[0] = time.perf_counter()

    with mlir_ctx, Location.unknown():
        module = Module.parse(text)
        dump("input", module)
        _refuse_unsupported_dtypes(module)
        p00_convert_unsupported_ops.run(module, scan_base_length=machine.VECTORLANE_SIZE)
        dump(pass_name(p00_convert_unsupported_ops), module)
        p01_convert_accesses_to_transfer.run(module, arg_numels=arg_numels)
        dump(pass_name(p01_convert_accesses_to_transfer), module)
        p02_materialize_layouts.run(module)
        dump(pass_name(p02_materialize_layouts), module)
        p03_fuse.run(module)
        dump(pass_name(p03_fuse), module)
        p04_bufferize.run(module)
        dump(pass_name(p04_bufferize), module)
        p05_vectorize.run(module, lanes=machine.VECTORLANE_SIZE)
        dump(pass_name(p05_vectorize), module)
        p06_legalize.run(module, lanes=machine.VECTORLANE_SIZE,
                         vlen=machine.VLEN_BITS, lmul=ctx.lmul)
        dump(pass_name(p06_legalize), module)
        p07_lower_copies.run(module, lanes=machine.VECTORLANE_SIZE)
        dump(pass_name(p07_lower_copies), module)
        counts = quality.count(module)
        one_lane = quality.count_one_lane(module)
        result = str(module)
    with open(out, "w") as f:
        f.write(result)
    if ran[1:] != PIPELINE:
        raise StageError(
            "transform",
            f"transform ran {ran[1:]} but pytorchsim_triton_compiler.lowering.passes.PHASES says {PIPELINE}",
            artifact=src)
    ctx.log(f"[transform] per-pass IR in {pass_dir}/")
    for title, names in PHASES:
        ctx.log(f"[transform] {title}: " + " -> ".join(names))
    quality.write(ctx.workdir, counts, one_lane)
    xlu = counts["xlu"]
    ctx.log(f"[transform] DMA: {counts['mvin']} MVIN, {counts['mvout']} MVOUT; "
            f"moves between lanes: {counts['round_trips']} through DRAM, "
            f"{sum(xlu.values())} through the XLU"
            + (" (" + ", ".join(f"{k} {n}" for k, n in sorted(xlu.items()))
               + ")" if xlu else ""))
    return out


def _refuse_unsupported_dtypes(module):
    from mlir.ir import FloatType, IntegerType
    from .lowering.passes.lib_ir import elem_type, funcs, walk
    for func in funcs(module):
        for op in walk(func.operation):
            for v in list(op.operands) + list(op.results):
                t = elem_type(v)
                for cls in (IntegerType, FloatType):
                    try:
                        cls(t)
                    except (TypeError, ValueError):
                        continue
                    if str(t) not in SUPPORTED_DTYPES:
                        raise NotImplementedError(
                            f"{t} is not an element type this machine runs -- "
                            f"SUPPORTED_DTYPES lists what was measured, and "
                            f"nothing lowers the rest: {op}")


def _check_no_zero_stride(name, module, artifact):
    for strides in _ZERO_STRIDE.findall(str(module)):
        if any(s.strip() == "0" for s in strides.split(",")):
            raise StageError(
                "transform",
                f"{name} printed a memref type with a zero stride: "
                f"strided<[{strides}]>. Our own MLIR parses that back and the "
                f"one reading the file may not -- PyTorchSim's says 'strides "
                f"must not be zero' -- so the refusal lands a stage later in "
                f"another repository. A unit axis does not move, which is what "
                f"the zero means; write 1, which names the same address. The IR "
                f"it produced is {artifact}.",
                artifact=artifact)


def _check_forbids(ctx, name, module, artifact):
    from .lowering.passes import forbidden, forbids
    left = forbidden(name, module, ctx.spec.extra)
    if not left:
        return
    patterns, _, source = forbids(name)
    shown = ", ".join(f"{n} x{c}" for n, c, _ in left)
    total = sum(c for _, c, _ in left)
    said = [p for p in patterns if not p.startswith("-")]
    spared = len(patterns) - len(said)
    raise StageError(
        "transform",
        f"{name} left {total} op(s) it declared would not survive it: {shown}. "
        f"First at {left[0][2]}. {os.path.basename(source or '?')} declares "
        f"FORBIDS = {tuple(said)!r}"
        + (f" (and {spared} exemption(s))" if spared else "")
        + f", and every pass below {name} is written as though these do not "
        f"exist -- so whatever fails next would name something else. The IR it "
        f"produced is {artifact}.",
        artifact=artifact)


def _fresh_dir(path):
    """`path` as an empty directory: a pass dump left from an earlier run would read as this one's."""
    os.makedirs(path, exist_ok=True)
    for stale in os.listdir(path):
        os.remove(os.path.join(path, stale))
    return path


# --------------------------------------------------------------------------
# Stage torchsim-compile, step lower: p08..p11 -> 09-custom.mlir, p12..p13 -> 10-lowered.mlir
# --------------------------------------------------------------------------
_OP_RE = re.compile(r"(?:^\s*|=\s*)([a-z_][a-z0-9_]*)\.[a-z_][a-z0-9_.]*",
                    re.MULTILINE)


def stage_lower(ctx):
    src = ctx.path("08-transformed.mlir")
    custom = ctx.path("09-custom.mlir")
    out = ctx.path("10-lowered.mlir")
    _ensure_bindings()
    from .lowering.passes import (run_module_passes, POST_OPT_PASSES,
                                  run_standard_lowering)
    from .lowering.passes.p13_lower_to_llvm import SpadReservationRefused

    pass_dir = _fresh_dir(ctx.path("09-passes"))
    step = [0]
    mark = [time.perf_counter()]

    def dump(name, module):
        wallclock.PROF.add_pass(name, time.perf_counter() - mark[0])
        step[0] += 1
        with open(os.path.join(pass_dir, f"{step[0]:02d}-{name}.mlir"), "w") as f:
            f.write(str(module))
        mark[0] = time.perf_counter()

    try:
        run_module_passes(src, custom, POST_OPT_PASSES, dump=dump, forbids_extra=ctx.spec.extra,
                          vectorlane=machine.VECTORLANE_SIZE, vlen=machine.VLEN_BITS,
                          queue_entries=machine.SYSTOLIC_QUEUE_ENTRIES)
    except Exception as e:  # noqa: BLE001
        raise StageError("lower", str(e), artifact=src, stderr=repr(e))
    ctx.log(f"[lower] POST_OPT_PASSES done "
            f"({', '.join(m.__name__.rsplit('.', 1)[-1] for m in POST_OPT_PASSES)})")

    try:
        reserved = run_standard_lowering(custom, out, dump=dump,
                                         vectorlane=machine.VECTORLANE_SIZE)
    except SpadReservationRefused as e:
        raise StageError("lower", str(e), artifact=custom) from None
    ctx.log(f"[lower] scratchpad reserved for {len(reserved)} global(s): "
            + " ".join(f"{n}={c}x{e}" for n, c, e in reserved))
    ctx.log(f"[lower] per-pass IR in {pass_dir}/")
    ctx.log("[lower] standard lowering to LLVM dialect done")

    with open(out) as f:
        text = f.read()
    residue = sorted({
        tok.split(".")[0] for tok in _OP_RE.findall(text)
    } - {"llvm", "builtin", "func", "vcix", "module"})
    if residue:
        raise StageError("lower", f"non-LLVM dialects survive lowering: {residue}", artifact=out)
    return out


# --------------------------------------------------------------------------
# Stage torchsim-compile, step binary; and --tog
# --------------------------------------------------------------------------
def stage_binary(ctx):
    """The kernel object: the ELF a device runs, and the manifest naming it."""
    return elf_build.build_kernel_object(ctx)


def stage_tog(ctx):
    """What a timing simulator reads: the TOG trace producer, every tile's compute type, and the
    one-tile binary gem5 measures."""
    from .lowering.tog import cycle_binary, trace
    cpp, types = trace.build_trace(ctx.spec, ctx.workdir, ctx.tog_grouping)
    elf = cycle_binary.build_cycle_binary(ctx.spec, ctx.kernel_path, ctx.workdir,
                                          grouping=ctx.tog_grouping)
    for made in (cpp, types, elf):
        ctx.log(f"[trace] {made}")
    return cpp


STAGE_FN = {
    "triton-compile": stage_triton_compile,
    "transform": stage_transform,
    "lower": stage_lower,
    "binary": stage_binary,
    "tog": stage_tog,
}

STAGE_DESC = {
    "triton-compile": "triton kernel -> TTGIR -> linalg/memref/scf.for",
    "torchsim-compile": "tts/linalg -> RISC-V ELF: transform, lower, binary",
    "transform": "the tts-level backend (pytorchsim_triton_compiler/lowering/passes/)",
    "lower": "matmul, vcix, xlu, descriptors -> DMA -> LLVM",
    "binary": "mlir-translate -> llc -> link RISC-V ELF",
    "tog": "the TOG trace producer, the tile types and the cycle ELF (--tog)",
}


# --------------------------------------------------------------------------
# expect_ir: the op counts a kernel file declares its artifacts hold
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


def _count_op(text, op):
    spellings = op if isinstance(op, tuple) else (op,)
    return sum(len(re.findall(r"(?<![\w.])" + re.escape(s) + r"(?![\w.])", text))
               for s in spellings)


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
        wrong = [(op, want, _count_op(text, op)) for op, want in counts.items()
                 if _count_op(text, op) != want]
        if wrong:
            raise StageError(stage, "%s does not say what the kernel declares: %s"
                             % (fname, ", ".join(
                                 f"{_spell(op)} is {got}, expect_ir says {want}"
                                 for op, want, got in wrong)),
                             artifact=path)
        ctx.log(f"[expect_ir] {fname}: {len(counts)} count(s) as declared")


# --------------------------------------------------------------------------
# Driving the stages
# --------------------------------------------------------------------------
def _ensure_bindings():
    toolchain.activate_bindings(sys.path)
    import mlir  # noqa: F401
    toolchain.assert_bindings_clean()


def run(spec, kernel_path, workdir, from_stage="triton-compile", to_stage=None,
        verbose=False, breakdown=False, tog=False, tog_grouping="typed"):
    """Drive the stages, then leave `timing.json` in the workdir either way."""
    wallclock.PROF.label = spec.name
    try:
        return _run_stages(spec, kernel_path, workdir, from_stage=from_stage,
                           to_stage=to_stage, verbose=verbose, tog=tog,
                           tog_grouping=tog_grouping)
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
                verbose=False, tog=False, tog_grouping="typed"):
    ctx = Ctx(spec, kernel_path, workdir, verbose=verbose, tog_grouping=tog_grouping)
    open(ctx.log_path, "w").close()

    todo, whole = _plan(from_stage, to_stage, tog)
    _clear_expect_ir(ctx, todo)
    if "transform" in todo:
        pathlib.Path(ctx.path(quality.FILENAME)).unlink(missing_ok=True)

    print(f"kernel   : {spec.name}   ({kernel_path})")
    print(f"workdir  : {ctx.workdir}")
    print(f"stages   : {' -> '.join(todo)}")
    print()

    checked_ir = set()
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
                _check_expect_ir(ctx, name, checked_ir)
        except StageError as e:
            dt = time.time() - t0
            if isinstance(e, SpillNotAllowed) and ctx.lmul > 1:
                ctx.lmul //= 2
                print(f"{label} RETRY ({dt:.1f}s)  {e}, which we do not allow; "
                      f"again from transform at LMUL {ctx.lmul}")
                ctx.log(f"[retry] {e}")
                todo, i = _rewind_to_transform(todo, i - 1)
                _clear_expect_ir(ctx, todo[i:])
                checked_ir.clear()
                continue
            if _stops_as_declared(spec, name):
                return _report_stopped(spec, label, name, dt, str(e))
            _report_failure(ctx, spec, kernel_path, label, name, dt, e)
            return False
        except Exception as e:  # noqa: BLE001
            dt = time.time() - t0
            if _stops_as_declared(spec, name):
                return _report_stopped(spec, label, name, dt,
                                       f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
            _report_failure(ctx, spec, kernel_path, label, name, dt, None)
            return False

        dt = time.time() - t0
        shown = os.path.relpath(str(artifact), ctx.workdir) if artifact else ""
        print(f"{label} ok    ({dt:.1f}s)  {STAGE_DESC[name]:<45} {shown}")

    return _report_finished(ctx, spec, todo, whole, checked_ir)


def _rewind_to_transform(todo, at):
    if "transform" in todo[:at]:
        return todo, todo.index("transform")
    return todo[:at] + ["transform", "lower"] + todo[at:], at


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
