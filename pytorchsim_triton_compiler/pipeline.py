"""The stage machine."""
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

_OP_RE = re.compile(r"(?:^\s*|=\s*)([a-z_][a-z0-9_]*)\.[a-z_][a-z0-9_.]*",
                    re.MULTILINE)

STAGES = ["triton-compile", "torchsim-compile"]

STEPS_OF = {"torchsim-compile": ["transform", "lower", "binary"]}


def steps():
    """Every step in order -- a stage's own steps, or the stage itself."""
    return [s for st in STAGES for s in STEPS_OF.get(st, [st])]


def stage_of(step):
    """The stage `step` belongs to, or None for one on the --tog branch."""
    return next((st for st in STAGES if step in STEPS_OF.get(st, [st])), None)

TOG_STAGES = ["tog"]


def stage_triton_compile(ctx):
    """The only stage that imports triton, and it does so in its OWN process: libtriton carries an
    MLIR of its own, and beside the bindings' one the dialect registries cross. The worker takes the
    kernel from its SPEC to 03-linalg.mlir: Triton IR (01), the no-ops nothing below parses dropped
    (02), TTGIR with every layout this machine fixes anchored in one pass and propagated once (02a,
    02b, 02h), lowered to linalg/memref with every encoding carried down (03a), then its layouts
    regularized -- every encoding the LinearLayout it is, unit extents folded (03)."""
    names = ("01-ttir.mlir", "02-ttclean.mlir", "02a-ttgir.mlir", "02b-anchored.mlir",
             "02h-layouts.mlir", "03a-linalg.mlir", "03-linalg.mlir")
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lowering", "triton_compile.py")
    run_tool(ctx, "triton-compile", [sys.executable, worker, ctx.kernel_path, machine.VECTORLANE_SIZE,
                                     *(ctx.path(n) for n in names)], env=toolchain.triton_env())
    _check_layouts_carried(ctx, ctx.path("03a-linalg.mlir"))
    return ctx.path("03-linalg.mlir")


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


SUPPORTED_DTYPES = frozenset((
    "i1", "i8", "i16", "i32", "i64",
    "f16", "f32", "f64", "bf16",
))


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


def stage_transform(ctx):
    src, out = ctx.path("03-linalg.mlir"), ctx.path("04-transformed.mlir")
    _ensure_bindings()
    from mlir.ir import Context, Module, Location
    with wallclock.PROF.step("import pytorchsim_triton_compiler.lowering.passes"):
        from .lowering import passes  # noqa: F401
    from .lowering.passes import (pass_name,
                         p00_convert_unsupported_ops,
                         p01_convert_accesses_to_transfer,
                         p02_materialize_layouts, p07_lower_copies,
                         p03_fuse,
                         p04_bufferize,
                         p05_vectorize, p06_legalize)

    with open(src) as f:
        text = f.read()

    arg_numels = {i: a.numel for i, a in enumerate(ctx.spec.args)}
    mlir_ctx = Context()
    mlir_ctx.allow_unregistered_dialects = True
    steps = os.path.join(ctx.workdir, "04-passes")
    os.makedirs(steps, exist_ok=True)
    for stale in os.listdir(steps):
        os.remove(os.path.join(steps, stale))
    ran = []
    mark = [time.perf_counter()]

    def dump(name, module):
        wallclock.PROF.add_pass(name, time.perf_counter() - mark[0])
        ran.append(name)
        path = os.path.join(steps, f"{len(ran):02d}-{name}.mlir")
        with open(path, "w") as f:
            f.write(str(module))
        _check_no_zero_stride(ctx, name, module, path)
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
    from .lowering.passes import PHASES, PIPELINE
    if ran[1:] != PIPELINE:
        raise StageError(
            "transform",
            f"stage 4 ran {ran[1:]} but pytorchsim_triton_compiler.lowering.passes.PHASES says {PIPELINE}",
            artifact=src)
    ctx.log(f"[stage4] per-pass IR in {steps}/")
    for title, names in PHASES:
        ctx.log(f"[stage4] {title}: " + " -> ".join(names))
    quality.write(ctx.workdir, counts, one_lane)
    xlu = counts["xlu"]
    ctx.log(f"[stage4] DMA: {counts['mvin']} MVIN, {counts['mvout']} MVOUT; "
            f"moves between lanes: {counts['round_trips']} through DRAM, "
            f"{sum(xlu.values())} through the XLU"
            + (" (" + ", ".join(f"{k} {n}" for k, n in sorted(xlu.items()))
               + ")" if xlu else ""))
    return out


_ZERO_STRIDE = re.compile(r"strided<\[([^\]]*)\]")


def _check_no_zero_stride(ctx, name, module, artifact):
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


def _count_op(text, op):
    import re as _re
    spellings = op if isinstance(op, tuple) else (op,)
    return sum(len(_re.findall(r"(?<![\w.])" + _re.escape(s) + r"(?![\w.])", text))
               for s in spellings)


def _spell(op):
    return " or ".join(op) if isinstance(op, tuple) else op


_WRITTEN_BY = {"01": "triton-compile", "02": "triton-compile", "03": "triton-compile",
               "04": "transform", "05": "lower", "06": "binary"}


def _clear_expect_ir(ctx, todo):
    for fname in _get_expect_ir(ctx.spec):
        if _WRITTEN_BY.get(fname[:2], todo[0]) in todo:
            pathlib.Path(os.path.join(ctx.workdir, fname)).unlink(missing_ok=True)


def _get_expect_ir(spec):
    return dict(spec.expect_ir or {})


def _clear_quality(ctx, todo):
    if "transform" not in todo:
        return
    from .metrics import quality
    pathlib.Path(os.path.join(ctx.workdir, quality.FILENAME)).unlink(
        missing_ok=True)


def _the_artifact_named(workdir, fname):
    import glob as _glob
    import re as _re
    path = os.path.join(workdir, fname)
    if os.path.isfile(path):
        return path
    m = _re.match(r"(.*/)?\d+-(.+)$", fname)
    if m is None:
        return None
    got = _glob.glob(os.path.join(workdir, m.group(1) or "", "*-" + m.group(2)))
    return got[0] if len(got) == 1 else None


def _check_expect_ir(ctx, stage, done):
    if os.environ.get("TORCHSIM_COMPILE_IGNORE_EXPECT_IR"):
        return
    for fname, counts in _get_expect_ir(ctx.spec).items():
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


def stage_lower(ctx):
    src = ctx.path("04-transformed.mlir")
    custom = ctx.path("05-custom.mlir")
    out = ctx.path("05-lowered.mlir")
    _ensure_bindings()
    from .lowering.passes import (run_module_passes, POST_OPT_PASSES,
                         run_standard_lowering)
    from .lowering.passes.p12_lower_to_llvm import SpadReservationRefused
    try:
        toolchain.assert_bindings_clean()
    except RuntimeError as e:
        raise StageError("lower", str(e).splitlines()[0],
                         stderr=str(e), artifact=src)

    steps = ctx.path("05-passes")
    os.makedirs(steps, exist_ok=True)
    for stale in os.listdir(steps):
        os.remove(os.path.join(steps, stale))
    step = [0]
    mark = [time.perf_counter()]

    def dump(name, module):
        wallclock.PROF.add_pass(name, time.perf_counter() - mark[0])
        step[0] += 1
        with open(os.path.join(steps, f"{step[0]:02d}-{name}.mlir"), "w") as f:
            f.write(str(module))
        mark[0] = time.perf_counter()

    try:
        run_module_passes(src, custom, POST_OPT_PASSES, dump=dump, forbids_extra=ctx.spec.extra,
                          vectorlane=machine.VECTORLANE_SIZE, vlen=machine.VLEN_BITS,
                          queue_entries=machine.SYSTOLIC_QUEUE_ENTRIES)
    except Exception as e:  # noqa: BLE001
        raise StageError("lower", str(e), artifact=src, stderr=repr(e))
    ctx.log(f"[stage5] POST_OPT_PASSES done "
            f"({', '.join(m.__name__.rsplit('.', 1)[-1] for m in POST_OPT_PASSES)})")

    try:
        reserved = run_standard_lowering(custom, out, dump=dump,
                                         vectorlane=machine.VECTORLANE_SIZE)
    except SpadReservationRefused as e:
        raise StageError("lower", str(e), artifact=custom) from None
    ctx.log(f"[stage5] scratchpad reserved for {len(reserved)} global(s): "
            + " ".join(f"{n}={c}x{e}" for n, c, e in reserved))
    ctx.log(f"[stage5] per-pass IR in {steps}/")
    ctx.log("[stage5] standard lowering to LLVM dialect done")

    with open(out) as f:
        text = f.read()
    residue = sorted({
        tok.split(".")[0] for tok in _OP_RE.findall(text)
    } - {"llvm", "builtin", "func", "vcix", "module"})
    if residue:
        raise StageError("lower", f"non-LLVM dialects survive lowering: {residue}", artifact=out)
    return out


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
    "lower": "stage-5 lowering (vcix -> DMA -> LLVM)",
    "binary": "mlir-translate -> llc -> link RISC-V ELF",
    "tog": "the TOG trace producer, the tile types and the cycle ELF (--tog)",
}

LAST_STAGE = "binary"


def register_stage(name, fn, desc, last=False, tog=False, after=None):
    """Add a stage after the compiler's own -- right after `after` when one is named -- or, with
    `tog`, on the branch --tog adds after `binary`; return the new stage list."""
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
    return STAGES


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


def _get_resume_step(ctx, name):
    narrowed = ctx.lmul < machine.ISA_MAX_LMUL
    return "transform" if narrowed and name in STEPS_OF["torchsim-compile"] else name


def _rewind_to_transform(todo, at):
    if "transform" in todo[:at]:
        return todo, todo.index("transform")
    return todo[:at] + ["transform", "lower"] + todo[at:], at


def _run_stages(spec, kernel_path, workdir, from_stage="triton-compile", to_stage=None,
                verbose=False, tog=False, tog_grouping="typed"):
    ctx = Ctx(spec, kernel_path, workdir, verbose=verbose, tog_grouping=tog_grouping)
    open(ctx.log_path, "w").close()

    chain = steps()
    first = STEPS_OF.get(from_stage, [from_stage])[0]
    last = STEPS_OF.get(to_stage, [to_stage])[-1] if to_stage else chain[-1]
    todo = chain[chain.index(first):chain.index(last) + 1]
    _clear_expect_ir(ctx, todo)
    _clear_quality(ctx, todo)
    if tog and "binary" in todo:
        todo = todo[:todo.index("binary") + 1] + TOG_STAGES + todo[todo.index("binary") + 1:]

    print(f"kernel   : {spec.name}   ({kernel_path})")
    print(f"workdir  : {ctx.workdir}")
    print(f"stages   : {' -> '.join(todo)}")
    print()

    def _is_a_refusal(stage):
        return spec.expect in ("refuse", "blocked") and stage != LAST_STAGE

    def _is_open(stage):
        return spec.expect == "open"

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
            if _is_a_refusal(name) or _is_open(name):
                word = ("REFUSED" if spec.expect == "refuse"
                        else "BLOCKED" if spec.expect == "blocked" else "OPEN")
                print(f"{label} {word} ({dt:.1f}s)  {STAGE_DESC[name]}")
                print()
                print(f"  {e}")
                print()
                print(f"{word} AS EXPECTED: {spec.expect_why}")
                return True
            print(f"{label} FAIL  ({dt:.1f}s)  {STAGE_DESC[name]}")
            print()
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
            print(f"  resume  : pytorchsim-triton-compiler {kernel_path} "
                  f"--from {_get_resume_step(ctx, name)}")
            return False
        except Exception as e:  # noqa: BLE001
            dt = time.time() - t0
            if _is_a_refusal(name) or _is_open(name):
                word = ("REFUSED" if spec.expect == "refuse"
                        else "BLOCKED" if spec.expect == "blocked" else "OPEN")
                print(f"{label} {word} ({dt:.1f}s)  {STAGE_DESC[name]}")
                print()
                print(f"  {type(e).__name__}: {str(e).splitlines()[0][:200]}")
                print()
                print(f"{word} AS EXPECTED: {spec.expect_why}")
                return True
            print(f"{label} FAIL  ({dt:.1f}s)  {STAGE_DESC[name]}")
            print()
            import traceback
            traceback.print_exc()
            print()
            print(f"  full log: {ctx.log_path}")
            print(f"  resume  : pytorchsim-triton-compiler {kernel_path} "
                  f"--from {_get_resume_step(ctx, name)}")
            return False

        dt = time.time() - t0
        shown = os.path.relpath(str(artifact), ctx.workdir) if artifact else ""
        print(f"{label} ok    ({dt:.1f}s)  {STAGE_DESC[name]:<45} {shown}")

    whole = first == chain[0] and to_stage is None
    missing = [f for f in (spec.expect_ir or {}) if f not in checked_ir] if whole else []
    missing = [f for f in missing if _the_artifact_named(ctx.workdir, f) is None]
    if missing:
        print()
        print(f"  expect_ir names {', '.join(missing)}, which this run never "
              f"wrote. Name the artifact the stage prints beside itself.")
        return False

    print()
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
