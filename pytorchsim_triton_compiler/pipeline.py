"""The stage machine."""
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

from . import machine
from . import toolchain
from .contract import kernel_object
from .metrics import wallclock
from .metrics import quality
from .lowering import wrapper

_OP_RE = re.compile(r"(?:^\s*|=\s*)([a-z_][a-z0-9_]*)\.[a-z_][a-z0-9_.]*",
                    re.MULTILINE)

STAGES = ["triton-compile", "triton-shared", "torchsim-compile"]

STEPS_OF = {"torchsim-compile": ["transform", "lower", "binary"]}


def steps():
    """Every step in order -- a stage's own steps, or the stage itself."""
    return [s for st in STAGES for s in STEPS_OF.get(st, [st])]


def stage_of(step):
    """The stage `step` belongs to, or None for one on the --trace branch."""
    return next((st for st in STAGES if step in STEPS_OF.get(st, [st])), None)

TRACE_STAGES = ["trace"]


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


def _run(ctx, stage, cmd, **kw):
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


def stage_triton_compile(ctx):
    """The only stage that imports triton, and it does so in its OWN process: libtriton carries an
    MLIR of its own, and beside the bindings' one the dialect registries cross."""
    out = ctx.path("01-ttir.mlir")
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "lowering", "triton_ttir.py")
    _run(ctx, "triton-compile", [sys.executable, worker, ctx.kernel_path, out],
         env=toolchain.triton_env())
    ctx.log(f"[stage1] ttir written: {out}")
    return out


NOOPS = ("llvm.intr.assume", "ttg.barrier")
_NOOP_ALT = "|".join(re.escape(n) for n in NOOPS)
_NOOP_LINES = re.compile(r'^[ \t]*(?:' + _NOOP_ALT + r')\b[^\n]*\n?', re.M)
_NOOP_LEFT = re.compile(r'^[^\n]*\b(?:' + _NOOP_ALT + r')\b[^\n]*', re.M)


def _drop_noops(text):
    out, n = _NOOP_LINES.subn("", text)
    left = _NOOP_LEFT.search(out)
    if left:
        raise StageError("triton-shared", "a no-op the text rule cannot take: %r -- "
                         "nothing below can parse it either" % left.group(0).strip())
    return out, n


def _write_ttshared_input(ctx):
    src, out = ctx.path("01-ttir.mlir"), ctx.path("02-ttclean.mlir")
    with open(src) as f:
        text = f.read()
    cleaned, n = _drop_noops(text)
    if n:
        ctx.log(f"[triton-shared] {n} unparseable no-op(s) taken out of the text")
        with open(out, "w") as f:
            f.write(cleaned)
    else:
        shutil.copyfile(src, out)
    return out


TTGIR_TARGET = "torchsim:0"
def stage_triton_shared(ctx):
    """02-ttclean.mlir to 03-ttshared.mlir in the stage-2 worker, on an unmodified Triton with the
    torchsim extension: the kernel as TritonGPU IR with every layout this machine fixes anchored in
    one pass and propagated once (02a, 02b, 02h), lowered by triton_shared with every encoding
    carried down (03a), then its layouts regularized -- every encoding the LinearLayout it is, unit
    extents folded (03)."""
    clean = _write_ttshared_input(ctx)
    names = ("02a-ttgir.mlir", "02b-anchored.mlir", "02h-layouts.mlir", "03a-ttshared.mlir",
             "03-ttshared.mlir")
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lowering", "triton_ttshared.py")
    _run(ctx, "triton-shared", [sys.executable, worker, clean, TTGIR_TARGET, machine.VECTORLANE_SIZE,
                                *(ctx.path(n) for n in names)], env=toolchain.triton_env())
    _check_layouts_carried(ctx, ctx.path("03a-ttshared.mlir"))
    return ctx.path("03-ttshared.mlir")


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
        raise StageError("triton-shared", "%s: %d tile(s) without a layout encoding, first %s"
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
    src, out = ctx.path("03-ttshared.mlir"), ctx.path("04-transformed.mlir")
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


_WRITTEN_BY = {"01": "triton-compile", "02": "triton-shared", "03": "triton-shared",
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
                          vectorlane=machine.VECTORLANE_SIZE, vlen=machine.VLEN_BITS)
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


_LANE_EXTRACT = ("vfmv.f.s", "vmv.x.s")


def _check_no_lane_extract(ctx, asm, stage="binary"):
    import re
    with open(asm) as f:
        bad = [(n, line.strip()) for n, line in enumerate(f, 1)
               if re.match(r"\s*(%s)\b" % "|".join(
                   i.replace(".", r"\.") for i in _LANE_EXTRACT), line)]
    if bad:
        where = "; ".join(f"{asm}:{n}: {t}" for n, t in bad[:4])
        raise StageError(
            stage,
            f"vector->scalar extract in a lane-banked kernel: {where}. There is "
            f"one scalar register file for all lanes, so this keeps lane 0 and "
            f"discards the rest. Keep the value in a vector register.",
            artifact=asm)


_LL_ELEM_BITS = {"i8": 8, "i16": 16, "i32": 32, "i64": 64, "i128": 128,
                 "half": 16, "bfloat": 16, "float": 32, "double": 64,
                 "ptr": 64}
_LL_VECTOR = re.compile(r"<(\d+) x ([a-z0-9]+)>")


SPAD_OVERFLOW_MARKER = "torchsim-spad-overflow"

_ELF_SECTION = re.compile(
    r"^\s*\[\s*\d+\]\s+(?P<name>\S+)\s+\S+\s+[0-9a-f]+\s+"
    r"[0-9a-f]+\s+(?P<size>[0-9a-f]+)", re.MULTILINE)


def _measure_spad(elf, readelf):
    from .lowering.passes import p12_lower_to_llvm as p12
    r = subprocess.run([readelf, "-SW", elf], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"readelf failed on {elf}: {r.stderr.strip()}")
    for m in _ELF_SECTION.finditer(r.stdout):
        if m.group("name") == p12.SPAD_SECTION:
            return int(m.group("size"), 16)
    return 0


def _check_spad_budget(ctx, elf, stage, tag):
    spad_size = machine.SPAD_SIZE
    try:
        usage = _measure_spad(elf, toolchain.RISCV_READELF)
    except RuntimeError as e:
        ctx.warn(f"could not measure .spad usage: {e}")
        return 0
    said = None
    if usage > spad_size:
        budget, said = spad_size, (
            f"scratchpad overflow: kernel claims {usage} bytes/lane but Spike maps "
            f"only {spad_size} (--scratchpad-size). Raise TORCHSIM_COMPILE_SPAD_SIZE or shrink "
            f"the tile.")
    elif usage > spad_size // 2:
        budget, said = spad_size // 2, (
            f"scratchpad usage {usage} bytes/lane exceeds the double-buffer "
            f"budget of {spad_size // 2} (spad/2), so this kernel cannot be "
            f"double buffered on the target. THERE ARE TWO SHAPES OF THIS and "
            f"the buffer count tells them apart: a handful of globals means ONE "
            f"of them is wrong -- dump the memref.globals and look for a tile "
            f"banked on a unit axis, which lands the whole thing in one lane and "
            f"reserves it in every lane -- while dozens of globals at a few KB "
            f"each means the TILE is too big for what the kernel fuses, and the "
            f"caller that chose the block size is the one that can fix it "
            f"(PyTorchSim's Inductor route reads the marker below and retries "
            f"with a smaller R0_BLOCK). Raise TORCHSIM_COMPILE_SPAD_SIZE only if the tile is "
            f"genuinely that big.")
    if said is not None:
        raise StageError(stage, f"{said}\n{SPAD_OVERFLOW_MARKER}: "
                                f"usage={usage} budget={budget}", artifact=elf)
    if usage:
        ctx.log(f"[{tag}] scratchpad: {usage} bytes/lane of {spad_size}")
    return usage


def _check_vector_budget(ctx, ll, stage="binary"):
    budget = machine.VLEN_BITS * machine.ISA_MAX_LMUL
    with open(ll) as f:
        text = f.read()
    text = re.sub(r"(shufflevector\b[^\n]*?), <\d+ x i32> [^\n]*", r"\1", text)
    worst = {}
    for n, elem in _LL_VECTOR.findall(text):
        bits = _LL_ELEM_BITS.get(elem)
        if bits is None:
            continue
        total = int(n) * bits
        if total > budget:
            worst[f"<{n} x {elem}>"] = total
    if worst:
        shown = "; ".join(
            f"{v} needs {b} bits (LMUL {b // machine.VLEN_BITS})"
            for v, b in sorted(worst.items(), key=lambda kv: -kv[1])[:3])
        raise StageError(
            stage,
            f"a vector is wider than one instruction can reach: {shown}. The "
            f"budget is VLEN {machine.VLEN_BITS} x LMUL {machine.ISA_MAX_LMUL} = "
            f"{budget} bits. The element count is chosen against the type a "
            f"chain STARTS with (legalize) and spent "
            f"against the widest type it reaches, so a widening op after the "
            f"choice is what usually does this.",
            artifact=ll)


_VEC_SPILL = re.compile(r"^\s*vs(\d+)r\.v\b", re.M)
_VEC_RELOAD = re.compile(r"^\s*vl(\d+)re?\d*\.v\b", re.M)
_CFA = re.compile(r"\.cfi_def_cfa_offset\s+(\d+)|\.cfi_escape[^#\n]*#\s*sp\s*\+"
                  r"\s*(\d+)(?:\s*\+\s*(\d+)\s*\*\s*vlenb)?")
_OBJDUMP_FIRST = re.compile(r"^[0-9a-f]+ <[^>]+>:\n\s*[0-9a-f]+:\s+([0-9a-f]+)\s", re.M)
_SPIKE_OPENS_A_WINDOW = ((0xe003, 0x6001), (0x707f, 0x0013))


def _get_stack_depth(text):
    depth = 0
    for m in _CFA.finditer(text):
        got = (int(m.group(1)) if m.group(1) else int(m.group(2))
               + int(m.group(3) or 0) * machine.VLEN_BITS // 8)
        depth = max(depth, got)
    return depth


def _get_first_instruction(ctx, stage, obj):
    out = _run(ctx, stage, [toolchain.RISCV_OBJDUMP, "-d", obj]).stdout
    m = _OBJDUMP_FIRST.search(out)
    return int(m.group(1), 16) if m else 0


def _is_a_window_spike_opens(bits):
    return any(bits & mask == match for mask, match in _SPIKE_OPENS_A_WINDOW)


def _check_stack_is_allowed(ctx, asm, obj, usage):
    with open(asm) as f:
        text = f.read()
    stores = [int(n) for n in _VEC_SPILL.findall(text)]
    loads = [int(n) for n in _VEC_RELOAD.findall(text)]
    depth = _get_stack_depth(text)
    fits = usage + depth <= machine.SPAD_SIZE
    if stores or loads:
        ctx.log(f"[stage6] llc spilled {len(stores)} vector register group(s) "
                f"and reloads {len(loads)} at LMUL {ctx.lmul}; the stack is "
                f"{depth} bytes over {usage} bytes/lane of tiles in a "
                f"{machine.SPAD_SIZE}-byte bank")
        opens = _is_a_window_spike_opens(_get_first_instruction(ctx, "binary", obj))
        if not opens or not fits:
            ctx.log("[stage6] spill not allowed: " + (
                "the first instruction opens no stack window" if not opens
                else "the stack does not fit above the tiles"))
            raise SpillNotAllowed(
                "binary",
                f"llc spilled {len(stores)} vector register group(s) "
                + ("even at LMUL 1" if ctx.lmul == 1 else f"at LMUL {ctx.lmul}"),
                artifact=obj)
    if not fits:
        raise StageError(
            "binary",
            f"the kernel's stack, {depth} bytes, does not fit above its tiles, "
            f"{usage} bytes/lane, in a {machine.SPAD_SIZE}-byte bank: spike "
            f"lays the stack on the top of each lane's bank and stops with "
            f"VI_ST_WHOLE ERROR: Accessing invalid spad address or Accessing "
            f"stack address.", artifact=asm)


def stage_binary(ctx):
    """The kernel object: the ELF a device runs, and the manifest naming it."""
    ll = ctx.path("06-kernel.ll")
    asm = ctx.path("06-kernel.s")
    wrap_c = ctx.path("06-wrapper.c")
    obj = ctx.path("06-kernel.o")
    elf, usage = build_elf(ctx, "binary", "stage6",
                           lowered=ctx.path("05-lowered.mlir"),
                           ll=ll, obj=obj, asm=asm, wrap_c=wrap_c,
                           elf=ctx.path(f"06-{ctx.spec.name}.elf"))
    _check_stack_is_allowed(ctx, asm, obj, usage)

    manifest = kernel_object.write(
        ctx.workdir, ctx.spec, machine.target(),
        kernel_addr=_kernel_addr_range(ctx, elf),
        artifacts={"elf": os.path.basename(elf),
                    "post_vcix_ir": _relative_if_present(ctx, "05-custom.mlir"),
                    "llvm_ir": os.path.basename(ll),
                    "asm": os.path.basename(asm),
                    "wrapper_c": os.path.basename(wrap_c),
                    "quality": _relative_if_present(ctx, quality.FILENAME)})

    ctx.log(f"[stage6] ELF: {elf}")
    ctx.log(f"[stage6] kernel object: {manifest}")
    return elf


def build_elf(ctx, stage, tag, lowered, ll, obj, asm, wrap_c, elf,
              single_program=False):
    """LLVM-dialect MLIR -> `(ELF, .spad bytes/lane)`, with the vector and scratchpad budgets
    checked."""
    _run(ctx, stage, [toolchain.MLIR_TRANSLATE, "-mlir-to-llvmir", lowered, "-o", ll])
    _check_vector_budget(ctx, ll, stage)
    common = ["-relocation-model=pic", "-march=riscv64", "-O3", "--stack-size-section",
              f"-mattr=+m,+a,+f,+d,+c,+v,+zvfh,+xsfvcp,+zvl{machine.VLEN_BITS}b", ll]
    _run(ctx, stage, [toolchain.LLC, *common, "-filetype=obj", "-o", obj])
    _run(ctx, stage, [toolchain.LLC, *common, "-o", asm])
    _check_no_lane_extract(ctx, asm, stage)

    override = wrapper.find_override(ctx.kernel_path)
    if override and single_program:
        raise StageError(
            stage,
            f"{os.path.basename(override)} is a hand-written wrapper, and it "
            f"runs the whole launch grid. The cycle binary times ONE tile, so "
            f"using it would multiply every count by the grid, and generating "
            f"one instead would ignore whatever the override exists to say. "
            f"Write a single-program wrapper for this kernel, or drop the "
            f"override.", artifact=override)
    if override:
        shutil.copyfile(override, wrap_c)
        ctx.log(f"[{tag}] using hand-written wrapper {override}")
    else:
        src = wrapper.generate(ctx.spec, single_program=single_program)
        with open(wrap_c, "w") as f:
            f.write(src)
        ctx.log(f"[{tag}] generated wrapper (see {os.path.basename(wrap_c)} "
                f"for the assumed ABI)")

    from .lowering.passes import p12_lower_to_llvm as p12
    link = [toolchain.RISCV_GCC, "-O3", "-march=rv64gcv_zfh", "-mabi=lp64d",
            wrap_c, obj, "-o", elf,
            f"-Wl,--section-start={p12.SPAD_SECTION}"
            f"=0x{machine.SPAD_VADDR:x}",
            f"-Wl,--section-start={p12.SPAD_TOP_SECTION}=0x"
            f"{machine.SPAD_VADDR + machine.SPAD_SIZE*machine.VECTORLANE_SIZE:x}"]
    _run(ctx, stage, link)

    usage = _check_spad_budget(ctx, elf, stage, tag)
    return elf, usage


def _relative_if_present(ctx, name):
    return name if os.path.exists(ctx.path(name)) else None


def _kernel_addr_range(ctx, elf):
    out = _run(ctx, "binary", [toolchain.RISCV_NM, "-S", elf]).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[3] == ctx.spec.name:
            start, size = int(parts[0], 16), int(parts[1], 16)
            ctx.log(f"[stage6] kernel {ctx.spec.name} at "
                    f"0x{start:x}..0x{start + size:x}")
            return [start, start + size]
    raise StageError(
        "binary",
        f"no sized symbol '{ctx.spec.name}' in the ELF -- cannot tell a device "
        f"the kernel PC range, and without it the vector unit runs single-lane",
        artifact=elf)


def stage_trace(ctx):
    """The TOGSim trace producer's C++ and the compute type of every tile."""
    from .simulation import emit_togsim_so

    cpp, types = emit_togsim_so.build_trace(ctx.spec, ctx.workdir)
    ctx.log(f"[trace] {cpp}")
    ctx.log(f"[trace] {types}")
    return cpp


def stage_simulate(ctx):
    """gem5 tile cycles and a TOGSim run of the trace, no PyTorchSim in between."""
    from .simulation import simulate

    try:
        r = simulate.simulate(ctx.spec, ctx.kernel_path, ctx.workdir)
    except simulate.SimulateError as e:
        raise StageError("simulate", str(e)) from e
    ctx.log(f"[simulate] tile cycles {r['tile_cycles']}")
    print(f"TOGSIM CYCLES: {r['total_cycles']}  (tiles {r['tile_cycles']})")
    return os.path.join(ctx.workdir, simulate.RESULT)


STAGE_FN = {
    "triton-compile": stage_triton_compile,
    "triton-shared": stage_triton_shared,
    "transform": stage_transform,
    "lower": stage_lower,
    "binary": stage_binary,
    "trace": stage_trace,
    "simulate": stage_simulate,
}

STAGE_DESC = {
    "triton-compile": "triton kernel -> Triton IR",
    "triton-shared": "Triton IR -> linalg/memref/scf.for (triton-shared)",
    "torchsim-compile": "tts/linalg -> RISC-V ELF: transform, lower, binary",
    "transform": "the tts-level backend (pytorchsim_triton_compiler/lowering/passes/)",
    "lower": "stage-5 lowering (vcix -> DMA -> LLVM)",
    "binary": "mlir-translate -> llc -> link RISC-V ELF",
    "trace": "the TOGSim trace producer, C++ and .so (--trace)",
    "simulate": "gem5 tile cycles, then TOGSim on the trace (--simulate)",
}

LAST_STAGE = "binary"


def register_stage(name, fn, desc, last=False, trace=False):
    """Add a stage after the compiler's own, and return the new stage list."""
    global LAST_STAGE
    if trace:
        if name not in TRACE_STAGES:
            TRACE_STAGES.append(name)
    elif name not in STAGES:
        STAGES.append(name)
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
        verbose=False, breakdown=False, trace=False, simulate=False):
    """Drive the stages, then leave `timing.json` in the workdir either way."""
    wallclock.PROF.label = spec.name
    try:
        return _run_stages(spec, kernel_path, workdir, from_stage=from_stage,
                           to_stage=to_stage, verbose=verbose, trace=trace,
                           simulate=simulate)
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
                verbose=False, trace=False, simulate=False):
    ctx = Ctx(spec, kernel_path, workdir, verbose=verbose)
    open(ctx.log_path, "w").close()

    chain = steps()
    first = STEPS_OF.get(from_stage, [from_stage])[0]
    last = STEPS_OF.get(to_stage, [to_stage])[-1] if to_stage else chain[-1]
    todo = chain[chain.index(first):chain.index(last) + 1]
    _clear_expect_ir(ctx, todo)
    _clear_quality(ctx, todo)
    if (trace or simulate) and "binary" in todo:
        extra = TRACE_STAGES + (["simulate"] if simulate else [])
        todo = todo[:todo.index("binary") + 1] + extra + todo[todo.index("binary") + 1:]

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
