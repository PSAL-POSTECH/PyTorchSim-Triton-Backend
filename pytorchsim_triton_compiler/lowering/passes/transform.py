"""Step transform: 07-regularized.mlir -> 08-transformed.mlir, PHASES in order. Every pass's IR goes
to 08-passes/ and is held there to three checks: no memref stride of zero, nothing left the pass
FORBIDS, and -- on the input -- no element type the machine does not run."""
import os
import time

from ... import machine, toolchain
from ...contract.target import TargetError
from ...dialects.torchsim import CopyOp
from ...metrics import quality, wallclock
from ...stage import StageError
from . import (forbidden, forbids, p00_convert_unsupported_ops, p01_convert_accesses_to_transfer,
               p02_materialize_layouts, p03_fuse, p03a_reuse_input_as_init, p04_bufferize,
               p04b_fit_to_numel, p04c_pipeline_reductions, p05_vectorize, p06_legalize,
               p06a_hoist_loop_invariants, p06b_narrow_index_arith,
               p07_make_transfers_async, pass_name)
from .lib_ir import elem_type, funcs, values, walk
from .lower import fresh_dir
from .slots import PassGateError, Slot, Unfilled, resolve, spelled

LOWER_COPIES = Slot("lower-copies", lowers=(CopyOp.OPERATION_NAME,))
# An extension's bound on a vector loop's LMUL, set before legalize chooses its widths
# (p06_legalize.MAX_LMUL_ATTR): what the loop must keep live at once that the kernel's LMUL does not know.
BOUND_LOOP_WIDTHS = Slot("bound-loop-widths")

PHASES = [
    ("tts -> transfers", [p00_convert_unsupported_ops, p01_convert_accesses_to_transfer]),
    ("layout made into shapes", [p02_materialize_layouts]),
    ("fusion after layout", [p03_fuse]),
    ("tensor -> memref", [p03a_reuse_input_as_init, p04_bufferize, p04b_fit_to_numel,
                          p04c_pipeline_reductions]),
    ("the machine is known", [p05_vectorize, BOUND_LOOP_WIDTHS, p06_legalize, p06b_narrow_index_arith,
                              p06a_hoist_loop_invariants,
                              LOWER_COPIES, p07_make_transfers_async]),
]

MACHINE_KNOWN = "the machine is known"

SUPPORTED_DTYPES = frozenset((
    "i1", "i8", "i16", "i32", "i64",
    "f16", "f32", "f64", "bf16",
))

def pipelines():
    """(title, list, whether the machine is known) of every phase. MACHINE_KNOWN is the first that
    may know it (rule 7): a pass that REQUIRES an extension, or a slot one fills, runs there on."""
    first = [title for title, _ in PHASES].index(MACHINE_KNOWN)
    return [(title, ps, i >= first) for i, (title, ps) in enumerate(PHASES)]


def phases(target):
    """PHASES as `target` runs them."""
    return [(title, resolve(ps, target)) for title, ps in PHASES]


def recipe():
    try:
        t = machine.target()
    except TargetError:
        t = None
    return [f"{title}: " + " -> ".join(spelled(ps, t)) for title, ps in PHASES] + [
        "-> 08-transformed.mlir" + (f"   (target {t.name})" if t else "")]


def run(ctx):
    src, out = ctx.path("07-regularized.mlir"), ctx.path("08-transformed.mlir")
    toolchain.ensure_bindings()
    from mlir.ir import Context, Module, Location

    with open(src) as f:
        text = f.read()
    try:
        ran_phases = phases(machine.target())
    except PassGateError as e:
        raise StageError("transform", str(e), artifact=src) from None

    opts = dict(lanes=machine.VECTORLANE_SIZE, vlen=machine.VLEN_BITS, lmul=ctx.lmul,
                loop_lmul=dict(getattr(ctx, "loop_lmul", {})),
                scan_base_length=machine.VECTORLANE_SIZE,
                arg_numels={i: a.numel for i, a in enumerate(ctx.spec.args)},
                grid=tuple(ctx.spec.grid))
    mlir_ctx = Context()
    mlir_ctx.allow_unregistered_dialects = True
    pass_dir = fresh_dir(ctx.path("08-passes"))
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
        for p in (p for _, ps in ran_phases for p in ps):
            try:
                p.run(module, **opts)
            except PassGateError as e:
                raise StageError("transform", str(e), artifact=src) from None
            if not isinstance(p, Unfilled):
                dump(pass_name(p), module)
        counts = quality.count(module)
        uniform = quality.count_uniform(module)
        # with locations: legalize's loop numbers ride them to llc (p06_legalize.LOOP_FILE)
        result = module.operation.get_asm(enable_debug_info=True)
    with open(out, "w") as f:
        f.write(result)

    ctx.log(f"[transform] per-pass IR in {pass_dir}/")
    for title, ps in ran_phases:
        ctx.log(f"[transform] {title}: " + " -> ".join(pass_name(p) for p in ps))
    quality.write(ctx.workdir, counts, uniform)
    xlu = counts["xlu"]
    ctx.log(f"[transform] DMA: {counts['mvin']} MVIN, {counts['mvout']} MVOUT; "
            f"moves between lanes: {counts['round_trips']} through DRAM, "
            f"{sum(xlu.values())} through the XLU"
            + (" (" + ", ".join(f"{k} {n}" for k, n in sorted(xlu.items()))
               + ")" if xlu else ""))
    return out


def _refuse_unsupported_dtypes(module):
    from mlir.ir import FloatType, IntegerType, Type
    allowed = {Type.parse(name, context=module.context) for name in SUPPORTED_DTYPES}
    for func in funcs(module):
        for op in walk(func.operation):
            for v in list(op.operands) + list(op.results):
                t = elem_type(v)
                for cls in (IntegerType, FloatType):
                    try:
                        cls(t)
                    except (TypeError, ValueError):
                        continue
                    if t not in allowed:
                        why = ("8-bit float is refused on this backend, and that is a decision "
                               "rather than a lowering still missing"
                               if cls is FloatType and FloatType(t).width == 8 else
                               "SUPPORTED_DTYPES lists what was measured, and nothing lowers "
                               "the rest")
                        raise NotImplementedError(
                            f"{t} is not an element type this machine runs -- {why}: {op}")


def _check_no_zero_stride(name, module, artifact):
    from mlir.ir import MemRefType, StridedLayoutAttr
    for v in values(module.operation):
        ty = v.type
        if not (isinstance(ty, MemRefType) and isinstance(ty.layout, StridedLayoutAttr)):
            continue
        strides = list(ty.layout.strides)
        if 0 in strides:
            raise StageError(
                "transform",
                f"{name} left a memref type with a zero stride: "
                f"{ty}. Our own MLIR parses that back and the "
                f"one reading the file may not -- PyTorchSim's says 'strides "
                f"must not be zero' -- so the refusal lands a stage later in "
                f"another repository. A unit axis does not move, which is what "
                f"the zero means; write 1, which names the same address. The IR "
                f"it produced is {artifact}.",
                artifact=artifact)


def _check_forbids(ctx, name, module, artifact):
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
