"""Step transform: 07-regularized.mlir -> 08-transformed.mlir, PHASES in order. Every pass's IR goes
to 08-passes/ and is held there to three checks: no memref stride of zero, nothing left the pass
FORBIDS, and -- on the input -- no element type the machine does not run."""
import os
import time

from ..driver import machine
from .. import toolchain
from ..contract.stop import Reason, Refusal
from ..contract.target import TargetError
from ..driver import stats, timer
from ..driver.stage import StageError
from .transforms import (
    expand_unsupported_ops, materialize_layouts, fuse_elementwise_ops, drop_unread_padding, bufferize,
    vectorize, legalize_vectors, simplify_index_arithmetic, hoist_loop_invariants)
from .conversion import tts_to_pytorchsim
from . import TargetPasses, expand, forbidden, forbidden_ops, forbids, names, pass_name
from .analysis import verify_lane_uniformity
from ..dialect.pytorchsim import ConvertLayoutOp
from .utils.ir_utils import LANES_ATTR, elem_type, funcs, lanes_of_module, values, walk
from .utils.linear_layout import get_bases
from .lower import fresh_dir
from ..targets.compiler import PassGateError

# Where the target's TRANSFORM_PASSES run: none of them may leave a pytorchsim.convert_layout.
TARGET_TRANSFORM = TargetPasses("TRANSFORM_PASSES", (ConvertLayoutOp.OPERATION_NAME,))

PHASES = [
    ("tts -> transfers", [expand_unsupported_ops, tts_to_pytorchsim]),
    ("layout made into shapes", [materialize_layouts]),
    ("fusion after layout", [fuse_elementwise_ops]),
    ("tiles fit to their numel", [drop_unread_padding]),
    ("tensor -> memref", [bufferize]),
    ("the machine is known", [vectorize, legalize_vectors, simplify_index_arithmetic,
                              hoist_loop_invariants,
                              TARGET_TRANSFORM]),
]

# What the transform's output must hold, checked once its passes are done: run as the pipeline's
# invariant checks, after the last pass, not as passes of their own.
CHECKS = [verify_lane_uniformity]

MACHINE_KNOWN = "the machine is known"

SUPPORTED_DTYPES = frozenset((
    "i1", "i8", "i16", "i32", "i64",
    "f16", "f32", "f64", "bf16",
))


def recipe():
    try:
        t = machine.subtarget()
    except TargetError:
        t = None
    return [f"{title}: " + " -> ".join(names(ps)) for title, ps in PHASES] + [
        "-> 08-transformed.mlir" + (f"   (subtarget {t.name})" if t else "")]


def run(ctx):
    """08-transformed.mlir keeps its locations: legalize's loop numbers ride them to llc
    (legalize_vectors.LOOP_FILE)."""
    from . import widths
    src, out = ctx.path("07-regularized.mlir"), ctx.path("08-transformed.mlir")
    toolchain.ensure_bindings()
    from mlir.ir import Context, Module, Location

    with open(src) as f:
        text = f.read()
    ran_phases = PHASES

    opts = dict(arg_numels={i: a.numel for i, a in enumerate(ctx.spec.args)},
                grid=tuple(ctx.spec.grid))
    known = dict(opts, lanes=machine.VECTORLANE_SIZE, vlen=machine.VLEN_BITS, lmul=ctx.lmul,
                 loop_lmul=dict(getattr(ctx, "loop_lmul", {})))
    first_known = [title for title, _ in ran_phases].index(MACHINE_KNOWN)
    mlir_ctx = Context()
    mlir_ctx.allow_unregistered_dialects = True
    pass_dir = fresh_dir(ctx.path("08-passes"))
    ran = []
    mark = [time.perf_counter()]

    def dump(name, module):
        timer.PROF.add_pass(name, time.perf_counter() - mark[0])
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
        _refuse_other_lane_count(module, src)
        _refuse_unsupported_dtypes(module)
        for p, at in ((p, i >= first_known) for i, (_, ps) in enumerate(ran_phases) for p in expand(ps)):
            if isinstance(p, TargetPasses):
                _check_target_left(p, module, src)
                continue
            try:
                p.run(module, **(known if at else opts))
            except PassGateError as e:
                raise StageError("transform", str(e), artifact=src) from e
            dump(pass_name(p), module)
        for check in CHECKS:
            check.run(module, **known)
        d = widths.decision(ctx.lmul, getattr(ctx, "loop_lmul", {}))
        widths.stamp(module, lmul=d["lmul"], loop_lmul=d["loop_lmul"])
        counts = stats.count(module)
        uniform = stats.count_uniform(module)
        result = module.operation.get_asm(enable_debug_info=True)
    with open(out, "w") as f:
        f.write(result)

    ctx.log(f"[transform] per-pass IR in {pass_dir}/")
    for title, ps in ran_phases:
        ctx.log(f"[transform] {title}: " + " -> ".join(names(ps)))
    stats.write(ctx.workdir, counts, uniform)
    moves = counts["moves"]
    ctx.log(f"[transform] DMA: {counts['loads']} loads, {counts['stores']} stores; "
            f"moves on the target's units and lane reduces: {sum(moves.values())}"
            + (" (" + ", ".join(f"{k} {n}" for k, n in sorted(moves.items()))
               + ")" if moves else ""))
    return out


def _refuse_other_lane_count(module, src):
    """The one place the IR's lane count meets the machine's: the module's LANES_ATTR, and the lane
    bases of every #ttg.linear in it, must say the machine's lanes, since below here passes read the
    lane count from either (an 07 kept from another subtarget, or edited out of step, stops here)."""
    from mlir.ir import RankedTensorType
    try:
        said = lanes_of_module(module)
    except ValueError as e:
        raise StageError("transform", str(e), artifact=src) from None
    if said != machine.VECTORLANE_SIZE:
        raise StageError("transform", f"the module was made for {said} lanes ({LANES_ATTR}) and "
                         f"the machine has {machine.VECTORLANE_SIZE}", artifact=src)
    for v in values(module.operation):
        if not isinstance(v.type, RankedTensorType) or v.type.encoding is None:
            continue
        bases = get_bases(v.type.encoding)
        if bases is not None and 1 << len(bases[1]) != said:
            raise StageError("transform", f"a tile of {1 << len(bases[1])} lanes in a module made for "
                             f"{said} ({LANES_ATTR}): {v.type}", artifact=src)


class DtypeNotRun(NotImplementedError, Refusal):
    """An element type this machine does not run."""
    reason = Reason.DTYPE_NOT_RUN


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
                        raise DtypeNotRun(
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


def _check_target_left(place, module, artifact):
    """Refuse what `place`'s passes (the target's) left of its FORBIDS, which no common pass takes."""
    left = forbidden_ops(place.FORBIDS, module)
    if left:
        shown = ", ".join(f"{n} x{c}" for n, c, _ in left)
        raise StageError("transform", f"{place} ({', '.join(names([place]))}) left {shown}, which every pass "
                         f"after them assumes gone. First at {left[0][2]}. The IR it ran on is {artifact}.",
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
