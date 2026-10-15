"""Step lower: 08-transformed.mlir -> TO_CUSTOM -> 09-custom.mlir, the post-vcix IR the TOG reads,
-> TO_LLVM -> 10-lowered.mlir, nothing but LLVM, func and vcix. A pass runs only when one of its
MARKERS is in the module; every pass's IR goes to 09-passes/, numbered in the order they ran."""
import os
import shutil
import tempfile
import time

from ... import machine, toolchain
from ...contract.target import TargetError
from ...metrics import wallclock
from ...dialects.torchsim import VcixOp, VLaneIdxOp
from ...stage import StageError
from . import (forbidden, lib_matmul, p08_refuse_unlowered_math,
               p11_fit_transfers_to_descriptor, p12_lower_lane_broadcasts,
               p12a_lower_lane_reductions, p13_lower_to_llvm,
               pass_name)
from .lib_cross_lane import CROSS_LANE_OP
from .lib_ir import walk
from .slots import PassGateError, Slot, resolve, spelled

LOWER_MATMUL = Slot("lower-matmul", lowers=lib_matmul.MATMUL_OPS)

SCHEDULE_MATMUL = Slot("schedule-matmul")

LOWER_VECTOR_FUNCTIONS = Slot("lower-vector-functions", lowers=(VLaneIdxOp.OPERATION_NAME,))

LOWER_CROSS_LANE = Slot("lower-cross-lane", lowers=(CROSS_LANE_OP,))

LOWER_KERNEL_VCIX = Slot("lower-kernel-vcix", lowers=(VcixOp.OPERATION_NAME,))

TO_CUSTOM = [
    LOWER_MATMUL,
    SCHEDULE_MATMUL,
    LOWER_VECTOR_FUNCTIONS,
    p08_refuse_unlowered_math,
    LOWER_CROSS_LANE,
    LOWER_KERNEL_VCIX,
    p11_fit_transfers_to_descriptor,
]

LOWER_DMA = Slot("lower-dma", exactly_one=True)

TO_LLVM = [
    p12_lower_lane_broadcasts,
    p12a_lower_lane_reductions,
    LOWER_DMA,
    p13_lower_to_llvm,
]

REDUCE_INTRINSIC = "llvm.intr.vector.reduce."


class LaneValueScalarized(Exception):
    """TO_LLVM left an op that makes a lane's own value a scalar, which the one scalar register
    file all lanes share holds for lane 0 only."""


def pipelines():
    """(title, list, whether the machine is known) of every list this step runs."""
    return [("TO_CUSTOM", TO_CUSTOM, True), ("TO_LLVM", TO_LLVM, True)]


def to_custom(target):
    """TO_CUSTOM as `target` runs it."""
    return resolve(TO_CUSTOM, target)


def dma_lowering(target):
    """The pass filling the lower-dma slot on `target`: the descriptor's owner, whose DESCRIPTOR_DIMS
    p11 fits to and whose insn() the ELF wrapper spells its index key with."""
    return next(m for m in resolve(TO_LLVM, target) if getattr(m, "SLOT", None) == LOWER_DMA.name)


def recipe():
    try:
        t = machine.target()
    except TargetError:
        t = None
    return [" -> ".join(spelled(TO_CUSTOM, t)) + " -> 09-custom.mlir"
            + (f"   (target {t.name})" if t else ""),
            " -> ".join(spelled(TO_LLVM, t)) + " -> 10-lowered.mlir"]


def run(ctx):
    src = ctx.path("08-transformed.mlir")
    custom = ctx.path("09-custom.mlir")
    out = ctx.path("10-lowered.mlir")
    toolchain.ensure_bindings()

    pass_dir = fresh_dir(ctx.path("09-passes"))
    step = [0]
    mark = [time.perf_counter()]

    def dump(name, module):
        wallclock.PROF.add_pass(name, time.perf_counter() - mark[0])
        step[0] += 1
        with open(os.path.join(pass_dir, f"{step[0]:02d}-{name}.mlir"), "w") as f:
            f.write(str(module))
        mark[0] = time.perf_counter()

    target = machine.target()
    try:
        passes = to_custom(target)
        dims = dma_lowering(target).DESCRIPTOR_DIMS
    except PassGateError as e:
        raise StageError("lower", str(e), artifact=src) from None
    ctx.log(f"[lower] TO_CUSTOM for {target.name} ({', '.join(target.isa)}): "
            + " -> ".join(pass_name(m) for m in passes))
    try:
        run_module_passes(src, custom, passes, dump=dump, forbids_extra=ctx.spec.extra,
                          target=target, descriptor_dims=dims, grid=tuple(ctx.spec.grid))
    except Exception as e:  # noqa: BLE001
        raise StageError("lower", str(e), artifact=src, stderr=repr(e))
    ctx.log(f"[lower] TO_CUSTOM done ({', '.join(pass_name(m) for m in passes)})")
    for why in kept_apart(custom):
        ctx.log(f"[lower] {lib_matmul.KEPT_APART}: {why}")

    try:
        reserved = to_llvm(custom, out, dump=dump, target=target)
    except (p12_lower_lane_broadcasts.LaneBroadcastRefused,
            p12a_lower_lane_reductions.LaneReductionRefused,
            p13_lower_to_llvm.SpadReservationRefused) as e:
        raise StageError("lower", str(e), artifact=custom) from None
    except LaneValueScalarized as e:
        raise StageError("lower", str(e), artifact=out) from None
    ctx.log(f"[lower] scratchpad reserved for {len(reserved)} global(s): "
            + " ".join(f"{n}={c}x{e}" for n, c, e in reserved))
    ctx.log(f"[lower] per-pass IR in {pass_dir}/")
    ctx.log("[lower] standard lowering to LLVM dialect done")

    residue = sorted({name.split(".")[0] for name in ops_in_file(out)}
                     - {"llvm", "builtin", "func", "vcix"})
    if residue:
        raise StageError("lower", f"non-LLVM dialects survive lowering: {residue}", artifact=out)
    return out


def run_module_passes(in_path, out_path, passes, dump=None, **opts):
    """Parse `in_path` once and run, in order, each pass one of whose MARKERS is an op of the module
    when its turn comes; print once to `out_path`, or copy `in_path` when no pass ran."""
    from mlir.ir import Context, Module, Location
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    ran = False
    with ctx, Location.unknown():
        with open(in_path) as f:
            module = Module.parse(f.read())
        held = ops_held(module)
        for p in passes:
            if held.isdisjoint(p.MARKERS):
                continue
            p.run(module, **opts)
            ran = True
            nm = pass_name(p)
            left = forbidden(nm, module, opts.get("forbids_extra"))
            if left:
                shown = ", ".join(f"{n} x{c}" for n, c, _ in left)
                raise RuntimeError(
                    f"{nm} declared these would not survive it and they did: "
                    f"{shown}. First one: {left[0][2]}")
            if dump is not None:
                dump(nm, module)
            held = ops_held(module)
        out = str(module) if ran else None

    if out is None:
        if out_path != in_path:
            shutil.copyfile(in_path, out_path)
        return False
    _write_atomic(out_path, out)
    return True


def to_llvm(in_path, out_path, timing=False, dump=None, *, target):
    """TO_LLVM as `target` runs it over `in_path`, written to `out_path`; return the scratchpad
    reservations. timing=True erases the DMA, for the one-tile binary gem5 times."""
    from mlir.ir import Context, Module, Location
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        with open(in_path) as f:
            module = Module.parse(f.read())
        reserved = None
        for p in resolve(TO_LLVM, target):
            got = p.run(module, timing=timing, vectorlane=target.vector_lanes, target=target,
                        dump=dump)
            if p is p13_lower_to_llvm:
                reserved = got
            elif dump is not None:
                dump(pass_name(p), module)
        with open(out_path, "w") as f:
            f.write(str(module))
        made = lane_values_made_scalar(module)
        if made:
            raise LaneValueScalarized(
                f"TO_LLVM left {len(made)} op(s) that make a lane's value a scalar, which every "
                f"lane's one shared scalar register file holds for lane 0 only. First: "
                f"{' '.join(str(made[0]).split())[:200]} at {made[0].location}; IR in {out_path}")
    return reserved


def lane_values_made_scalar(module):
    """Every op of `module` taking a scalar out of a lane's value: an llvm.extractelement of a
    one-element vector, a vector.extract to a scalar, a vector.reduction or its LLVM intrinsic."""
    from mlir.ir import VectorType
    made = []
    for op in walk(module.operation):
        n = op.name
        if n == "llvm.extractelement":
            src = op.operands[0].type
            hit = isinstance(src, VectorType) and all(d == 1 for d in VectorType(src).shape)
        elif n == "vector.extract":
            hit = not isinstance(op.results[0].type, VectorType)
        else:
            hit = n == "vector.reduction" or n.startswith(REDUCE_INTRINSIC)
        if hit:
            made.append(op)
    return made


def ops_in_file(path):
    """The name of every op in the module at `path`, read from a parse of it."""
    from mlir.ir import Context, Module
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, open(path) as f:
        return ops_held(Module.parse(f.read()))


def kept_apart(path):
    """What each op of the module at `path` stamped with lib_matmul.KEPT_APART says: why a loop
    beside a dot was left a loop."""
    from mlir.ir import Context, Module, StringAttr
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, open(path) as f:
        module = Module.parse(f.read())
        return [StringAttr(o.attributes[lib_matmul.KEPT_APART]).value
                for o in walk(module.operation) if lib_matmul.KEPT_APART in o.attributes]


def ops_held(module):
    """The name of every op under `module` -- what a pass's MARKERS, op names all, are read
    against: a pass has work in a module that holds one of them."""
    return {op.name for op in walk(module.operation)}


def fresh_dir(path):
    """`path` as an empty directory: a pass dump left from an earlier run would read as this one's."""
    os.makedirs(path, exist_ok=True)
    for stale in os.listdir(path):
        os.remove(os.path.join(path, stale))
    return path


def _write_atomic(out_path, text):
    d = os.path.dirname(os.path.abspath(out_path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, out_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
