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
from ...dialects.pytorchsim import VcixOp, VLaneIdxOp
from ...stage import StageError
from . import (forbidden, lib_matmul, p08_refuse_unlowered_math,
               p11_fit_transfers_to_descriptor, p11a_mark_runtime_attributes,
               p11b_lower_runtime_attributes, p12_lower_lane_broadcasts,
               p12a_lower_lane_reductions, p12aa_check_spad_containment,
               p12ab_materialize_spad_allocs, p12b_share_spad_buffers,
               p13_lower_to_llvm, p13b_check_lane_values_stay_vector,
               pass_name)
from .lib_bank_layout import DISTRIBUTED_REDUCTION_OP
from .lib_ir import walk
from .slots import PassGateError, Slot, pack, resolve, spelled

LOWER_MATMUL = Slot("lower-matmul", lowers=lib_matmul.MATMUL_OPS)

SCHEDULE_MATMUL = Slot("schedule-matmul")

LOWER_VECTOR_FUNCTIONS = Slot("lower-vector-functions", lowers=(VLaneIdxOp.OPERATION_NAME,))

# Where the vector loops' units are placed once their functions are lowered: pipelining across trips.
SCHEDULE_VPU = Slot("schedule-vpu")

LOWER_DISTRIBUTED_REDUCTION = Slot("lower-distributed-reduction", lowers=(DISTRIBUTED_REDUCTION_OP,))

LOWER_GATHER = Slot("lower-gather", lowers=("pytorchsim.gather",))

# The ops of the pack's own dialect (pytorchsim_<family>.*), which its earlier passes minted.
LOWER_TARGET_OPS = Slot("lower-target-ops")

LOWER_KERNEL_VCIX = Slot("lower-kernel-vcix", lowers=(VcixOp.OPERATION_NAME,))

TO_CUSTOM = [
    LOWER_MATMUL,
    SCHEDULE_MATMUL,
    LOWER_VECTOR_FUNCTIONS,
    SCHEDULE_VPU,
    p08_refuse_unlowered_math,
    LOWER_DISTRIBUTED_REDUCTION,
    LOWER_GATHER,
    LOWER_TARGET_OPS,
    LOWER_KERNEL_VCIX,
    p11_fit_transfers_to_descriptor,
    p11a_mark_runtime_attributes,
]

LOWER_DMA = Slot("lower-dma", exactly_one=True)

TO_LLVM = [
    p11b_lower_runtime_attributes,
    p12_lower_lane_broadcasts,
    p12a_lower_lane_reductions,
    p12aa_check_spad_containment,
    p12ab_materialize_spad_allocs,
    p12b_share_spad_buffers,
    LOWER_DMA,
    p13_lower_to_llvm,
    p13b_check_lane_values_stay_vector,
]


def pipelines():
    """(title, list, whether the machine is known) of every list this step runs."""
    return [("TO_CUSTOM", TO_CUSTOM, True), ("TO_LLVM", TO_LLVM, True)]


def to_custom(target):
    """TO_CUSTOM as `target` runs it."""
    return resolve(TO_CUSTOM, target)


def dma_lowering(target):
    """The pass filling the lower-dma slot on `target` (slots.DmaProvider): the descriptor's owner, whose
    DESCRIPTOR_DIMS p11 fits to and whose insn() the ELF wrapper spells its index key with."""
    return pack(target)[LOWER_DMA.name]


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
        raise StageError("lower", str(e), artifact=src) from e
    ctx.log(f"[lower] TO_CUSTOM for {target.name} ({', '.join(target.isa)}): "
            + " -> ".join(pass_name(m) for m in passes))
    ctx.dot_widths.clear()
    try:
        run_module_passes(src, custom, passes, dump=dump, forbids_extra=ctx.spec.extra,
                          target=target, descriptor_dims=dims, grid=tuple(ctx.spec.grid),
                          lmul=ctx.lmul, dot_lmul=dict(ctx.dot_lmul), dot_widths=ctx.dot_widths)
    except Exception as e:  # noqa: BLE001
        raise StageError("lower", str(e), artifact=src, stderr=repr(e)) from e
    from .. import widths
    widths.stamp_file(custom, dot_lmul=widths.decision(ctx.lmul, dot_lmul=ctx.dot_lmul)["dot_lmul"])
    ctx.log(f"[lower] TO_CUSTOM done ({', '.join(pass_name(m) for m in passes)})")
    if ctx.dot_widths:
        ctx.log("[lower] matmul streams: " + ", ".join(f"LMUL {w} in dot {n}"
                                                      for n, w in sorted(ctx.dot_widths.items())))
    for why in kept_apart(custom):
        ctx.log(f"[lower] {lib_matmul.KEPT_APART}: {why}")

    try:
        reserved = to_llvm(custom, out, dump=dump, target=target, log=ctx.log,
                           promote=ctx.loops_settled)
    except (p12_lower_lane_broadcasts.LaneBroadcastRefused,
            p12a_lower_lane_reductions.LaneReductionRefused,
            p12aa_check_spad_containment.SpadOverrun,
            p12ab_materialize_spad_allocs.SpadAllocRefused,
            p13_lower_to_llvm.SpadReservationRefused,
            p13b_check_lane_values_stay_vector.LaneValueMadeScalar,
            p13b_check_lane_values_stay_vector.ScratchpadAccessRefused) as e:
        raise StageError("lower", str(e), artifact=custom) from e
    ctx.log(f"[lower] scratchpad reserved for {len(reserved)} global(s): "
            + " ".join(f"{n}={c}x{e}" for n, c, e in reserved))
    ctx.log(f"[lower] per-pass IR in {pass_dir}/")
    ctx.log("[lower] standard lowering to LLVM dialect done")

    _lower_without_records(ctx, custom, out, target)

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
        out = module.operation.get_asm(enable_debug_info=True) if ran else None

    if out is None:
        if out_path != in_path:
            shutil.copyfile(in_path, out_path)
        return False
    _write_atomic(out_path, out)
    return True


def to_llvm(in_path, out_path, timing=False, dump=None, *, target, log=None, records_kept=True,
            promote=True):
    """TO_LLVM as `target` runs it over `in_path`, written to `out_path`; return the scratchpad
    reservations. timing=True erases the DMA, for the one-tile binary gem5 times; records_kept=False
    the runtime attribute records alone; `log(line)` takes what a pass reports; promote=False keeps
    one-trip loops loops while the retry sets widths."""
    from mlir.ir import Context, Module, Location
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        with open(in_path) as f:
            module = Module.parse(f.read())
        reserved = None
        for p in resolve(TO_LLVM, target):
            got = p.run(module, timing=timing, vectorlane=target.vector_lanes, target=target,
                        dump=dump, log=log, records_kept=records_kept, promote=promote)
            if p is p13_lower_to_llvm:
                reserved = got
            elif dump is not None:
                dump(pass_name(p), module)
        # every function a debug scope, so the loop locations legalize gave survive into the LLVM
        # IR as locations llc's regalloc remarks report (elf/build.py, read_places)
        from mlir.passmanager import PassManager
        PassManager.parse("builtin.module(ensure-debug-info-scope-on-llvm-func)").run(
            module.operation)
        with open(out_path, "w") as f:
            f.write(module.operation.get_asm(enable_debug_info=True))
    return reserved


NO_RECORDS = "10-lowered-no-records.mlir"
"""09-custom.mlir lowered with its runtime attribute records erased, when it has any: what the binary
step decides the transform's widths on, since the timed binary has no records either."""


def _lower_without_records(ctx, custom, out, target):
    from .p11b_lower_runtime_attributes import TABLE
    path = ctx.path(NO_RECORDS)
    if os.path.exists(path):
        os.remove(path)
    if not module_states(out, TABLE):
        return
    to_llvm(custom, path, target=target, records_kept=False, promote=ctx.loops_settled)
    ctx.log(f"[lower] {NO_RECORDS}: the records erased, for the spill check the widths answer to")


def module_states(path, name):
    """Whether the module at `path` carries the attribute `name`."""
    from mlir.ir import Context, Module
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, open(path) as f:
        return name in Module.parse(f.read()).operation.attributes


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
