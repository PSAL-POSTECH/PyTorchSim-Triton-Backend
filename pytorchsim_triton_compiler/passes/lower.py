"""Step lower: 08-transformed.mlir -> TO_CUSTOM -> 09-custom.mlir, the post-vcix IR the TOG reads,
-> TO_LLVM -> 10-lowered.mlir, nothing but LLVM, func and vcix. A pass with MARKERS runs only when one of
them is in the module; with ctx.dump_passes, every pass's IR goes to 09-passes/, numbered in the order
they ran."""
import os
import shutil
import tempfile
import time

from ..driver import machine
from .. import toolchain
from ..contract.kernel_object import ITEMSIZE
from ..contract.target import TargetError
from ..driver import timer
from ..driver.stage import NotAPrefetchCandidate, SpadOverflow, StageError
from . import TargetPasses, expand, forbidden, forbidden_ops, names, pass_name
from .utils import matmul
from .transforms import (
    place_transfer_waits, legalize_transfers, plan_copy_node, mark_runtime_attributes,
    lower_runtime_attributes, materialize_spad_buffers)
from .conversion import transfers_to_dma, pytorchsim_to_llvm
from .analysis import verify_supported_math, verify_spad_containment, verify_lane_values_stay_vector
from .conversion import to_vcix, pytorchsim_extern_to_vcix
from .utils.ir_utils import module_facts, walk
from .utils.register_groups import LaneBroadcastRefused, LaneReductionRefused
from ..dialect.pytorchsim import GatherOp, LaneIdOp
from .utils.bank_layout import DISTRIBUTED_REDUCTION_OP
from ..targets.compiler import PassGateError
from ..targets import target

# Where the target's LOWER_PASSES run: every pytorchsim op its units compute is its own op after them.
TARGET_LOWER = TargetPasses("LOWER_PASSES", matmul.MATMUL_OPS + (
    LaneIdOp.OPERATION_NAME, DISTRIBUTED_REDUCTION_OP, GatherOp.OPERATION_NAME))

TO_CUSTOM = [
    TARGET_LOWER,
    to_vcix,
    pytorchsim_extern_to_vcix,
    place_transfer_waits,
    legalize_transfers,
    plan_copy_node,
    mark_runtime_attributes,
]

# What TO_CUSTOM's and TO_LLVM's outputs must hold, checked once each list is done (the pipeline's
# invariant checks, not passes); the scratchpad's containment is no check here: materialize-spad-buffers
# runs verify_spad_containment first, as its precondition.
CUSTOM_CHECKS = [verify_supported_math]

TO_LLVM = [
    lower_runtime_attributes,
    materialize_spad_buffers,
    transfers_to_dma,
    pytorchsim_to_llvm,
]


LLVM_CHECKS = [verify_lane_values_stay_vector]


def _to_custom(ctx, src, custom, passes, dump, subtarget, dims):
    """TO_CUSTOM over 08-transformed.mlir into 09-custom.mlir, stamped with the dot widths it built at."""
    from . import widths
    ctx.dot_widths.clear()
    try:
        run_module_passes(src, custom, passes, dump=dump, checks=CUSTOM_CHECKS,
                          forbids_extra=ctx.spec.extra,
                          subtarget=subtarget, descriptor_dims=dims, grid=tuple(ctx.spec.grid),
                          lmul=ctx.lmul, dot_lmul=dict(ctx.dot_lmul), dot_widths=ctx.dot_widths,
                          arg_tensors=[(tuple(a.shape), ITEMSIZE[a.dtype]) for a in ctx.spec.args],
                          graph_prefetch=ctx.spec.graph_prefetch)
    except (SpadOverflow, NotAPrefetchCandidate) as e:
        e.artifact = e.artifact or src
        raise
    except Exception as e:  # noqa: BLE001
        raise StageError("lower", str(e), artifact=src, stderr=repr(e)) from e
    widths.stamp_file(custom, dot_lmul=widths.decision(ctx.lmul, dot_lmul=ctx.dot_lmul)["dot_lmul"])


def recipe():
    try:
        t = machine.subtarget()
    except TargetError:
        t = None
    return [" -> ".join(names(TO_CUSTOM)) + " -> 09-custom.mlir"
            + (f"   (subtarget {t.name})" if t else ""),
            " -> ".join(map(pass_name, TO_LLVM)) + " -> 10-lowered.mlir"]


def run(ctx):
    """TO_CUSTOM then TO_LLVM, whose passes are dumped once the widths are settled. A lower from an input and
    widths an earlier lower of this run had writes the 09-custom.mlir, dumps and matmul streams TO_CUSTOM made
    then (ctx.custom_kept, one per input and widths) instead of running it again."""
    src = ctx.path("08-transformed.mlir")
    custom = ctx.path("09-custom.mlir")
    out = ctx.path("10-lowered.mlir")
    toolchain.ensure_bindings()

    pass_dir = pass_dir_for(ctx, "09-passes")
    step = [0]
    mark = [time.perf_counter()]

    def dump(name, module):
        timer.PROF.add_pass(name, time.perf_counter() - mark[0])
        step[0] += 1
        if ctx.dump_passes:
            with open(os.path.join(pass_dir, f"{step[0]:02d}-{name}.mlir"), "w") as f:
                f.write(str(module))
        mark[0] = time.perf_counter()

    def lap(name):
        timer.PROF.add_pass(name, time.perf_counter() - mark[0])
        mark[0] = time.perf_counter()

    subtarget = machine.subtarget()
    try:
        passes = TO_CUSTOM
        dims = target(subtarget).DMA_DIMS
    except PassGateError as e:
        raise StageError("lower", str(e), artifact=src) from e
    ctx.log(f"[lower] TO_CUSTOM for {subtarget.name} ({', '.join(subtarget.isa)}): "
            + " -> ".join(names(passes)))
    with open(src) as f:
        key = (f.read(), ctx.lmul, tuple(sorted(ctx.dot_lmul.items())))
    if not hasattr(ctx, "custom_kept"):
        ctx.custom_kept = {}
    kept = ctx.custom_kept.get(key)
    if kept is not None:
        text, dumped, streams = kept
        for name, body in dumped:
            step[0] += 1
            with open(os.path.join(pass_dir, f"{step[0]:02d}-{name}.mlir"), "w") as f:
                f.write(body)
        with open(custom, "w") as f:
            f.write(text)
        ctx.dot_widths.clear()
        ctx.dot_widths.update(streams)
        mark[0] = time.perf_counter()
    else:
        _to_custom(ctx, src, custom, passes, dump, subtarget, dims)
        dumped = ([(n[3:-5], open(os.path.join(pass_dir, n)).read()) for n in sorted(os.listdir(pass_dir))]
                  if ctx.dump_passes else [])
        ctx.custom_kept[key] = (open(custom).read(), dumped, dict(ctx.dot_widths))
    ctx.log(f"[lower] TO_CUSTOM done ({', '.join(names(passes))})")
    if ctx.dot_widths:
        ctx.log("[lower] matmul streams: " + ", ".join(f"LMUL {w} in dot {n}"
                                                      for n, w in sorted(ctx.dot_widths.items())))
    for why in kept_apart(custom):
        ctx.log(f"[lower] {matmul.KEPT_APART}: {why}")

    try:
        reserved = to_llvm(custom, out, dump=dump if ctx.loops_settled else lambda name, _: lap(name),
                           subtarget=subtarget, log=ctx.log, promote=ctx.loops_settled, lap=lap)
    except (LaneBroadcastRefused, LaneReductionRefused,
            verify_spad_containment.SpadOverrun,
            materialize_spad_buffers.SpadAllocRefused,
            pytorchsim_to_llvm.SpadReservationRefused,
            verify_lane_values_stay_vector.LaneValueMadeScalar,
            verify_lane_values_stay_vector.ScratchpadAccessRefused) as e:
        raise StageError("lower", str(e), artifact=custom) from e
    ctx.log(f"[lower] scratchpad reserved for {len(reserved)} global(s): "
            + " ".join(f"{n}={c}x{e}" for n, c, e in reserved))
    if ctx.dump_passes:
        ctx.log(f"[lower] per-pass IR in {pass_dir}/")
    ctx.log("[lower] standard lowering to LLVM dialect done")

    _lower_without_records(ctx, custom, out, subtarget)

    residue = sorted({name.split(".")[0] for name in ops_in_file(out)}
                     - {"llvm", "builtin", "func", "vcix"})
    if residue:
        raise StageError("lower", f"non-LLVM dialects survive lowering: {residue}", artifact=out)
    return out


def run_module_passes(in_path, out_path, passes, dump=None, checks=(), **opts):
    """Parse `in_path` once and run, in order, each pass that has no MARKERS or one of whose MARKERS is an op
    of the module when its turn comes, then every check of `checks`; print once to `out_path`, or copy
    `in_path` when no pass ran."""
    from mlir.ir import Context, Module, Location
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    ran = False
    with ctx, Location.unknown():
        with open(in_path) as f:
            module = Module.parse(f.read())
        held = ops_held(module)
        for p in expand(passes):
            if isinstance(p, TargetPasses):
                left = forbidden_ops(p.FORBIDS, module)
                if left:
                    shown = ", ".join(f"{n} x{c}" for n, c, _ in left)
                    raise RuntimeError(f"{p} ({', '.join(names([p]))}) left {shown}, which every pass after "
                                       f"them assumes gone. First one: {left[0][2]}")
                continue
            if hasattr(p, "MARKERS") and held.isdisjoint(p.MARKERS):
                continue
            p.run(module, **opts)
            ran = True
            nm = pass_name(p)
            ops = list(walk(module.operation))
            left = forbidden(nm, module, opts.get("forbids_extra"), ops)
            if left:
                shown = ", ".join(f"{n} x{c}" for n, c, _ in left)
                raise RuntimeError(
                    f"{nm} declared these would not survive it and they did: "
                    f"{shown}. First one: {left[0][2]}")
            if dump is not None:
                dump(nm, module)
            held = {op.name for op in ops}
        for check in checks:
            check.run(module, **opts)
        out = module.operation.get_asm(enable_debug_info=True) if ran else None

    if out is None:
        if out_path != in_path:
            shutil.copyfile(in_path, out_path)
        return False
    write_atomic(out_path, out)
    return True


def to_llvm(in_path, out_path, timing=False, dump=None, *, subtarget, log=None, records_kept=True,
            promote=True, checked=True, lap=None):
    """TO_LLVM as `subtarget` runs it over `in_path`, written to `out_path`; return the scratchpad
    reservations. timing=True erases the DMA, for the one-tile binary gem5 times; records_kept=False
    the runtime attribute records alone; `log(line)` takes what a pass reports; promote=False keeps
    one-trip loops loops while the retry sets widths; checked=False skips the checks, for a cut of a kernel
    whose own lower ran them; `lap(name)` times a step that writes no dump."""
    from mlir.ir import Context, Module, Location
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        with open(in_path) as f:
            module = Module.parse(f.read())
        reserved = None
        for p in TO_LLVM:
            got = p.run(module, timing=timing, vectorlane=subtarget.vector_lanes, subtarget=subtarget,
                        dump=dump, log=log, records_kept=records_kept, promote=promote, checked=checked,
                        lap=lap)
            if p is pytorchsim_to_llvm:
                reserved = got
            elif dump is not None:
                dump(pass_name(p), module)
        for check in LLVM_CHECKS if checked else ():
            check.run(module, subtarget=subtarget)
            if lap is not None:
                lap(pass_name(check))
        # every function a debug scope, so the loop locations legalize gave survive into the LLVM
        # IR as locations llc's regalloc remarks report (elf/build.py, read_places)
        from mlir.passmanager import PassManager
        PassManager.parse("builtin.module(ensure-debug-info-scope-on-llvm-func)").run(
            module.operation)
        with open(out_path, "w") as f:
            f.write(module.operation.get_asm(enable_debug_info=True))
        module_facts(out_path, module)
    return reserved


NO_RECORDS = "10-lowered-no-records.mlir"
"""09-custom.mlir lowered with its runtime attribute records erased, when it has any: what the binary
step decides the transform's widths on, since the timed binary has no records either."""


def _lower_without_records(ctx, custom, out, subtarget):
    from .transforms.lower_runtime_attributes import TABLE
    path = ctx.path(NO_RECORDS)
    if os.path.exists(path):
        os.remove(path)
    if not module_states(out, TABLE):
        return
    with timer.PROF.step("no-records-lower"):
        to_llvm(custom, path, subtarget=subtarget, records_kept=False, promote=ctx.loops_settled,
                checked=False)
    ctx.log(f"[lower] {NO_RECORDS}: the records erased, for the spill check the widths answer to")


def module_states(path, name):
    """Whether the module at `path` carries the attribute `name`."""
    from .utils.ir_utils import module_facts
    return name in module_facts(path)[0]


def ops_in_file(path):
    """The name of every op in the module at `path`, read from a parse of it."""
    from mlir.ir import Context, Module
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, open(path) as f:
        return ops_held(Module.parse(f.read()))


def kept_apart(path):
    """What each op of the module at `path` stamped with matmul.KEPT_APART says: why a loop
    beside a dot was left a loop."""
    from mlir.ir import Context, Module, StringAttr
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, open(path) as f:
        module = Module.parse(f.read())
        return [StringAttr(o.attributes[matmul.KEPT_APART]).value
                for o in walk(module.operation) if matmul.KEPT_APART in o.attributes]


def ops_held(module):
    """The name of every op under `module` -- what a pass's MARKERS, op names all, are read
    against: a pass has work in a module that holds one of them."""
    return {op.name for op in walk(module.operation)}


def pass_dir_for(ctx, name):
    """The workdir's `name` dump directory, emptied when this run writes it (ctx.dump_passes) and gone
    when it does not: either way no dump of an earlier run is left to read as this one's."""
    path = ctx.path(name)
    if ctx.dump_passes:
        return fresh_dir(path)
    shutil.rmtree(path, ignore_errors=True)
    return path


def fresh_dir(path):
    """`path` as an empty directory: a pass dump left from an earlier run would read as this one's."""
    os.makedirs(path, exist_ok=True)
    for stale in os.listdir(path):
        os.remove(os.path.join(path, stale))
    return path


def write_atomic(out_path, text):
    """`text` into `out_path` through a temporary file beside it, so no reader sees it half written."""
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
