"""Step lower: 08-transformed.mlir -> 09-custom.mlir -> 10-lowered.mlir.

  TO_CUSTOM  p08 lower-matmul, p09 lower-to-vcix, p10 lower-to-xlu, p11 fit-transfers-to-descriptor
             -> 09-custom.mlir, the post-vcix IR the TOG reads; a pass runs only when one of its
             MARKERS is an op of the module
  TO_LLVM    p12 lower-to-dma, p13 lower-to-llvm -> 10-lowered.mlir, nothing but LLVM, func and vcix

Every pass's IR goes to 09-passes/, numbered in the order the passes ran."""
import os
import shutil
import tempfile
import time

from ... import machine, toolchain
from ...metrics import wallclock
from ...stage import StageError
from . import (forbidden, p08_lower_matmul, p08_lower_matmul_mxu, p09_lower_to_vcix,
               p10_lower_to_xlu, p11_fit_transfers_to_descriptor, p12_lower_to_dma,
               p13_lower_to_llvm, pass_name)
from .lib_ir import walk

TO_CUSTOM = [
    p08_lower_matmul,
    p09_lower_to_vcix,
    p10_lower_to_xlu,
    p11_fit_transfers_to_descriptor,
]

TO_LLVM = [
    p12_lower_to_dma,
    p13_lower_to_llvm,
]

def to_custom():
    """TO_CUSTOM for the target's accelerator: the TPU v6 model has a matmul lowering of its own,
    and every other target the list as it stands."""
    if machine.target().accel == "vcix_tpuv6":
        return [p08_lower_matmul_mxu if p is p08_lower_matmul else p for p in TO_CUSTOM]
    return TO_CUSTOM


def recipe():
    return [" -> ".join(pass_name(p) for p in TO_CUSTOM) + " -> 09-custom.mlir",
            " -> ".join(pass_name(p) for p in TO_LLVM) + " -> 10-lowered.mlir"]


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

    try:
        run_module_passes(src, custom, to_custom(), dump=dump, forbids_extra=ctx.spec.extra,
                          vectorlane=machine.VECTORLANE_SIZE, vlen=machine.VLEN_BITS,
                          target=machine.target())
    except Exception as e:  # noqa: BLE001
        raise StageError("lower", str(e), artifact=src, stderr=repr(e))
    ctx.log(f"[lower] TO_CUSTOM done ({', '.join(pass_name(m) for m in to_custom())})")
    for why in kept_apart(custom):
        ctx.log(f"[lower] {p08_lower_matmul.KEPT_APART}: {why}")

    try:
        reserved = to_llvm(custom, out, dump=dump, vectorlane=machine.VECTORLANE_SIZE)
    except p13_lower_to_llvm.SpadReservationRefused as e:
        raise StageError("lower", str(e), artifact=custom) from None
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


def to_llvm(in_path, out_path, timing=False, dump=None, *, vectorlane):
    """TO_LLVM over `in_path`, written to `out_path`; return the scratchpad reservations. timing=True
    erases the DMA, for the one-tile binary gem5 times."""
    from mlir.ir import Context, Module, Location
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        with open(in_path) as f:
            module = Module.parse(f.read())
        p12_lower_to_dma.run(module, timing=timing, vectorlane=vectorlane)
        if dump is not None:
            dump(pass_name(p12_lower_to_dma), module)
        reserved = p13_lower_to_llvm.run(module, vectorlane=vectorlane, dump=dump)
        with open(out_path, "w") as f:
            f.write(str(module))
    return reserved


def ops_in_file(path):
    """The name of every op in the module at `path`, read from a parse of it."""
    from mlir.ir import Context, Module
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, open(path) as f:
        return ops_held(Module.parse(f.read()))


def kept_apart(path):
    """What each op of the module at `path` stamped with p08's KEPT_APART says: why a loop beside a
    dot was left a loop."""
    from mlir.ir import Context, Module, StringAttr
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, open(path) as f:
        module = Module.parse(f.read())
        return [StringAttr(o.attributes[p08_lower_matmul.KEPT_APART]).value
                for o in walk(module.operation) if p08_lower_matmul.KEPT_APART in o.attributes]


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
