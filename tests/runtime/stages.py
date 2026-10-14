"""The stages that run what the compiler built: spike runs the kernel object and verify checks
what it computed; togcheck replays the TOG (--tog); timing runs gem5 and TOGSim
(--timing).

Registered onto the compiler's pipeline by this package's __init__. They live
here and not in it because this repository's suite is the only caller that runs
a compiled program -- the compiler produces one.
"""
import os

from pytorchsim_triton_compiler.contract import kernel_object
from pytorchsim_triton_compiler.contract.kernel_spec import is_read
from pytorchsim_triton_compiler.metrics import wallclock
from pytorchsim_triton_compiler.stage import StageError, run_tool

from . import device, verify
from .device import as_compiled


class _Run:
    """What spike hands verify: the reference inputs and where each tensor's .raw is. The
    compiler's Ctx knows nothing of either -- it builds the program, and this suite runs it."""

    def __init__(self, workdir):
        self.runtime_dir = os.path.join(workdir, "runtime")
        self.inputs = None
        self.raw_paths = {}


def _run(ctx):
    if not hasattr(ctx, "run"):
        ctx.run = _Run(ctx.workdir)
    return ctx.run


# --------------------------------------------------------------------------
# spike -- run the kernel object on Spike
# --------------------------------------------------------------------------
def stage_spike(ctx):
    """Hand the compiled object to a device, having written its tensors first.

    BUILDING THE TENSORS IS NOT RUNNING THE KERNEL, and on a small kernel it is
    the larger half -- this is where torch is imported. Charging it to the
    device reads as the simulator being slow before it has started.
    """
    run = _run(ctx)
    runtime = run.runtime_dir
    os.makedirs(os.path.join(runtime, "dma_access"), exist_ok=True)

    with wallclock.PROF.step("materialise_inputs"):
        _materialise_inputs(ctx, run)

    manifest = kernel_object.read(ctx.workdir)
    kernel_object.refuse_if_it_disagrees(manifest, ctx.spec)

    req = device.Launch(
        obj_dir=ctx.workdir,
        manifest=manifest,
        runtime_dir=runtime,
        raw_paths=run.raw_paths,
        log=ctx.log,
        run=lambda stage, cmd, **kw: run_tool(ctx, stage, cmd, **kw),
        step=wallclock.PROF.step,
        error=lambda message, **kw: StageError("spike", message, **kw))

    device.get(ctx.spec.extra.get("device", "spike")).launch(req)
    ctx.log("[spike] spike finished")
    return runtime



def _materialise_inputs(ctx, run):
    """Generate the reference inputs once and write every tensor as .raw."""
    import numpy as np
    import torch

    if run.inputs is None:
        run.inputs = ctx.spec.make_inputs(torch, seed=0)

    for arg in ctx.spec.args:
        p = os.path.join(run.runtime_dir, f"{arg.name}.raw")
        run.raw_paths[arg.name] = p
        if is_read(arg):
            t = run.inputs[arg.name].contiguous().cpu()
            # BITS, NOT A CAST, for a dtype numpy does not have.
            t = (t.view(torch.uint16) if arg.dtype == "bfloat16" else t)
            t.numpy().astype(_np_dtype(arg.dtype)).tofile(p)
        else:
            np.zeros(arg.numel, dtype=_np_dtype(arg.dtype)).tofile(p)
    ctx.log(f"[spike] inputs written to {run.runtime_dir}")


def _np_dtype(name):
    from .verify import NP_DTYPE
    return NP_DTYPE[name]


# --------------------------------------------------------------------------
# verify -- compare against the torch reference
# --------------------------------------------------------------------------
def stage_verify(ctx):
    import torch
    from . import verify

    run = _run(ctx)
    runtime = run.runtime_dir
    if run.inputs is None:
        run.inputs = ctx.spec.make_inputs(torch, seed=0)
    if not run.raw_paths:
        for arg in ctx.spec.args:
            run.raw_paths[arg.name] = os.path.join(runtime, f"{arg.name}.raw")

    ok, lines = verify.compare(ctx.spec, run.raw_paths, run.inputs, torch)
    for line in lines:
        print(line)
        ctx.log(line)

    # A kernel the linalg lowering itself miscompiles cannot be made to match from
    # here, and leaving it as a bare FAIL means nobody can tell it from a
    # regression we caused. The numbers are still COMPUTED and printed -- what
    # the marker changes is only whether the mismatch is news.
    known = ctx.spec.extra.get("upstream_miscompile")
    if known:
        if ok:
            ctx.warn(f"marked upstream_miscompile but the numbers MATCH now -- "
                     f"re-check the marker: {known}")
        else:
            ctx.warn(f"numbers not trusted (known miscompile in the linalg lowering): "
                     f"{known}")
        return runtime

    if not ok:
        raise StageError("verify", "numerical mismatch against the torch reference")
    return runtime


# --------------------------------------------------------------------------
# the --tog branch -- check the TOG against what was compiled
# --------------------------------------------------------------------------
def stage_togcheck(ctx):
    """Replay tog.so and hold it against kernel.json and the post-vcix IR.

    THE COMPILER EMITS, THE SUITE RUNS: this dlopens what stage `tog` built,
    which the compiler will not do, for the same reason `spike` lives here.
    """
    from . import tog_replay

    problems, notes = tog_replay.check(ctx.workdir)
    for note in notes:
        ctx.warn(f"[togcheck] {note}")
    for problem in problems:
        ctx.log(f"[togcheck] {problem}")
    if problems:
        # ONE LINE, because the sweep reports one line per kernel and reads the
        # first indented line under the banner. The rest are in the log, which
        # the report already names.
        raise StageError("togcheck", problems[0] + (
            f"  (+{len(problems) - 1} more, see the log)"
            if len(problems) > 1 else ""))
    ctx.log(f"[togcheck] the TOG agrees with the manifest and the IR"
            + (f" ({len(notes)} approximation(s) noted)" if notes else ""))
    return ctx.workdir


def time_launch(req):
    """gem5's tile cycles, then TOGSim's grid; the TOGSim config is resolved and checked against
    the target first, so a machine that disagrees is refused before gem5 spends a run on it."""
    machine = as_compiled.togsim_config(req)
    cycles, windows = device.get("gem5").measure(req, ports=True)
    return device.get("togsim").simulate(req, machine, cycles, windows)


def stage_timing(ctx):
    """The gem5 device times one sample tile, the TOGSim device the whole grid on its TOG."""
    import json
    req = device.Launch(
        obj_dir=ctx.workdir,
        manifest=kernel_object.read(ctx.workdir),
        runtime_dir=_run(ctx).runtime_dir,
        raw_paths={},
        log=ctx.log,
        run=lambda stage, cmd, **kw: run_tool(ctx, stage, cmd, **kw),
        step=wallclock.PROF.step,
        error=lambda message, **kw: StageError("timing", message, **kw))
    result = time_launch(req)
    path = os.path.join(ctx.workdir, "togsim.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print(f"TOGSIM CYCLES: {result['total_cycles']}  (tiles {result['tile_cycles']})")
    for unit, u in result.get("units", {}).items():
        print(f"UNIT {unit}: utilization {u['utilization']:.4f}, active_cycles {u['active_cycles']}, "
              f"idle_cycles {u['idle_cycles']} (from TOGSim)")
    return path

