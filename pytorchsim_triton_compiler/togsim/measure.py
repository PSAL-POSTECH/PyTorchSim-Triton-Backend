"""The cycles gem5 times a build at: its tile types and one-tile binary made (stage tog's, without the
TOG), the cycle binary run on the model's gem5 machine, every call of every tile summed. The spill retry
compares two widths by it (driver.compiler._judge_spills)."""
import json
import subprocess

GEM5_DIR = "m5out-retry"


def tile_cycles(ctx):
    """The summed tile cycles of the build in ctx.workdir; raises StageError when gem5 fails."""
    from .. import toolchain
    from ..contract import gem5_samples, kernel_object
    from ..driver import machine
    from ..driver.stage import StageError
    from . import cycle_binary, emit_tog
    emit_tog.emit_tile_types(ctx.spec, ctx.workdir)
    cycle_binary.build_cycle_binary(ctx)
    t = machine.subtarget()
    manifest = kernel_object.read(ctx.workdir)
    run = kernel_object.run_options(t, ctx.workdir, toolchain.COPROCESSOR_BUILD, manifest)
    out = ctx.path(GEM5_DIR)
    p = subprocess.run([toolchain.GEM5, "-r", "--stdout-file=sto.log", "-d", out, run["gem5_script"],
                        "-c", kernel_object.artifact(ctx.workdir, "cycle_elf", manifest=manifest),
                        "--model", run["model"], "--machine-config", run["machine_config"],
                        "--vlane", str(t.vector_lanes), "--vlen", str(t.vlen_bits)],
                       capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        raise StageError("binary", f"gem5 exit {p.returncode} timing the retry's build ({out}/sto.log)")
    tiles = json.load(open(kernel_object.artifact(ctx.workdir, "tile_types", manifest=manifest)))
    rows = gem5_samples.read(out, len(tiles["compute_types"]))
    total = sum(w["cycles"] for w in rows)
    ctx.log(f"[retry] gem5 tile cycles {[w['cycles'] for w in rows]}, {total} in all")
    return total
