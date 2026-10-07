"""Time a compiled kernel on TOGSim with no PyTorchSim: gem5 measures one tile's compute cycles, the
trace producer replays every tile, TOGSim schedules them."""
import json
import os
import re
import subprocess

from .. import machine, toolchain
from ..contract import kernel_object
from . import emit_gem5_binary
from .passes import cycle_table as ct

CYCLE_TSV = "trace_cycles.tsv"
SHAPE_TXT = "trace_shape.txt"
TENSORS_TXT = "trace_tensors.txt"
GEM5_DIR = "m5out"
LOG = "togsim.log"
RESULT = "togsim.json"

_GEM5_DIED = re.compile(r"^.*\b(?:panic|fatal):.*$|^.*Assertion.*$", re.M)
_STATS = {
    "total_cycles": r"Total execution cycles: (\d+)",
    "vector_active": r"Vector unit utilization\(%\): [\d.]+, active cycle: (\d+)",
    "dma_active": r"DMA active_cycles: (\d+)",
}


class SimulateError(RuntimeError):
    pass


def tile_cycles(spec, kernel_path, workdir):
    """One sample tile under gem5: numCycles per compute marker, the run's total dropped."""
    elf = emit_gem5_binary.build_cycle_binary(spec, kernel_path, workdir)
    out = os.path.join(workdir, GEM5_DIR)
    env = dict(os.environ, TORCHSIM_DIR=toolchain.TOGSIM_DIR)
    yml = machine.target().write_model_yaml(os.path.join(workdir, "machine.yml"))
    p = subprocess.run([toolchain.GEM5, "-r", "--stdout-file=sto.log", "-d", out,
                        toolchain.GEM5_SCRIPT, "-c", elf,
                        "--model", toolchain.VCIX_MODEL,
                        "--machine-config", yml,
                        "--vlane", str(machine.VECTORLANE_SIZE),
                        "--vlen", str(machine.VLEN_BITS)],
                       capture_output=True, text=True, errors="replace", env=env)
    if p.returncode != 0:
        log = os.path.join(out, "sto.log")
        text = open(log, errors="replace").read() if os.path.isfile(log) else p.stderr
        said = _GEM5_DIED.findall(text) or text.strip().splitlines()[-3:]
        raise SimulateError(f"gem5 exit {p.returncode}: {' | '.join(said)[:300]} ({log})")
    stats = open(os.path.join(out, "stats.txt")).read()
    return [int(c) for c in re.findall(r"^system\.cpu\.numCycles\s+(\d+)", stats, re.M)][:-1]


def write_inputs(spec, workdir, cycles):
    """The cycle table, the grid and each argument's bytes, beside trace.so where TOGSim looks for
    them."""
    types = json.load(open(kernel_object.artifact(workdir, "tile_types")))["compute_types"]
    if len(cycles) != len(types):
        raise SimulateError(f"gem5 gave {len(cycles)} tile cycle(s) for {len(types)} "
                            f"compute tile(s); the table would key the wrong samples")
    table = [(c, ct.overlapping_cycle(c, t, machine.VECTORLANE_SIZE, 0))
             for c, t in zip(cycles, types)]
    ct.dump_cycle_table_tsv(table, os.path.join(workdir, CYCLE_TSV))
    with open(os.path.join(workdir, SHAPE_TXT), "w") as fh:
        fh.write("".join(f"{int(g)}\n" for g in spec.grid))
    with open(os.path.join(workdir, TENSORS_TXT), "w") as fh:
        fh.write("".join(f"{a.nbytes}\n" for a in spec.args))
    return table


def run_togsim(workdir):
    """TOGSim on trace.so; its stdout kept as togsim.log, its totals returned."""
    cmd = [toolchain.TOGSIM, "--config", toolchain.TOGSIM_CONFIG,
           "--trace_so", os.path.join(workdir, "trace.so"),
           "--cycle_table", os.path.join(workdir, CYCLE_TSV)]
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    text = p.stdout + p.stderr
    with open(os.path.join(workdir, LOG), "w") as fh:
        fh.write(text)
    if p.returncode != 0 or "Total execution cycles" not in text:
        tail = " | ".join(text.strip().splitlines()[-3:])
        raise SimulateError(f"TOGSim exit {p.returncode}: {tail[:300]}")
    result = {k: int(m.group(1)) for k, rx in _STATS.items()
              if (m := re.search(rx, text))}
    result["config"] = toolchain.TOGSIM_CONFIG
    return result


def simulate(spec, kernel_path, workdir):
    """gem5 tile cycles -> cycle table + grid -> TOGSim."""
    cycles = tile_cycles(spec, kernel_path, workdir)
    table = write_inputs(spec, workdir, cycles)
    result = run_togsim(workdir)
    result["tile_cycles"] = [c for c, _ in table]
    with open(os.path.join(workdir, RESULT), "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    return result
