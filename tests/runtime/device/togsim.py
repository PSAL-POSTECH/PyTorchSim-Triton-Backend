"""TOGSim replays a kernel object's TOG over its whole grid, fed the tile cycles gem5
measured and, given gem5's unit counts, each tile's admitted work; TOGSim computes utilization."""
import json
import math
import os
import re
import subprocess

from pytorchsim_triton_compiler.contract import kernel_object as ko
from . import config
from .. import tog_replay
from .base import Device

CYCLE_TSV = "tog_cycles.tsv"
SHAPE_TXT = "tog_shape.txt"
TENSORS_TXT = "tog_tensors.txt"
TENSOR_SHAPES_TXT = "tog_tensor_shapes.txt"
UNITS_TSV = "tog_units.tsv"
LOG = "togsim.log"
RUNTIME = "runtime"

_STATS = {
    "total_cycles": r"Total execution cycles: (\d+)",
    "vector_active": r"Vector unit utilization\(%\): [\d.]+, active cycle: (\d+)",
    "dma_active": r"DMA active_cycles: (\d+)",
    "indirect_dumped": r"indirect dmas: (\d+) with their index dump",
    "indirect_undumped": r"indirect dmas: \d+ with their index dump, (\d+) without",
}

_UNIT = re.compile(r"(Core \[(\d+)\]|Total) : (\S+) utilization\(%\): ([\d.]+), "
                   r"active_cycles: (\d+), idle_cycles: (-?\d+)$", re.M)


def _runs_per_tile(workdir, manifest, tiles):
    """How many times each of `tiles` sampled tiles runs over the grid, from a replay of the TOG;
    a tile the TOG runs that gem5 did not sample, or one gem5 sampled the TOG never runs, raises."""
    runs = [0] * tiles
    for e in tog_replay.replay(workdir, ko.launch_shape(manifest)):
        if e["op"] != "compute":
            continue
        if not 0 <= e["tile"] < tiles:
            raise ValueError(f"the TOG runs tile {e['tile']}; gem5 sampled {tiles}")
        runs[e["tile"]] += 1
    idle = [k for k, n in enumerate(runs) if n == 0]
    if idle:
        raise ValueError(f"gem5 sampled tile(s) {idle}; the TOG never runs them")
    return runs


def _unit_rows(windows):
    """--unit_table rows, one per tile and unit: what one run of the tile admitted at the unit."""
    return [(k, unit, u["unit_of_work"], int(u["capacity"]), int(u["admitted"]))
            for k, w in enumerate(windows) for unit, u in w["units"].items()]


def _units(text, rows, runs):
    """togsim.json 'units' from the unit lines of TOGSim's end-of-run block: the Total line, or the
    one core's; admitted is what the TOG's runs of each tile were sent, capacity and unit of work
    as the table said."""
    final = text.rsplit("Simulation finished", 1)[-1]
    units = {}
    for k, unit, work, capacity, admitted in rows:
        u = units.setdefault(unit, {"admitted": 0, "capacity": capacity, "unit_of_work": work, "cores": []})
        u["admitted"] += runs[k] * admitted
    for m in _UNIT.finditer(final):
        who, core, unit, pct, active, idle = m.groups()
        if unit not in units:
            continue
        line = {"utilization": round(float(pct) / 100, 4), "active_cycles": int(active), "idle_cycles": int(idle)}
        if core is not None:
            units[unit]["cores"].append(dict(line, core=int(core)))
        else:
            units[unit].update(line)
    for unit, u in units.items():
        if "utilization" not in u and len(u["cores"]) == 1:
            u.update({k: v for k, v in u["cores"][0].items() if k != "core"})
        if "utilization" not in u:
            raise ValueError(f"TOGSim printed no end-of-run line for unit {unit}")
    return units


def _tensor_shape_line(arg):
    """`rows cols elem_bytes` of one argument as TOGSim's tog_tensor_tile lays it out: the last
    axis is the columns, every leading axis folds into the rows; a scalar is one 1x1 tensor."""
    shape = arg["shape"]
    cols = shape[-1] if shape else 1
    return f"{math.prod(shape[:-1])} {cols} {ko.ITEMSIZE[arg['dtype']]}\n"


class TOGSimDevice(Device):
    """Writes the cycle table, the grid and each argument's bytes and 2-D shape beside tog.so,
    then runs TOGSim; an indirect dma reads the index dumps spike left in runtime/indirect_access."""

    name = "togsim"

    def simulate(self, req, tile_cycles, windows=None):
        """TOGSim's totals for the whole grid; given gem5's per-tile windows, also each unit's
        utilization as TOGSim reports it, and per tile the admitted work it was sent."""
        offsets = json.load(open(req.artifact("tile_types")))["overlap_offset"]
        if len(tile_cycles) != len(offsets):
            raise req.error(f"gem5 gave {len(tile_cycles)} tile cycle(s) for {len(offsets)} "
                            f"compute tile(s); the table would key the wrong samples")
        table = [(c, 0 if off is None else max(int(c) - off, 0))
                 for c, off in zip(tile_cycles, offsets)]
        d = req.obj_dir
        with open(os.path.join(d, CYCLE_TSV), "w") as fh:
            fh.write("".join("%d\t%d\n" % (int(c), int(o)) for c, o in table))
        with open(os.path.join(d, SHAPE_TXT), "w") as fh:
            fh.write("".join(f"{int(g)}\n" for g in ko.launch_shape(req.manifest)))
        with open(os.path.join(d, TENSORS_TXT), "w") as fh:
            fh.write("".join(f"{ko.nbytes(a)}\n" for a in req.manifest["args"]))
        with open(os.path.join(d, TENSOR_SHAPES_TXT), "w") as fh:
            fh.write("".join(_tensor_shape_line(a) for a in req.manifest["args"]))
        _runtime_beside_tog(req)
        cmd = [config.TOGSIM, "--config", config.TOGSIM_CONFIG,
               "--tog_so", req.artifact("tog_so"),
               "--cycle_table", os.path.join(d, CYCLE_TSV)]
        rows, runs = [], None
        if windows is not None:
            try:
                runs = _runs_per_tile(d, req.manifest, len(windows))
                rows = _unit_rows(windows)
            except ValueError as e:
                raise req.error(f"unit table: {e}") from None
        units_tsv = os.path.join(d, UNITS_TSV)
        if os.path.exists(units_tsv):
            os.remove(units_tsv)
        if rows:
            with open(units_tsv, "w") as fh:
                fh.write("".join("\t".join(str(f) for f in r) + "\n" for r in rows))
            cmd += ["--unit_table", units_tsv]
        with req.step("togsim"):
            p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        text = p.stdout + p.stderr
        with open(os.path.join(d, LOG), "w") as fh:
            fh.write(text)
        if p.returncode != 0 or "Total execution cycles" not in text:
            tail = " | ".join(text.strip().splitlines()[-3:])
            raise req.error(f"TOGSim exit {p.returncode}: {tail[:300]}")
        result = {k: int(m.group(1)) for k, rx in _STATS.items() if (m := re.search(rx, text))}
        if result.get("indirect_undumped"):
            raise req.error(f"{result['indirect_undumped']} indirect dma(s) found no index dump "
                            f"under {req.runtime_dir}/indirect_access, so TOGSim timed them "
                            f"as dense tiles: run spike before timing")
        result["config"] = config.TOGSIM_CONFIG
        result["tile_cycles"] = [c for c, _ in table]
        if runs is not None:
            try:
                result["units"] = _units(text, rows, runs)
            except ValueError as e:
                raise req.error(str(e)) from None
            result["segments"] = [
                {"tile": k, "runs": n, "cycles": result["tile_cycles"][k],
                 "admitted": {unit: u["admitted"] for unit, u in w["units"].items()}}
                for k, (n, w) in enumerate(zip(runs, windows))]
        return result


def _runtime_beside_tog(req):
    """Make `runtime/` beside tog.so the launch's runtime dir, which is where TOGSim looks."""
    here = os.path.join(req.obj_dir, RUNTIME)
    if os.path.realpath(here) == os.path.realpath(req.runtime_dir):
        return
    if os.path.lexists(here):
        raise req.error(f"{here} is not the launch's runtime dir {req.runtime_dir}, and TOGSim "
                        f"reads the index dumps from the one beside tog.so")
    os.symlink(os.path.abspath(req.runtime_dir), here)
