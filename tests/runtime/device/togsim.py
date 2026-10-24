"""TOGSim simulates a kernel object's TOG over its whole grid, fed the cycles gem5 measured for every
call of every tile and, given gem5's unit counts, each one's admitted work; TOGSim computes
utilization."""
import json
import math
import os
import re
import subprocess

from pytorchsim_triton_compiler.contract import kernel_object as ko
from . import config
from .. import tog_record
from .base import Device

CYCLE_TSV = "tog_cycles.tsv"
SHAPE_TXT = "tog_shape.txt"
TENSORS_TXT = "tog_tensors.txt"
TENSOR_SHAPES_TXT = "tog_tensor_shapes.txt"
UNITS_TSV = "tog_units.tsv"
LOG = "togsim.log"

_STATS = {
    "total_cycles": r"Total execution cycles: (\d+)",
    "vector_active": r"Vector unit utilization\(%\): [\d.]+, active cycle: (\d+)",
    "dma_active": r"DMA active_cycles: (\d+)",
}

_UNIT = re.compile(r"(Core \[(\d+)\]|Total) : (\S+) utilization\(%\): ([\d.]+), "
                   r"active_cycles: (\d+), idle_cycles: (-?\d+)$", re.M)


def _runs_per_row(workdir, manifest, tiles, attributes=None):
    """How many times each row (call c of tile t at t + tiles * c) runs over the grid, from a
    recording of the TOG; a compute of a row gem5 did not sample, or of a call other than the last,
    raises, and so does a tile gem5 sampled whose last call the TOG never runs outside a branch arm."""
    rows, last = tiles * ko.CYCLE_CALLS, tiles * (ko.CYCLE_CALLS - 1)
    runs = [0] * rows
    for e in tog_record.record(workdir, ko.shape_args(manifest), attributes=attributes):
        if e["op"] != "compute":
            continue
        if not last <= e["tile"] < rows:
            raise ValueError(f"the TOG computes on row {e['tile']}; gem5's last call is rows "
                             f"{last}..{rows - 1}")
        runs[e["tile"]] += 1
    armed = tog_record.ir_branch_tiles(workdir)
    idle = [k for k in range(tiles) if runs[last + k] == 0 and k not in armed]
    if idle:
        raise ValueError(f"gem5 sampled tile(s) {idle}; the TOG never runs them")
    return runs


def _unit_rows(windows):
    """--unit_table rows, one per row and unit: what one run of that call of the tile admitted at
    the unit."""
    return [(k, unit, u["unit_of_work"], int(u["capacity"]), int(u["admitted"]))
            for k, w in enumerate(windows) for unit, u in w["units"].items()]


def _units(text, rows, runs):
    """togsim.json 'units' from the unit lines of TOGSim's end-of-run block: the Total line, or the
    one core's; admitted is what the TOG's runs of each row were sent, capacity and unit of work
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


def _write_launch(d, manifest, shape):
    """The files TOGSim reads beside a TOG in `d`: the shape_args, each argument's bytes and 2-D shape."""
    with open(os.path.join(d, SHAPE_TXT), "w") as fh:
        fh.write("".join(f"{int(g)}\n" for g in shape))
    with open(os.path.join(d, TENSORS_TXT), "w") as fh:
        fh.write("".join(f"{ko.nbytes(a)}\n" for a in manifest["args"]))
    with open(os.path.join(d, TENSOR_SHAPES_TXT), "w") as fh:
        fh.write("".join(_tensor_shape_line(a) for a in manifest["args"]))


def _togsim(req, cmd, d):
    """TOGSim's output of `cmd`, kept as `d`/LOG; a run that exits non-zero or ends with no total raises."""
    with req.step("togsim"):
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    text = p.stdout + p.stderr
    with open(os.path.join(d, LOG), "w") as fh:
        fh.write(text)
    if p.returncode != 0 or "Total execution cycles" not in text:
        tail = " | ".join(text.strip().splitlines()[-3:])
        raise req.error(f"TOGSim exit {p.returncode}: {tail[:300]}")
    return text


class TOGSimDevice(Device):
    """Writes the cycle table (a row per call of each tile), the grid and each argument's bytes and
    2-D shape beside tog.so, then runs TOGSim on the runtime attributes spike recorded."""

    name = "togsim"

    def simulate(self, req, machine, tile_cycles, windows=None):
        """TOGSim's totals on the config `machine`, given gem5's `tile_cycles` of every row; given
        its windows, also each unit's utilization and each tile's admitted work."""
        offsets = json.load(open(req.artifact("tile_types")))["overlap_offset"]
        tiles = len(offsets)
        if len(tile_cycles) != tiles * ko.CYCLE_CALLS:
            raise req.error(f"gem5 gave {len(tile_cycles)} row cycle(s) for {tiles} compute "
                            f"tile(s) of {ko.CYCLE_CALLS} call(s); the table would key the wrong "
                            f"samples")
        table = [(c, 0 if offsets[r % tiles] is None else max(int(c) - offsets[r % tiles], 0))
                 for r, c in enumerate(tile_cycles)]
        d = req.obj_dir
        with open(os.path.join(d, CYCLE_TSV), "w") as fh:
            fh.write("".join("%d\t%d\n" % (int(c), int(o)) for c, o in table))
        _write_launch(d, req.manifest, ko.shape_args(req.manifest))
        attributes = tog_record.attributes_of(req.runtime_dir)
        cmd = [config.TOGSIM, "--config", machine,
               "--tog_so", req.artifact("tog_so"),
               "--cycle_table", os.path.join(d, CYCLE_TSV),
               "--attribute_path", attributes]
        rows, runs = [], None
        if windows is not None:
            try:
                runs = _runs_per_row(d, req.manifest, tiles, attributes)
                rows = _unit_rows(windows)
            except ValueError as e:
                raise req.error(f"unit table: {e}") from None
        units_tsv = os.path.join(d, UNITS_TSV)
        if os.path.exists(units_tsv):
            os.remove(units_tsv)
        if rows or not tiles:
            with open(units_tsv, "w") as fh:
                fh.write("".join("\t".join(str(f) for f in r) + "\n" for r in rows))
            cmd += ["--unit_table", units_tsv]
        text = _togsim(req, cmd, d)
        result = {k: int(m.group(1)) for k, rx in _STATS.items() if (m := re.search(rx, text))}
        result["config"] = machine
        result["tile_cycles"] = [c for c, _ in table]
        if runs is not None:
            try:
                result["units"] = _units(text, rows, runs)
            except ValueError as e:
                raise req.error(str(e)) from None
            last = tiles * (ko.CYCLE_CALLS - 1)
            result["segments"] = [
                {"tile": k, "runs": runs[last + k], "cycles": result["tile_cycles"][last + k],
                 "admitted": {unit: u["admitted"] for unit, u in windows[last + k]["units"].items()}}
                for k in range(tiles)]
        return result

    def simulate_copy_node(self, req, machine, node):
        """TOGSim's total cycles of one copy node (kernel_object.COPY_NODES) on the config `machine`: a TOG
        that only loads, so its cycle and unit tables are empty and gem5 has nothing to time."""
        so = os.path.join(req.obj_dir, node["tog_so"])
        d = os.path.dirname(so)
        _write_launch(d, req.manifest, [])
        for name in (CYCLE_TSV, UNITS_TSV):
            open(os.path.join(d, name), "w").close()
        text = _togsim(req, [config.TOGSIM, "--config", machine, "--tog_so", so,
                             "--cycle_table", os.path.join(d, CYCLE_TSV),
                             "--unit_table", os.path.join(d, UNITS_TSV)], d)
        return {"arg": node["arg"], "bytes": node["bytes"],
                "total_cycles": int(re.search(_STATS["total_cycles"], text).group(1))}
