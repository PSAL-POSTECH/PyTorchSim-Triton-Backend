"""TOGSim runs a kernel object's trace producer over its whole grid, fed the tile cycles gem5
measured."""
import json
import os
import re
import subprocess

from pytorchsim_triton_compiler.contract import kernel_object as ko
from . import config
from .base import Device

CYCLE_TSV = "trace_cycles.tsv"
SHAPE_TXT = "trace_shape.txt"
TENSORS_TXT = "trace_tensors.txt"
LOG = "togsim.log"

_STATS = {
    "total_cycles": r"Total execution cycles: (\d+)",
    "vector_active": r"Vector unit utilization\(%\): [\d.]+, active cycle: (\d+)",
    "dma_active": r"DMA active_cycles: (\d+)",
}


class TOGSimDevice(Device):
    """Writes the cycle table, the grid and each argument's bytes beside trace.so, then runs TOGSim."""

    name = "togsim"

    def simulate(self, req, tile_cycles):
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
            fh.write("".join(f"{int(g)}\n" for g in req.manifest["grid"]))
        with open(os.path.join(d, TENSORS_TXT), "w") as fh:
            fh.write("".join(f"{ko.nbytes(a)}\n" for a in req.manifest["args"]))
        cmd = [config.TOGSIM, "--config", config.TOGSIM_CONFIG,
               "--trace_so", req.artifact("trace_so"),
               "--cycle_table", os.path.join(d, CYCLE_TSV)]
        with req.step("togsim"):
            p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        text = p.stdout + p.stderr
        with open(os.path.join(d, LOG), "w") as fh:
            fh.write(text)
        if p.returncode != 0 or "Total execution cycles" not in text:
            tail = " | ".join(text.strip().splitlines()[-3:])
            raise req.error(f"TOGSim exit {p.returncode}: {tail[:300]}")
        result = {k: int(m.group(1)) for k, rx in _STATS.items() if (m := re.search(rx, text))}
        result["config"] = config.TOGSIM_CONFIG
        result["tile_cycles"] = [c for c, _ in table]
        return result
