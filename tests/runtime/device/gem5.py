"""gem5 times one sample tile of a kernel object: the per-tile compute cycles a TOGSim run reads."""
import json
import os
import re
import subprocess

from pytorchsim_triton_compiler.contract import gem5_samples

from . import as_compiled, config
from .base import Device

GEM5_DIR = "m5out"

_DIED = re.compile(r"^.*\b(?:panic|fatal):.*$|^.*Assertion.*$", re.M)


class Gem5Device(Device):
    """Runs the object's cycle_elf under gem5 and reads the dump of each call of each tile by the
    tile id and call its marker named."""

    name = "gem5"

    def measure(self, req, ports=False):
        """The cycles of every call of every tile the TOG holds, call c of tile t at row t + n * c;
        with ports, also those rows' dumps as gem5_samples reads them."""
        elf = req.artifact("cycle_elf")
        out = os.path.join(req.obj_dir, GEM5_DIR)
        t, run = as_compiled.target(req), as_compiled.run_options(req)
        with req.step("gem5"):
            p = subprocess.run([config.GEM5, "-r", "--stdout-file=sto.log", "-d", out,
                                run["gem5_script"], "-c", elf,
                                "--model", run["model"],
                                "--machine-config", run["machine_config"],
                                "--vlane", str(t.vector_lanes),
                                "--vlen", str(t.vlen_bits)],
                               capture_output=True, text=True, errors="replace")
        if p.returncode != 0:
            log = os.path.join(out, "sto.log")
            text = open(log, errors="replace").read() if os.path.isfile(log) else p.stderr
            said = _DIED.findall(text) or text.strip().splitlines()[-3:]
            raise req.error(f"gem5 exit {p.returncode}: {' | '.join(said)[:300]} ({log})")
        n_tiles = len(json.load(open(req.artifact("tile_types")))["compute_types"])
        try:
            rows = gem5_samples.read(out, n_tiles)
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise req.error(f"gem5 {out}: {type(e).__name__}: {e}")
        cycles = [w["cycles"] for w in rows]
        req.log(f"[gem5] row cycles {cycles}")
        return (cycles, rows) if ports else cycles
