"""gem5 times one sample tile of a kernel object: the per-tile compute cycles a TOGSim run reads."""
import os
import re
import subprocess

from pytorchsim_triton_compiler.contract import target as _target
from . import config
from .base import Device

GEM5_DIR = "m5out"

_DIED = re.compile(r"^.*\b(?:panic|fatal):.*$|^.*Assertion.*$", re.M)


class Gem5Device(Device):
    """Runs the object's cycle_elf under gem5 and reads numCycles per compute marker."""

    name = "gem5"

    def measure(self, req):
        elf = req.artifact("cycle_elf")
        out = os.path.join(req.obj_dir, GEM5_DIR)
        t = _target.parse(req.manifest["target"], f"{req.obj_dir}: manifest target")
        yml = config.MACHINE_CONFIG or t.write_model_yaml(os.path.join(req.obj_dir, "machine.yml"))
        env = dict(os.environ, TORCHSIM_DIR=config.TOGSIM_DIR)
        with req.step("gem5"):
            p = subprocess.run([config.GEM5, "-r", "--stdout-file=sto.log", "-d", out,
                                config.GEM5_SCRIPT, "-c", elf,
                                "--model", config.VCIX_MODEL,
                                "--machine-config", yml,
                                "--vlane", str(t.vector_lanes),
                                "--vlen", str(t.vlen_bits)],
                               capture_output=True, text=True, errors="replace", env=env)
        if p.returncode != 0:
            log = os.path.join(out, "sto.log")
            text = open(log, errors="replace").read() if os.path.isfile(log) else p.stderr
            said = _DIED.findall(text) or text.strip().splitlines()[-3:]
            raise req.error(f"gem5 exit {p.returncode}: {' | '.join(said)[:300]} ({log})")
        stats = open(os.path.join(out, "stats.txt")).read()
        cycles = [int(c) for c in re.findall(r"^system\.cpu\.numCycles\s+(\d+)", stats, re.M)][:-1]
        req.log(f"[gem5] tile cycles {cycles}")
        return cycles
