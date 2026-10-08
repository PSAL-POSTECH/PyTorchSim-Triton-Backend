"""gem5 times one sample tile of a kernel object: the per-tile compute cycles a TOGSim run reads."""
import os
import re
import subprocess

from pytorchsim_triton_compiler.contract import target as _target
from . import config
from .base import Device

GEM5_DIR = "m5out"

_CPU_UNITS = "system.cpu.units."

_UNIT_STATS = ("admitted", "capacity", "cycles", "occupancy", "utilized_cycles")

_DIED = re.compile(r"^.*\b(?:panic|fatal):.*$|^.*Assertion.*$", re.M)


def _number(text):
    """A stats.txt value: an int when it is one, else a float (nan included)."""
    try:
        return int(text)
    except ValueError:
        return float(text)


def _unit_of_work(line):
    """What a unit admits, as gem5 describes its admitted statistic: '# <unit_of_work> admitted at port <port> (Count)'."""
    desc = line.split("#", 1)[1].strip() if "#" in line else ""
    head, sep, _ = desc.partition(" admitted at port ")
    if not sep or not head or any(c.isspace() for c in head):
        raise ValueError(f"{line.split()[0]}: no unit of work in its description '{desc}'")
    return head


def windows(stats):
    """Each dump of stats.txt as {cycles, units, committed}, read off the one FU's vcix group and the
    CPU's units group (system.cpu.units.<unit>, an FU-tagged unit's busy cycles); units maps a unit to
    {admitted, capacity, cycles, [occupancy,] utilized_cycles, unit_of_work}. A unit in both raises."""
    out, owner, source = [], None, {}
    for block in stats.split("Begin Simulation Statistics")[1:]:
        w = {"cycles": None, "units": {}, "committed": {}}
        for line in block.splitlines():
            field = line.split()
            if len(field) < 2 or line[0].isspace():
                continue
            name, value = field[0], field[1]
            if name == "system.cpu.numCycles":
                w["cycles"] = _number(value)
            if name.startswith(_CPU_UNITS):
                group, part = "units", name[len(_CPU_UNITS):].split(".")
                if len(part) != 2 or part[1] not in _UNIT_STATS:
                    raise ValueError(f"{name}: not a unit statistic windows() reads")
            elif ".vcix." in name:
                fu, path = name.split(".vcix.", 1)
                if owner is not None and fu != owner:
                    raise ValueError(f"both {owner} and {fu} print vcix statistics; windows() reads one FU's")
                owner, group, part = fu, "vcix", path.split(".")
            else:
                continue
            if group == "vcix" and len(part) == 2 and part[0] == "committed":
                w["committed"][part[1]] = _number(value)
            elif len(part) == 2 and part[1] in _UNIT_STATS:
                if source.setdefault(part[0], group) != group:
                    raise ValueError(f"unit {part[0]} is both a vcix unit and an FU-tagged unit")
                unit = w["units"].setdefault(part[0], {})
                unit[part[1]] = _number(value)
                if part[1] == "admitted":
                    unit["unit_of_work"] = _unit_of_work(line)
            else:
                raise ValueError(f"{name}: not a vcix statistic windows() reads")
        out.append(w)
    return out


class Gem5Device(Device):
    """Runs the object's cycle_elf under gem5 and reads numCycles per compute marker."""

    name = "gem5"

    def measure(self, req, ports=False):
        """The tile cycles, one per sampled tile; with ports, also each tile's dump as windows()
        reads it, the end-of-run dump left out as the cycles leave it out."""
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
        if not ports:
            return cycles
        try:
            return cycles, windows(stats)[:-1]
        except ValueError as e:
            raise req.error(f"gem5 stats.txt: {e}")
