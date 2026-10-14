"""gem5 times one sample tile of a kernel object: the per-tile compute cycles a TOGSim run reads."""
import json
import os
import re
import subprocess

from . import as_compiled, config
from .base import Device

GEM5_DIR = "m5out"

_CPU_UNITS = "system.cpu.units."

_UNIT_STATS = ("admitted", "capacity", "cycles", "occupancy", "utilized_cycles")

_COMMITTED = "system.cpu.commitStats0.numOps"
ISSUE = "Issue"

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


def issue_width(config):
    """The CPU's issue width in micro-ops, the unit numOps counts: executeIssueLimit of system.cpu
    in gem5's config.json."""
    cpu = config["system"]["cpu"]
    if not isinstance(cpu, dict) or "executeIssueLimit" not in cpu:
        raise ValueError("config.json: system.cpu is not one CPU with an executeIssueLimit")
    return int(cpu["executeIssueLimit"])


def windows(stats, width):
    """Each dump of stats.txt as {cycles, units, committed}: the one FU's vcix units, the CPU's
    FU-tagged units, and Issue (committed micro-ops, capacity `width`); a unit maps to {admitted,
    capacity, cycles, [occupancy,] [utilized_cycles,] unit_of_work}. A unit named twice raises."""
    out, owner, source = [], None, {ISSUE: "issue"}
    for block in stats.split("Begin Simulation Statistics")[1:]:
        w = {"cycles": None, "units": {}, "committed": {}}
        for line in block.splitlines():
            field = line.split()
            if len(field) < 2 or line[0].isspace():
                continue
            name, value = field[0], field[1]
            if name == "system.cpu.numCycles":
                w["cycles"] = _number(value)
            if name == _COMMITTED:
                w["units"][ISSUE] = {"admitted": _number(value), "capacity": width,
                                     "unit_of_work": "micro-ops"}
                continue
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
                    raise ValueError(f"unit {part[0]} is named by both the {source[part[0]]} and the {group} statistics")
                unit = w["units"].setdefault(part[0], {})
                unit[part[1]] = _number(value)
                if part[1] == "admitted":
                    unit["unit_of_work"] = _unit_of_work(line)
            else:
                raise ValueError(f"{name}: not a vcix statistic windows() reads")
        if ISSUE not in w["units"]:
            raise ValueError(f"a dump has no {_COMMITTED}")
        w["units"][ISSUE]["cycles"] = w["cycles"]
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
        t = as_compiled.target(req)
        script, model = as_compiled.gem5_script(req), as_compiled.model(req)
        yml = as_compiled.machine_config(req)
        with req.step("gem5"):
            p = subprocess.run([config.GEM5, "-r", "--stdout-file=sto.log", "-d", out,
                                script, "-c", elf,
                                "--model", model,
                                "--machine-config", yml,
                                "--vlane", str(t.vector_lanes),
                                "--vlen", str(t.vlen_bits)],
                               capture_output=True, text=True, errors="replace")
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
        cfg_path = os.path.join(out, "config.json")
        try:
            width = issue_width(json.load(open(cfg_path)))
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise req.error(f"gem5 {cfg_path}: {type(e).__name__}: {e}")
        try:
            return cycles, windows(stats, width)[:-1]
        except ValueError as e:
            raise req.error(f"gem5 stats.txt: {e}")
