"""What gem5 measured for a cycle binary, keyed by the (tile id, call) each dump names: the one reader
of m5out. A dump is the tile id and call the opening marker passed (system.sample.*), its cycles and
its units; the end-of-run dump names -1 and is dropped. The model's units, and what each counts, are
its counters with a capacity (vcix_model.counters), read by the names gem5 prints them under."""
import json
import os
import re

from . import vcix_model
from .kernel_object import CYCLE_CALLS

STATS = "stats.txt"
CONFIG = "config.json"

TILE_ID = "system.sample.tileId"
CALL = "system.sample.call"
CYCLES = "system.cpu.numCycles"
COMMITTED = "system.cpu.commitStats0.numOps"
ISSUE = "Issue"

_CPU_UNITS = "system.cpu.units."
_CPU_UNIT_STATS = ("admitted", "capacity", "cycles", "utilized_cycles")
# What a CPU unit admits: its busy cycles (gem5 src/cpu/minor/execute.cc).
_CPU_UNIT_OF_WORK = "cycles"
_UNMARKED = -1

SAMPLE_OPEN = ".insn r CUSTOM_3, 0, 0x40, x0, x0, x0"
"""gem5's RISC-V m5op resetstats (opcode CUSTOM_3, funct7 M5OP_RESET_STATS): opens a sample, its
tile id in a0 and its call in a1."""
SAMPLE_CLOSE = ".insn r CUSTOM_3, 0, 0x41, x0, x0, x0"
"""gem5's RISC-V m5op dumpstats (funct7 M5OP_DUMP_STATS): closes the sample, one dump of stats.txt."""


class SampleError(ValueError):
    """m5out does not hold exactly one dump for every (tile, call) the cycle binary runs."""


def _number(text):
    """A stats.txt value: an int when it is one, else a float (nan included)."""
    try:
        return int(text)
    except ValueError:
        return float(text)


def _stat_name(name):
    """A model's group or counter name as gem5 prints it (statName, src/cpu/minor/coprocessor_model.cc)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def issue_width(config):
    """The CPU's issue width in micro-ops, the unit numOps counts: executeIssueLimit of system.cpu
    in gem5's config.json."""
    cpu = config["system"]["cpu"]
    if not isinstance(cpu, dict) or "executeIssueLimit" not in cpu:
        raise SampleError("config.json: system.cpu is not one CPU with an executeIssueLimit")
    return int(cpu["executeIssueLimit"])


def model_counters(config):
    """The counters (vcix_model.counters) of the model the one FU of gem5's config.json that names one
    ran, created with the settings that FU handed it; () when no FU names a model."""
    fus = config["system"]["cpu"].get("executeFuncUnits", {}).get("funcUnits", [])
    named = [fu for fu in fus if fu.get("coprocessorModel")]
    if len(named) > 1:
        raise SampleError(f"config.json: {len(named)} FUs name a coprocessor model; this reader reads one FU's")
    if not named:
        return ()
    fu = named[0]
    return vcix_model.counters(fu["coprocessorModel"],
                               dict(zip(fu["coprocessorConfigKeys"], fu["coprocessorConfigValues"])))


def dumps(stats, width, counters=()):
    """Each dump of stats.txt, in order, as {tile, call, cycles, units, committed}: the one FU's vcix
    units (`counters` with a capacity, '<group>.<name>'), the CPU's FU-tagged units, and Issue
    (committed micro-ops, capacity `width`); a unit maps to {admitted, capacity, unit_of_work[, cycles]}."""
    by_name = {f"{_stat_name(g)}.{_stat_name(n)}": (g, n, c) for g, n, c in counters}
    out, owner, source = [], None, {ISSUE: "issue"}
    for block in stats.split("Begin Simulation Statistics")[1:]:
        w = {"tile": None, "call": None, "cycles": None, "units": {}, "committed": {}}
        seen = set()
        for line in block.splitlines():
            field = line.split()
            if len(field) < 2 or line[0].isspace():
                continue
            name, value = field[0], field[1]
            if name in (TILE_ID, CALL):
                w["tile" if name == TILE_ID else "call"] = int(float(value))
                continue
            if name == CYCLES:
                w["cycles"] = _number(value)
            if name == COMMITTED:
                w["units"][ISSUE] = {"admitted": _number(value), "capacity": width,
                                     "unit_of_work": "micro-ops"}
                continue
            if name.startswith(_CPU_UNITS):
                part = name[len(_CPU_UNITS):].split(".")
                if len(part) != 2 or part[1] not in _CPU_UNIT_STATS:
                    raise SampleError(f"{name}: not a unit statistic this reader reads")
                unit, group, field = part[0], "units", part[1]
                u = w["units"].setdefault(unit, {"unit_of_work": _CPU_UNIT_OF_WORK})
            elif ".vcix." in name:
                fu, path = name.split(".vcix.", 1)
                if owner is not None and fu != owner:
                    raise SampleError(f"both {owner} and {fu} print vcix statistics; this reader reads one FU's")
                if path not in by_name:
                    raise SampleError(f"{name}: no counter of the model the FU ran")
                owner, (g, n, capacity) = fu, by_name[path]
                seen.add(path)
                if g == vcix_model.COMMITTED:
                    w["committed"][n] = _number(value)
                if g == vcix_model.COMMITTED or capacity == 0:
                    continue
                unit, group, field = f"{g}.{n}", "vcix", "admitted"
                u = w["units"].setdefault(unit, {"capacity": capacity, "unit_of_work": n})
            else:
                continue
            if source.setdefault(unit, group) != group:
                raise SampleError(f"unit {unit} is named by both the {source[unit]} and the {group} statistics")
            u[field] = _number(value)
        missing = sorted(set(by_name) - seen)
        if missing:
            raise SampleError(f"a dump has no vcix.{missing[0]}, a counter of the model the FU ran")
        if w["tile"] is None or w["call"] is None:
            raise SampleError(f"a dump has no {TILE_ID} / {CALL}: this gem5 does not name its samples")
        if ISSUE not in w["units"]:
            raise SampleError(f"a dump has no {COMMITTED}")
        w["units"][ISSUE]["cycles"] = w["cycles"]
        out.append(w)
    return out


def samples(stats, width, counters=()):
    """{(tile, call): dump} of every marked dump; the last dump must be the unmarked end-of-run one,
    and no other may be unmarked or name a (tile, call) twice."""
    every = dumps(stats, width, counters)
    if not every or (every[-1]["tile"], every[-1]["call"]) != (_UNMARKED, _UNMARKED):
        raise SampleError("the last dump is not gem5's unmarked end-of-run dump")
    out = {}
    for w in every[:-1]:
        key = (w["tile"], w["call"])
        if _UNMARKED in key:
            raise SampleError(f"a dump before the end of the run names {key}: no marker opened it")
        if key in out:
            raise SampleError(f"two dumps name tile {key[0]} call {key[1]}")
        out[key] = w
    return out


def rows(by_key, n_tiles):
    """Every dump in row order, call c of tile t at row t + n_tiles * c: the rows of TOGSim's cycle
    and unit tables. Refused unless the dumps are exactly tiles 0..n_tiles-1 times calls
    0..CYCLE_CALLS-1."""
    want = {(t, c) for t in range(n_tiles) for c in range(CYCLE_CALLS)}
    missing, extra = sorted(want - set(by_key)), sorted(set(by_key) - want)
    if missing or extra:
        raise SampleError(f"dumps for {n_tiles} tile(s) x {CYCLE_CALLS} call(s): missing (tile, call) "
                          f"{missing}, unexpected {extra}")
    return [by_key[(t, c)] for c in range(CYCLE_CALLS) for t in range(n_tiles)]


def read(m5out, n_tiles):
    """The dumps of a cycle binary's gem5 run in `m5out`, one per row: every call of every tile."""
    config = json.load(open(os.path.join(m5out, CONFIG)))
    with open(os.path.join(m5out, STATS)) as fh:
        return rows(samples(fh.read(), issue_width(config), model_counters(config)), n_tiles)
