"""What gem5 measured for a cycle binary, keyed by the (tile id, call) each dump names: the one reader
of m5out. A dump is the tile id and call the opening marker passed (system.sample.*), its cycles and
its units; the end-of-run dump names -1 and is dropped."""
import json
import os

from .kernel_object import CYCLE_CALLS

STATS = "stats.txt"
CONFIG = "config.json"

TILE_ID = "system.sample.tileId"
CALL = "system.sample.call"
CYCLES = "system.cpu.numCycles"
COMMITTED = "system.cpu.commitStats0.numOps"
ISSUE = "Issue"

_CPU_UNITS = "system.cpu.units."
_UNIT_STATS = ("admitted", "capacity", "cycles", "occupancy", "utilized_cycles")
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


def _unit_of_work(line):
    """What a unit admits, as gem5 describes its admitted statistic: '# <unit_of_work> admitted at port <port> (Count)'."""
    desc = line.split("#", 1)[1].strip() if "#" in line else ""
    head, sep, _ = desc.partition(" admitted at port ")
    if not sep or not head or any(c.isspace() for c in head):
        raise SampleError(f"{line.split()[0]}: no unit of work in its description '{desc}'")
    return head


def issue_width(config):
    """The CPU's issue width in micro-ops, the unit numOps counts: executeIssueLimit of system.cpu
    in gem5's config.json."""
    cpu = config["system"]["cpu"]
    if not isinstance(cpu, dict) or "executeIssueLimit" not in cpu:
        raise SampleError("config.json: system.cpu is not one CPU with an executeIssueLimit")
    return int(cpu["executeIssueLimit"])


def dumps(stats, width):
    """Each dump of stats.txt, in order, as {tile, call, cycles, units, committed}: the one FU's vcix
    units, the CPU's FU-tagged units, and Issue (committed micro-ops, capacity `width`); a unit maps
    to {admitted, capacity, cycles, [occupancy,] [utilized_cycles,] unit_of_work}."""
    out, owner, source = [], None, {ISSUE: "issue"}
    for block in stats.split("Begin Simulation Statistics")[1:]:
        w = {"tile": None, "call": None, "cycles": None, "units": {}, "committed": {}}
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
                group, part = "units", name[len(_CPU_UNITS):].split(".")
                if len(part) != 2 or part[1] not in _UNIT_STATS:
                    raise SampleError(f"{name}: not a unit statistic this reader reads")
            elif ".vcix." in name:
                fu, path = name.split(".vcix.", 1)
                if owner is not None and fu != owner:
                    raise SampleError(f"both {owner} and {fu} print vcix statistics; this reader reads one FU's")
                owner, group, part = fu, "vcix", path.split(".")
            else:
                continue
            if group == "vcix" and len(part) == 2 and part[0] == "committed":
                w["committed"][part[1]] = _number(value)
            elif len(part) == 2 and part[1] in _UNIT_STATS:
                if source.setdefault(part[0], group) != group:
                    raise SampleError(f"unit {part[0]} is named by both the {source[part[0]]} and the {group} statistics")
                unit = w["units"].setdefault(part[0], {})
                unit[part[1]] = _number(value)
                if part[1] == "admitted":
                    unit["unit_of_work"] = _unit_of_work(line)
            else:
                raise SampleError(f"{name}: not a vcix statistic this reader reads")
        if w["tile"] is None or w["call"] is None:
            raise SampleError(f"a dump has no {TILE_ID} / {CALL}: this gem5 does not name its samples")
        if ISSUE not in w["units"]:
            raise SampleError(f"a dump has no {COMMITTED}")
        w["units"][ISSUE]["cycles"] = w["cycles"]
        out.append(w)
    return out


def samples(stats, width):
    """{(tile, call): dump} of every marked dump; the last dump must be the unmarked end-of-run one,
    and no other may be unmarked or name a (tile, call) twice."""
    every = dumps(stats, width)
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
    width = issue_width(json.load(open(os.path.join(m5out, CONFIG))))
    with open(os.path.join(m5out, STATS)) as fh:
        return rows(samples(fh.read(), width), n_tiles)
