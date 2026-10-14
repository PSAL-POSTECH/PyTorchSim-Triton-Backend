"""The machine this backend compiles for, as data rather than an import: what it is, and every
spelling of it a tool takes (llc, gcc, Spike, the accelerator model, TOGSim)."""
import json
import os

#: Where the shipped targets live, so a bare name resolves: targets/<name>.json at the repo root.
TARGETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "targets")

FIELDS = {
    "vector_lanes": int,
    "vlen_bits": int,
    "spad_size": int,
    "spad_vaddr": int,
    "spad_paddr": int,
    "dram_base": int,
    "dram_size": int,
    "systolic_queue_entries": int,
}


# Each extension of `isa` as (llc -mattr, gcc -march, Spike --isa) spell it; None where that tool
# takes no name for it. Spike runs vector f16 under zfh with v, and the wrapper gcc builds is
# scalar C, so it needs neither zvfbfmin nor the vcix adapter.
EXTENSIONS = {
    "m": ("+m", "m", "m"),
    "a": ("+a", "a", "a"),
    "f": ("+f", "f", "f"),
    "d": ("+d", "d", "d"),
    "c": ("+c", "c", "c"),
    "v": ("+v", "v", "v"),
    "zvfh": ("+zvfh", "zfh", "zfh"),
    "zvfbfmin": ("+zvfbfmin", None, "zvfbfmin"),
    "xsfvcp": ("+xsfvcp", None, "xvcixaccel"),
}

XLEN = 64

_CANONICAL = "imafdgqlcbkjtpvh"

# What the matmul lowering drives, and what runs it: the systolic array of libtpu.so under gem5's
# script_systolic.py, or the matrix unit of libtpu_mxu.so (2 x vector_lanes fed two registers an
# operand), which has no gem5 script. A description that does not say has the systolic array.
MATMUL_UNITS = {"systolic": ("libtpu.so", "script_systolic.py"),
                "mxu": ("libtpu_mxu.so", None)}
DEFAULT_MATMUL_UNIT = "systolic"

# TOGSim's systolic_array_unit names, as this file's matmul_unit names them, and the one TOGSim
# runs when a yaml names none (SimulationConfig.h).
TOGSIM_UNITS = {"Systolic": "systolic", "Mxu": "mxu"}
TOGSIM_DEFAULT_UNIT = "Systolic"

# The queue depth a TOGSim yaml states, under the key of its unit.
TOGSIM_QUEUE_KEY = {"systolic": "tpu_systolic_queue_entries", "mxu": "tpu_mxu_queue_entries"}


class TargetError(RuntimeError):
    pass


def _number(value, field, where):
    if isinstance(value, bool):
        raise TargetError(f"{where}: `{field}` is a boolean, not a number")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            pass
    raise TargetError(f"{where}: `{field}` is not a number: {value!r}")


class Target:
    """One machine, fully stated."""

    def __init__(self, fields, name, origin):
        self.__dict__.update(fields)
        self.name = name
        self.origin = origin

    def check(self):
        """Problems with this description, as a list."""
        bad = []
        n = self.vector_lanes
        if n <= 0 or n & (n - 1):
            bad.append(f"vector_lanes={n} is not a positive power of two. What enforces "
                       f"that is p05_vectorize._get_lane_split, which refuses a tile dim "
                       f"neither <= lanes nor a multiple of them -- and Triton tile dims "
                       f"are powers of two, so a non-power-of-two lane count refuses "
                       f"nearly every kernel")
        if self.vlen_bits <= 0 or self.vlen_bits % 8:
            bad.append(f"vlen_bits={self.vlen_bits} is not a positive multiple of 8")
        if self.spad_size <= 0:
            bad.append(f"spad_size={self.spad_size} is not positive")
        if self.systolic_queue_entries <= 0:
            bad.append(f"systolic_queue_entries={self.systolic_queue_entries} is not positive")
        if self.matmul_unit not in MATMUL_UNITS:
            bad.append(f"matmul_unit={self.matmul_unit!r} is not one of {', '.join(MATMUL_UNITS)}")
        if not isinstance(self.isa, list) or not all(isinstance(e, str) for e in self.isa):
            bad.append(f"isa={self.isa!r} is not a list of extension names")
        else:
            unknown = [e for e in self.isa if e not in EXTENSIONS]
            if unknown:
                bad.append(f"isa names {', '.join(unknown)}, which no tool spelling is known for; "
                           f"known: {', '.join(EXTENSIONS)}")
        if self.togsim_config is not None and (not isinstance(self.togsim_config, str)
                                               or os.path.basename(self.togsim_config)
                                               != self.togsim_config):
            bad.append(f"togsim_config={self.togsim_config!r} is not a file name under "
                       f"$TOGSIM_DIR/configs")
        return bad

    def _spelled(self, tool):
        """The extension names `tool` (0 llc, 1 gcc, 2 Spike) takes for `isa`, in its order, once."""
        got = []
        for e in self.isa:
            name = EXTENSIONS[e][tool]
            if name is not None and name not in got:
                got.append(name)
        return got

    def _march(self, tool):
        """`rv64` and the single letters in canonical order (m, a, f, d together as g), then the
        `_`-joined long names."""
        names = self._spelled(tool)
        letters = sorted((n for n in names if len(n) == 1), key=_CANONICAL.index)
        if {"m", "a", "f", "d"} <= set(letters):
            letters = ["g"] + [n for n in letters if n not in "mafd"]
        return "_".join([f"rv{XLEN}" + "".join(letters)] + [n for n in names if len(n) > 1])

    def llc_march(self):
        """llc's -march."""
        return f"riscv{XLEN}"

    def llc_mattr(self):
        """llc's -mattr: every extension, then the vector length."""
        return ",".join(self._spelled(0) + [f"+zvl{self.vlen_bits}b"])

    def gcc_march(self):
        """gcc's -march for the scalar C wrapper the kernel links with."""
        return self._march(1)

    def gcc_mabi(self):
        """gcc's -mabi: doubles in registers when the machine has d, floats with f alone."""
        return f"lp{XLEN}" + ("d" if "d" in self.isa else "f" if "f" in self.isa else "")

    def spike_isa(self):
        """Spike's --isa: the extensions the ELF was compiled with, as Spike names them."""
        return self._march(2)

    def model_library(self):
        """The accelerator model Spike and gem5 load for this matmul unit, a file under the vcix build."""
        return MATMUL_UNITS[self.matmul_unit][0]

    def gem5_script(self):
        """The vcix-accelerator gem5 script that times this matmul unit, or None when there is none."""
        return MATMUL_UNITS[self.matmul_unit][1]

    def memory(self):
        """Spike's -m regions: DRAM, then the banked scratchpad's physical window."""
        return [[self.dram_base, self.dram_size], [self.spad_paddr, self.banked_bytes()]]

    def togsim_disagreements(self, yml):
        """What a decoded TOGSim yaml says differently from this machine, among the keys TOGSim and
        its frontend read; a systolic_array_unit TOGSIM_UNITS does not know raises TargetError."""
        kb = self.spad_size // 1024
        said = [("vpu_num_lanes", self.vector_lanes), ("vpu_spad_size_kb_per_lane", kb),
                ("core_spad_size_kb", kb * self.vector_lanes)]
        bad = [f"{key}: the yaml says {yml[key]}, the target {mine}"
               for key, mine in said if key in yml and int(yml[key]) != mine]
        missing = [key for key, _ in said[:2] if key not in yml]
        bad += [f"{key}: the yaml does not say" for key in missing]
        unit = togsim_unit(yml, "the yaml")
        if unit != self.matmul_unit:
            said_unit = yml.get("systolic_array_unit", f"nothing ({TOGSIM_DEFAULT_UNIT})")
            bad.append(f"systolic_array_unit: the yaml says {said_unit}, "
                       f"the target's matmul_unit is {self.matmul_unit}")
        return bad

    def banked_bytes(self):
        """The whole scratchpad window, not one lane's share -- sim.h:74."""
        return self.spad_size * self.vector_lanes

    def memory_map_problems(self):
        """Problems with the scratchpad/DRAM layout, as a list."""
        problems = []
        lo, hi = self.spad_vaddr, self.spad_vaddr + self.banked_bytes()
        dram_lo, dram_hi = self.dram_base, self.dram_base + self.dram_size
        if not (dram_lo <= lo and hi < dram_hi):
            problems.append(
                f"the banked scratchpad window 0x{lo:x}-0x{hi:x} is not inside "
                f"DRAM 0x{dram_lo:x}-0x{dram_hi:x}. Everything the simulator "
                f"does NOT redirect -- the .spadtop anchor and every malloc'd "
                f"buffer above it -- needs real memory there.")
        p_lo, p_hi = self.spad_paddr, self.spad_paddr + self.banked_bytes()
        if p_lo < dram_hi and dram_lo < p_hi:
            problems.append(
                f"the scratchpad's physical region 0x{p_lo:x}-0x{p_hi:x} "
                f"overlaps DRAM 0x{dram_lo:x}-0x{dram_hi:x}; the simulator is "
                f"given them as two separate -m regions and they must not "
                f"intersect.")
        return problems

    def as_dict(self):
        """The fields every description states, then what this one states beyond them: a target
        of the systolic array reads as it did before there was another unit."""
        said = {f: getattr(self, f) for f in FIELDS}
        said["isa"] = list(self.isa)
        if self.matmul_unit != DEFAULT_MATMUL_UNIT:
            said["matmul_unit"] = self.matmul_unit
        if self.togsim_config is not None:
            said["togsim_config"] = self.togsim_config
        return said

    def write_model_yaml(self, path):
        """This same machine in the spelling the accelerator model reads, for Spike's and gem5's
        --machine-config."""
        kb, rem = divmod(self.spad_size, 1024)
        if rem:
            raise ValueError(f"scratchpad {self.spad_size} B is not whole kilobytes")
        with open(path, "w") as f:
            f.write(f"vpu_num_lanes: {self.vector_lanes}\n"
                    f"vpu_spad_size_kb_per_lane: {kb}\n"
                    f"vpu_spad_base_vaddr: 0x{self.spad_vaddr:x}\n"
                    f"vpu_vector_length_bits: {self.vlen_bits}\n"
                    f"tpu_systolic_queue_entries: {self.systolic_queue_entries}\n")
        return path


def parse(raw, origin):
    """A Target out of an already-decoded mapping, or raise TargetError."""
    if not isinstance(raw, dict):
        raise TargetError(f"{origin}: a target description must be a mapping")
    missing = [f for f in FIELDS if f not in raw]
    if missing:
        raise TargetError(
            f"{origin}: target description is missing {', '.join(missing)}. "
            f"Every field is required: a partially stated machine is the seam "
            f"this file exists to close.")
    if "isa" not in raw:
        raise TargetError(f"{origin}: target description is missing isa, the extension list the "
                          f"ELF is compiled with and run under")
    fields = {f: _number(raw[f], f, origin) for f in FIELDS}
    fields["isa"] = raw["isa"]
    fields["matmul_unit"] = raw.get("matmul_unit", DEFAULT_MATMUL_UNIT)
    fields["togsim_config"] = raw.get("togsim_config")
    t = Target(fields, raw.get("name", os.path.basename(origin)), origin)
    bad = t.check()
    if bad:
        raise TargetError(f"{origin}: " + "; ".join(bad))
    return t


def resolve(said):
    """The file a target is named by: a path as given, a bare name as TARGETS_DIR/<name>.json."""
    if os.sep in said or said.endswith(".json"):
        return said
    path = os.path.join(TARGETS_DIR, said + ".json")
    if not os.path.isfile(path):
        have = sorted(f[:-len(".json")] for f in os.listdir(TARGETS_DIR) if f.endswith(".json")) \
            if os.path.isdir(TARGETS_DIR) else []
        raise TargetError(f"no target named {said!r} under {TARGETS_DIR}; have: "
                          f"{', '.join(have) or 'none'}. A path to a description works too.")
    return path


def load(path):
    """The machine at `path`, or the shipped one a bare name names."""
    path = resolve(path)
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except OSError as e:
        raise TargetError(f"{path}: {e.strerror}")
    except json.JSONDecodeError as e:
        raise TargetError(f"{path}: not valid JSON ({e})")
    return parse(raw, path)



def togsim_unit(yml, where):
    """The matmul_unit a decoded TOGSim yaml runs: its systolic_array_unit, TOGSIM_DEFAULT_UNIT when
    it names none as TOGSim reads it; refuses a unit this file has no matmul_unit for."""
    name = yml.get("systolic_array_unit", TOGSIM_DEFAULT_UNIT)
    if name not in TOGSIM_UNITS:
        raise TargetError(f"{where}: systolic_array_unit {name!r} is no matmul_unit this compiler "
                          f"lowers for; known: {', '.join(TOGSIM_UNITS)}")
    return TOGSIM_UNITS[name]


def from_togsim(yml, like):
    """A draft target dict from a decoded TOGSim yaml; what the yaml does not state (isa, the
    address map) is taken from the Target `like`. The yaml's two statements of the scratchpad must agree."""
    unit = togsim_unit(yml, "the TOGSim config")
    queue = TOGSIM_QUEUE_KEY[unit]
    need = ("vpu_num_lanes", "vpu_vector_length_bits", "vpu_spad_size_kb_per_lane", queue)
    missing = [k for k in need if k not in yml]
    if missing:
        raise TargetError(f"the TOGSim config has no {', '.join(missing)}")
    lanes = int(yml["vpu_num_lanes"])
    per_lane_kb = int(yml["vpu_spad_size_kb_per_lane"])
    whole = yml.get("core_spad_size_kb")
    if whole is not None and int(whole) != per_lane_kb * lanes:
        raise TargetError(
            f"this TOGSim config disagrees with itself about the scratchpad: "
            f"vpu_spad_size_kb_per_lane={per_lane_kb} x vpu_num_lanes={lanes} is "
            f"{per_lane_kb * lanes} kB, but core_spad_size_kb says {whole}.")
    draft = {
        "vector_lanes": lanes,
        "vlen_bits": int(yml["vpu_vector_length_bits"]),
        "spad_size": per_lane_kb * 1024,
        "spad_vaddr": hex(like.spad_vaddr),
        "spad_paddr": hex(like.spad_paddr),
        "dram_base": hex(like.dram_base),
        "dram_size": like.dram_size,
        "systolic_queue_entries": int(yml[queue]),
        "isa": list(like.isa),
    }
    if unit != DEFAULT_MATMUL_UNIT:
        draft["matmul_unit"] = unit
    return draft
