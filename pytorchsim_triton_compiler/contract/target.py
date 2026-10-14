"""The machine this backend compiles for, as data, and how each tool spells it."""
import json
import os

# The shipped targets (package data): a bare name resolves to targets/<name>.json.
TARGETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "targets")

FIELDS = {
    "vector_lanes": int,
    "spad_size": int,
    "spad_vaddr": int,
    "spad_paddr": int,
    "dram_base": int,
    "dram_size": int,
}


# Each `isa` extension as (llc -mattr, gcc -march, Spike --isa) spell it; None: that tool takes none.
EXTENSIONS = {
    "m": ("+m", "m", "m"),
    "a": ("+a", "a", "a"),
    "f": ("+f", "f", "f"),
    "d": ("+d", "d", "d"),
    "c": ("+c", "c", "c"),
    "v": ("+v", "v", "v"),
    "zvfh": ("+zvfh", "zfh", "zfh"),
    "zvfbfmin": ("+zvfbfmin", None, "zvfbfmin"),
}

# vcix_<name>: the accelerator, vcix-accelerator's model <name>; every one is LLVM's xsfvcp and
# Spike's xvcixaccel adapter.
ACCEL_PREFIX = "vcix_"
ACCEL_SPELLING = ("+xsfvcp", None, "xvcixaccel")

XLEN = 64

_CANONICAL = "imafdgqlcbkjtpvh"

# The TOGSim systolic_array_unit each accelerator is, for the one check against a TOGSim yaml, and
# what the pinned TOGSim runs when a yaml names none.
TOGSIM_UNITS = {"vcix_tpu": "Systolic", "vcix_tpuv6": "TPUV6_MXU"}
TOGSIM_DEFAULT_UNIT = "Systolic"


def zvl_bits(ext):
    """N of an extension spelled zvl<N>b, else None."""
    if ext.startswith("zvl") and ext.endswith("b") and ext[3:-1].isdigit():
        return int(ext[3:-1])
    return None


def is_accel(ext):
    """Whether `ext` names an accelerator, vcix_<name>."""
    return ext.startswith(ACCEL_PREFIX) and len(ext) > len(ACCEL_PREFIX)


def spelling(ext):
    """(llc, gcc, Spike) spellings of one isa extension, or None when it is no known one."""
    if ext in EXTENSIONS:
        return EXTENSIONS[ext]
    if zvl_bits(ext) is not None:
        return (f"+{ext}", None, None)
    if is_accel(ext):
        return ACCEL_SPELLING
    return None


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

    @property
    def vlen_bits(self):
        """The vector length, as the isa's zvl<N>b states it."""
        return next(zvl_bits(e) for e in self.isa if zvl_bits(e) is not None)

    @property
    def accel(self):
        """The accelerator extension, vcix_<name>."""
        return next(e for e in self.isa if is_accel(e))

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
        if self.spad_size <= 0:
            bad.append(f"spad_size={self.spad_size} is not positive")
        if not isinstance(self.isa, list) or not all(isinstance(e, str) for e in self.isa):
            bad.append(f"isa={self.isa!r} is not a list of extension names")
        else:
            bad += _isa_problems(self.isa)
        if not isinstance(self.vcix, dict) or not all(
                isinstance(k, str) and isinstance(v, (int, float, str)) and not isinstance(v, bool)
                for k, v in self.vcix.items()):
            bad.append(f"vcix={self.vcix!r} is not a mapping of the model's keys to numbers or "
                       f"strings")
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
            name = spelling(e)[tool]
            if name is not None and name not in got:
                got.append(name)
        return got

    def _march(self, tool):
        """`rv64`, the single letters in canonical order (mafd as g), then `_`-joined long names."""
        names = self._spelled(tool)
        letters = sorted((n for n in names if len(n) == 1), key=_CANONICAL.index)
        if {"m", "a", "f", "d"} <= set(letters):
            letters = ["g"] + [n for n in letters if n not in "mafd"]
        return "_".join([f"rv{XLEN}" + "".join(letters)] + [n for n in names if len(n) > 1])

    def llc_march(self):
        """llc's -march."""
        return f"riscv{XLEN}"

    def llc_mattr(self):
        """llc's -mattr: every extension, in the isa's order."""
        return ",".join(self._spelled(0))

    def gcc_march(self):
        """gcc's -march for the scalar C wrapper the kernel links with."""
        return self._march(1)

    def gcc_mabi(self):
        """gcc's -mabi: doubles in registers when the machine has d, floats with f alone."""
        return f"lp{XLEN}" + ("d" if "d" in self.isa else "f" if "f" in self.isa else "")

    def spike_isa(self):
        """Spike's --isa: the extensions the ELF was compiled with, as Spike names them."""
        return self._march(2)

    def memory(self):
        """Spike's -m regions: DRAM, then the banked scratchpad's physical window."""
        return [[self.dram_base, self.dram_size], [self.spad_paddr, self.banked_bytes()]]

    def togsim_disagreements(self, yml):
        """What a decoded TOGSim yaml states differently from this machine."""
        kb = self.spad_size // 1024
        said = [("vpu_num_lanes", self.vector_lanes), ("vpu_spad_size_kb_per_lane", kb),
                ("core_spad_size_kb", kb * self.vector_lanes)]
        bad = [f"{key}: the yaml says {yml[key]}, the target {mine}"
               for key, mine in said if key in yml and int(yml[key]) != mine]
        missing = [key for key, _ in said[:2] if key not in yml]
        bad += [f"{key}: the yaml does not say" for key in missing]
        unit = yml.get("systolic_array_unit", TOGSIM_DEFAULT_UNIT)
        if unit != TOGSIM_UNITS.get(self.accel):
            said_unit = yml.get("systolic_array_unit", f"nothing ({TOGSIM_DEFAULT_UNIT})")
            bad.append(f"systolic_array_unit: the yaml says {said_unit}, the target's "
                       f"accelerator {self.accel} is {TOGSIM_UNITS.get(self.accel, 'no unit TOGSim knows')}")
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
        """The description as a target file states it."""
        said = {f: getattr(self, f) for f in FIELDS}
        said["isa"] = list(self.isa)
        said["vcix"] = dict(self.vcix)
        if self.togsim_config is not None:
            said["togsim_config"] = self.togsim_config
        return said

    def write_model_yaml(self, path):
        """This same machine in the spelling the accelerator model reads, for Spike's and gem5's
        --machine-config: the core's keys, then the vcix block as it is."""
        kb, rem = divmod(self.spad_size, 1024)
        if rem:
            raise ValueError(f"scratchpad {self.spad_size} B is not whole kilobytes")
        with open(path, "w") as f:
            f.write(f"vpu_num_lanes: {self.vector_lanes}\n"
                    f"vpu_spad_size_kb_per_lane: {kb}\n"
                    f"vpu_spad_base_vaddr: 0x{self.spad_vaddr:x}\n"
                    f"vpu_vector_length_bits: {self.vlen_bits}\n")
            for key, value in self.vcix.items():
                f.write(f"{key}: {json.dumps(value)}\n")
        return path


def _isa_problems(isa):
    """What is wrong with an isa list: an unknown name, or not exactly one zvl<N>b and one vcix_*."""
    bad = []
    unknown = [e for e in isa if spelling(e) is None]
    if unknown:
        bad.append(f"isa names {', '.join(unknown)}, which no tool spelling is known for; "
                   f"known: {', '.join(EXTENSIONS)}, zvl<N>b, {ACCEL_PREFIX}<model>"
                   + ("; xsfvcp is not stated, every vcix_* is it" if "xsfvcp" in unknown else ""))
    zvl = [e for e in isa if zvl_bits(e) is not None]
    if len(zvl) != 1:
        bad.append(f"isa states the vector length {len(zvl)} times ({', '.join(zvl) or 'no zvl<N>b'}); "
                   f"it is stated once")
    elif zvl_bits(zvl[0]) < 32 or zvl_bits(zvl[0]) & (zvl_bits(zvl[0]) - 1):
        bad.append(f"{zvl[0]}: the vector length is not a power of two of at least 32 bits")
    accel = [e for e in isa if is_accel(e)]
    if len(accel) != 1:
        bad.append(f"isa names {len(accel)} accelerators ({', '.join(accel) or f'no {ACCEL_PREFIX}*'}); "
                   f"the VCIX opcodes are one model's, so a machine has exactly one")
    return bad


def parse(raw, origin):
    """A Target out of an already-decoded mapping, or raise TargetError."""
    if not isinstance(raw, dict):
        raise TargetError(f"{origin}: a target description must be a mapping")
    retired = [f for f in ("vlen_bits", "matmul_unit", "systolic_queue_entries") if f in raw]
    if retired:
        raise TargetError(f"{origin}: {', '.join(retired)} are retired: the vector length is the "
                          f"isa's zvl<N>b, the accelerator its vcix_<model>, and the model's "
                          f"settings its own keys in the vcix block")
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
    fields["vcix"] = raw.get("vcix", {})
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
