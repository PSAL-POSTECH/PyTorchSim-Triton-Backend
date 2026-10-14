"""The machine this backend compiles for, as data rather than an import."""
import json
import os

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


# What the matmul lowering drives: the systolic array of libtpu.so, or the matrix unit of
# libtpu_mxu.so, an array of 2 x vector_lanes fed two registers an operand. A description that
# does not say has the systolic array.
MATMUL_UNITS = ("systolic", "mxu")
# How many such arrays the matrix unit has (the TPU v6e's bundles name two, mxu0 and mxu1), each
# with its own weights and queues, named by an instruction in a field of 5 bits. A description
# that does not say has one; the systolic array is one.
MATMUL_UNITS_MOST = 32


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
        if not 1 <= self.matmul_units <= MATMUL_UNITS_MOST:
            bad.append(f"matmul_units={self.matmul_units} is not 1 to {MATMUL_UNITS_MOST}, what an "
                       f"instruction's field of 5 bits names")
        elif self.matmul_units != 1 and self.matmul_unit != "mxu":
            bad.append(f"matmul_units={self.matmul_units}, and only the matrix unit (matmul_unit "
                       f"\"mxu\") has more arrays than one")
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
        if self.matmul_unit != MATMUL_UNITS[0]:
            said["matmul_unit"] = self.matmul_unit
        if self.matmul_units != 1:
            said["matmul_units"] = self.matmul_units
        return said


    def write_model_yaml(self, path):
        """This same machine in the spelling the accelerator model reads, for Spike's and gem5's
        --machine-config. A target of the matrix unit states its arrays (libtpu_mxu.so's
        tpu_mxu_units); one of the systolic array reads as it did."""
        kb, rem = divmod(self.spad_size, 1024)
        if rem:
            raise ValueError(f"scratchpad {self.spad_size} B is not whole kilobytes")
        with open(path, "w") as f:
            f.write(f"vpu_num_lanes: {self.vector_lanes}\n"
                    f"vpu_spad_size_kb_per_lane: {kb}\n"
                    f"vpu_spad_base_vaddr: 0x{self.spad_vaddr:x}\n"
                    f"vpu_vector_length_bits: {self.vlen_bits}\n"
                    f"tpu_systolic_queue_entries: {self.systolic_queue_entries}\n")
            if self.matmul_unit == "mxu":
                f.write(f"tpu_mxu_units: {self.matmul_units}\n")
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
    fields = {f: _number(raw[f], f, origin) for f in FIELDS}
    fields["matmul_unit"] = raw.get("matmul_unit", MATMUL_UNITS[0])
    fields["matmul_units"] = _number(raw.get("matmul_units", 1), "matmul_units", origin)
    t = Target(fields, raw.get("name", os.path.basename(origin)), origin)
    bad = t.check()
    if bad:
        raise TargetError(f"{origin}: " + "; ".join(bad))
    return t


def load(path):
    """The machine at `path`."""
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except OSError as e:
        raise TargetError(f"{path}: {e.strerror}")
    except json.JSONDecodeError as e:
        raise TargetError(f"{path}: not valid JSON ({e})")
    return parse(raw, path)

