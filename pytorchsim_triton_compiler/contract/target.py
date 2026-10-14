"""The machine this backend compiles for, as data rather than an import."""
import json
import os
from collections import namedtuple

FIELDS = {
    "vector_lanes": int,
    "vlen_bits": int,
    "spad_size": int,
    "spad_vaddr": int,
    "spad_paddr": int,
    "dram_base": int,
    "dram_size": int,
}


# What the matmul lowering drives, by the name the model reports it under, lowercased: SYSTOLIC of
# libtpu.so, or TPUV6_MXU of libtpuv6.so, an array of 2 x vector_lanes fed two registers an operand.
# A description that does not say has the systolic array. Each states its queue's rows in its own
# field, which the model and TOGSim read as `key`.
Queue = namedtuple("Queue", "field key")
MATMUL_UNITS = {
    "systolic": Queue("systolic_queue_entries", "tpu_systolic_queue_entries"),
    "tpuv6_mxu": Queue("mxu_queue_entries", "tpu_mxu_queue_entries"),
}
SYSTOLIC = "systolic"


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
        if self.queue_entries <= 0:
            bad.append(f"{self.queue.field}={self.queue_entries} is not positive")
        return bad

    @property
    def queue(self):
        """The field and the model's key of this target's matmul unit's queue."""
        return MATMUL_UNITS[self.matmul_unit]

    @property
    def queue_entries(self):
        """The rows the matmul unit's input and output queues hold, as this target states them."""
        return getattr(self, self.queue.field)

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
        said[self.queue.field] = self.queue_entries
        if self.matmul_unit != SYSTOLIC:
            said["matmul_unit"] = self.matmul_unit
        return said

    def restated_for(self, unit):
        """This machine's description with `unit` as its matmul unit, the same queue's rows in
        that unit's own field: the v6e's numbers on the systolic array, say."""
        said = {f: v for f, v in self.as_dict().items()
                if f not in ("matmul_unit", self.queue.field)}
        said[MATMUL_UNITS[unit].field] = self.queue_entries
        if unit != SYSTOLIC:
            said["matmul_unit"] = unit
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
                    f"{self.queue.key}: {self.queue_entries}\n")
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
    unit = fields["matmul_unit"] = raw.get("matmul_unit", SYSTOLIC)
    if unit not in MATMUL_UNITS:
        raise TargetError(f"{origin}: matmul_unit={unit!r} is not one of {', '.join(MATMUL_UNITS)}")
    own, key = MATMUL_UNITS[unit]
    for other, queue in MATMUL_UNITS.items():
        if other != unit and queue.field in raw:
            raise TargetError(
                f"{origin}: `{queue.field}` is the queue of the {other} unit, and this target's "
                f"matmul_unit is {unit!r}, whose queue is `{own}`. A target states its own unit's "
                f"queue: the {unit} model reads `{key}` alone, so a size under the {other} unit's "
                f"name would reach neither the model nor the lowering.")
    if own not in raw:
        raise TargetError(f"{origin}: target description is missing {own}, the queue of its "
                          f"matmul_unit {unit!r}")
    fields[own] = _number(raw[own], own, origin)
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

