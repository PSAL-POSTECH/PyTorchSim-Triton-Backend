"""The machine this backend compiles for: one TOGSim machine file, $TOGSIM_DIR/configs/<name>.yml,
read for the keys the compiler needs (TOGSim's own names, and isa, vcix, address_map)."""
import json
import os

import yaml

# The address_map keys, every one required.
ADDRESS_MAP = ("spad_vaddr", "dram_base", "dram_size")


# Each `isa` extension as (llc -mattr, gcc -march, Spike --isa) spell it; None: that tool takes none.
# spad-vector-memory is the LLVM fork's: llc refuses a vector memory access outside the scratchpad's
# address space, so one through a plain pointer fails to compile instead of reading another lane's bank.
EXTENSIONS = {
    "m": ("+m", "m", "m"),
    "a": ("+a", "a", "a"),
    "f": ("+f", "f", "f"),
    "d": ("+d", "d", "d"),
    "c": ("+c", "c", "c"),
    "v": ("+v", "v", "v"),
    "zvfh": ("+zvfh", "zfh", "zfh"),
    "zvfbfmin": ("+zvfbfmin", None, "zvfbfmin"),
    "spad-vector-memory": ("+spad-vector-memory", None, None),
}

# pytorchsim_<name>: the accelerator, Coprocessor's model <name>; every one is LLVM's xsfvcp and
# Spike's xcoprocessor adapter.
ACCEL_PREFIX = "pytorchsim_"
ACCEL_SPELLING = ("+xsfvcp", None, "xcoprocessor")

XLEN = 64

_CANONICAL = "imafdgqlcbkjtpvh"

# The target kinds the compiler lowers. A target file states its own (`kind: tpu`); a kind is the devices that
# share one architecture -- its units and their instructions -- and so one lowering, lowering/ext/<kind>_kind,
# and one device dialect, pytorchsim_<kind>.
KINDS = ("tpu",)

# The machine file's mapping of the keys the compiler reads and TOGSim does not.
COMPILER_KEY = "compiler"

# The kind of each built-in accelerator whose machine file (TOGSim's configs/) does not state one yet.
ACCELERATORS = {"pytorchsim_tpuv6e": "tpu", "pytorchsim_tpuv3": "tpu"}

# The vcix key of the registers a matrix-unit operand spans (lower_matmul builds its stream for it).
MXU_REGISTERS_KEY = "tpu_mxu_registers"


def zvl_bits(ext):
    """N of an extension spelled zvl<N>b, else None."""
    if ext.startswith("zvl") and ext.endswith("b") and ext[3:-1].isdigit():
        return int(ext[3:-1])
    return None


def is_accel(ext):
    """Whether `ext` names an accelerator, pytorchsim_<name>."""
    return ext.startswith(ACCEL_PREFIX) and len(ext) > len(ACCEL_PREFIX)


def kind_of(target):
    """The package under lowering/ext/ that lowers `target`'s kind (Target.kind), <kind>_kind; None when
    the compiler lowers no such kind."""
    return f"{target.kind}_kind" if target.kind in KINDS else None


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
        self.file = None

    @property
    def vlen_bits(self):
        """The vector length, as the isa's zvl<N>b states it."""
        return next(zvl_bits(e) for e in self.isa if zvl_bits(e) is not None)

    @property
    def spad_budget(self):
        """The bytes of a lane's scratchpad one kernel may claim: the scratchpad over the copies of
        each buffer the compiler keeps (vpu_spad_buffers, 2 when unsaid: one loading while the
        other is read). The compiler refuses a kernel over it; the front end sizes blocks to it."""
        return self.spad_size // self.spad_buffers

    @property
    def resident_lane_bytes(self):
        """The bytes of a lane's scratchpad a graph may hold an operand resident in across its kernels:
        the scratchpad but the share it keeps for its kernels' own buffers (compiler.spad_scoped_kb_per_lane,
        0 when unsaid; v6e XLA keeps 32 of its 128 MiB, its default scoped VMEM limit)."""
        return self.spad_size - self.spad_scoped

    @property
    def matmul_width(self):
        """The columns one push of the matrix unit covers, MXU_REGISTERS_KEY registers of vector_lanes: a dot
        tile narrower than that costs it as much. Refused when the vcix block does not state the key."""
        if MXU_REGISTERS_KEY not in self.vcix:
            raise TargetError(f"{self.name}: the vcix block does not state {MXU_REGISTERS_KEY}")
        return int(self.vcix[MXU_REGISTERS_KEY]) * self.vector_lanes

    @property
    def accel(self):
        """The accelerator extension, pytorchsim_<name>."""
        return next(e for e in self.isa if is_accel(e))

    @property
    def kind(self):
        """The target kind: the machine file's `kind`, else its built-in accelerator's (ACCELERATORS); none
        for an isa without an accelerator."""
        if self.kind_said is not None:
            return self.kind_said
        return ACCELERATORS.get(next((e for e in self.isa if is_accel(e)), None))

    def check(self):
        """Problems with this description, as a list."""
        bad = []
        n = self.vector_lanes
        if n <= 0 or n & (n - 1):
            bad.append(f"vpu_num_lanes={n} is not a positive power of two. What enforces "
                       f"that is lib_bank_layout.lane_split, which refuses a tile dim "
                       f"neither <= lanes nor a multiple of them -- and Triton tile dims "
                       f"are powers of two, so a non-power-of-two lane count refuses "
                       f"nearly every kernel")
        if self.spad_size <= 0:
            bad.append(f"vpu_spad_size_kb_per_lane={self.spad_size // 1024} is not positive")
        if self.spad_buffers <= 0:
            bad.append(f"vpu_spad_buffers={self.spad_buffers} is not positive")
        if not 0 <= self.spad_scoped < self.spad_size:
            bad.append(f"compiler.spad_scoped_kb_per_lane={self.spad_scoped // 1024} is not at least 0 and "
                       f"under the lane's scratchpad")
        if self.max_lmul not in (1, 2, 4, 8):
            bad.append(f"vpu_max_lmul={self.max_lmul} is none of RVV's register groupings 1, 2, 4, 8")
        if self.vector_registers <= 0:
            bad.append(f"vpu_vector_registers={self.vector_registers} is not positive")
        if not isinstance(self.vcix, dict) or not all(
                isinstance(k, str) and isinstance(v, (int, float, str)) and not isinstance(v, bool)
                for k, v in self.vcix.items()):
            bad.append(f"vcix={self.vcix!r} is not a mapping of the model's keys to numbers or "
                       f"strings")
        if not isinstance(self.isa, list) or not all(isinstance(e, str) for e in self.isa):
            return bad + [f"isa={self.isa!r} is not a list of extension names"]
        isa_bad = _isa_problems(self.isa)
        if isa_bad:
            return bad + isa_bad
        if self.kind_said is not None and not isinstance(self.kind_said, str):
            bad.append(f"kind={self.kind_said!r} is not a target kind's name")
        elif self.kind not in KINDS:
            bad.append(f"the accelerator {self.accel} is of no target kind the compiler lowers: kind is "
                       f"{self.kind_said!r}, and {', '.join(ACCELERATORS)} have theirs built in; kinds: "
                       f"{', '.join(sorted(KINDS))}")
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
        """Spike's -m regions: DRAM alone. The scratchpad is Spike's own buffer at its virtual
        window, with no physical address."""
        return [[self.dram_base, self.dram_size]]

    def banked_bytes(self):
        """The whole scratchpad window, not one lane's share -- sim.h:74."""
        return self.spad_size * self.vector_lanes

    def as_dict(self):
        """The keys of the machine file the compiler reads, as the file states them."""
        said = {"vpu_num_lanes": self.vector_lanes, "vpu_spad_size_kb_per_lane": self.spad_size // 1024,
                "isa": list(self.isa), "vcix": dict(self.vcix),
                "address_map": {k: getattr(self, k) for k in ADDRESS_MAP}}
        if self.togsim_unit is not None:
            said["systolic_array_unit"] = self.togsim_unit
        if self.kind_said is not None:
            said["kind"] = self.kind_said
        if self.dma_load_latency_cycles is not None:
            said["dma_load_latency_cycles"] = self.dma_load_latency_cycles
        if self.spad_scoped:
            said[COMPILER_KEY] = {"spad_scoped_kb_per_lane": self.spad_scoped // 1024}
        return said

    def write_model_yaml(self, path):
        """This same machine in the spelling the accelerator model reads, for Spike's and gem5's
        --machine-config: the core's keys, then the vcix block as it is."""
        with open(path, "w") as f:
            f.write(f"vpu_num_lanes: {self.vector_lanes}\n"
                    f"vpu_spad_size_kb_per_lane: {self.spad_size // 1024}\n"
                    f"vpu_spad_base_vaddr: 0x{self.spad_vaddr:x}\n"
                    f"vpu_vector_length_bits: {self.vlen_bits}\n")
            for key, value in self.vcix.items():
                f.write(f"{key}: {json.dumps(value)}\n")
        return path


def _isa_problems(isa):
    """What is wrong with an isa list: an unknown name, or not exactly one zvl<N>b and one pytorchsim_*."""
    bad = []
    unknown = [e for e in isa if spelling(e) is None]
    if unknown:
        bad.append(f"isa names {', '.join(unknown)}, which no tool spelling is known for; "
                   f"known: {', '.join(EXTENSIONS)}, zvl<N>b, {ACCEL_PREFIX}<model>"
                   + ("; xsfvcp is not stated, every pytorchsim_* is it" if "xsfvcp" in unknown else ""))
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
    """A Target out of an already-decoded machine file, or raise TargetError. Unsaid, max_lmul and
    vector_registers are RVV's widest register group and register file as the core has them."""
    if not isinstance(raw, dict):
        raise TargetError(f"{origin}: a machine file must be a mapping")
    amap = raw.get("address_map")
    missing = [k for k in ("vpu_num_lanes", "vpu_spad_size_kb_per_lane", "isa", "address_map")
               if k not in raw]
    missing += [f"address_map.{k}" for k in ADDRESS_MAP if isinstance(amap, dict) and k not in amap]
    if missing:
        raise TargetError(
            f"{origin}: the machine file does not state {', '.join(missing)}. The compiler reads "
            f"the lanes and their scratchpad, the isa it compiles for and the address map; a "
            f"partly stated machine is refused, not completed")
    if not isinstance(amap, dict):
        raise TargetError(f"{origin}: address_map is not a mapping")
    fields = {k: _number(amap[k], f"address_map.{k}", origin) for k in ADDRESS_MAP}
    fields["vector_lanes"] = _number(raw["vpu_num_lanes"], "vpu_num_lanes", origin)
    fields["spad_size"] = _number(raw["vpu_spad_size_kb_per_lane"], "vpu_spad_size_kb_per_lane",
                                  origin) * 1024
    fields["spad_buffers"] = _number(raw.get("vpu_spad_buffers", 2), "vpu_spad_buffers", origin)
    compiler = raw.get(COMPILER_KEY, {})
    if not isinstance(compiler, dict):
        raise TargetError(f"{origin}: {COMPILER_KEY}={compiler!r} is not a mapping of the compiler's keys")
    fields["spad_scoped"] = _number(compiler.get("spad_scoped_kb_per_lane", 0),
                                    f"{COMPILER_KEY}.spad_scoped_kb_per_lane", origin) * 1024
    fields["max_lmul"] = _number(raw.get("vpu_max_lmul", 8), "vpu_max_lmul", origin)
    fields["vector_registers"] = _number(raw.get("vpu_vector_registers", 32), "vpu_vector_registers", origin)
    fields["isa"] = raw["isa"]
    fields["vcix"] = raw.get("vcix", {})
    fields["togsim_unit"] = raw.get("systolic_array_unit")
    fields["kind_said"] = raw.get("kind")
    fields["dma_load_latency_cycles"] = (None if "dma_load_latency_cycles" not in raw else _number(
        raw["dma_load_latency_cycles"], "dma_load_latency_cycles", origin))
    name = os.path.splitext(os.path.basename(origin))[0]
    t = Target(fields, name, origin)
    bad = t.check()
    if bad:
        raise TargetError(f"{origin}: " + "; ".join(bad))
    return t


def configs_dir():
    """Where a bare name is looked up: the TOGSim checkout's configs/."""
    from .. import toolchain
    return os.path.join(toolchain.TOGSIM_DIR, "configs")


def _decode(path):
    try:
        with open(path) as fh:
            return yaml.safe_load(fh)
    except OSError as e:
        raise TargetError(f"{path}: {e.strerror}")
    except yaml.YAMLError as e:
        raise TargetError(f"{path}: not valid YAML ({e})")


def machines():
    """The bare names under configs_dir() whose files load as a target, the machines the compiler takes."""
    d = configs_dir()
    names = sorted(f[:-len(".yml")] for f in os.listdir(d) if f.endswith(".yml")) \
        if os.path.isdir(d) else []
    return [n for n in names if _loads(os.path.join(d, n + ".yml"))]


def _loads(path):
    try:
        load(path)
        return True
    except TargetError:
        return False


def models():
    """The Coprocessor model each of machines() names in its isa (pytorchsim_<model>), each once."""
    return sorted({load(n).accel[len(ACCEL_PREFIX):] for n in machines()})


def resolve(said):
    """The file a target is named by: a path as given, a bare name as $TOGSIM_DIR/configs/<name>.yml."""
    if os.sep in said or said.endswith((".yml", ".yaml")):
        return said
    path = os.path.join(configs_dir(), said + ".yml")
    if not os.path.isfile(path):
        raise TargetError(f"no machine named {said!r} under {configs_dir()}; have: "
                          f"{', '.join(machines()) or 'none'}. A path to a machine file works too.")
    return path


def load(path):
    """The machine in the file at `path`, or in the TOGSim config a bare name names."""
    path = resolve(path)
    t = parse(_decode(path), path)
    t.file = os.path.abspath(path)
    return t


def as_compiled(said, origin):
    """The machine a kernel object was compiled for: the file its `said` names, loaded, refused
    when that file no longer states what `said` recorded of it at compile time."""
    if not isinstance(said, dict) or not said.get("file"):
        raise TargetError(f"{origin}: names no machine file")
    t = load(said["file"])
    now, then = t.as_dict(), {k: v for k, v in said.items() if k not in ("name", "file")}
    moved = sorted(k for k in set(now) | set(then) if now.get(k) != then.get(k))
    if moved:
        raise TargetError(f"{origin}: {t.file} now states {', '.join(moved)} otherwise than when "
                          f"the object was compiled for it; compile it again")
    return t
