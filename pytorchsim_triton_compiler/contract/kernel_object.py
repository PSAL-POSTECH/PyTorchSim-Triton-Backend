"""What the compiler hands a device, declared instead of guessed from filenames."""
import json
import math
import os

from . import vcix_model

MANIFEST = "kernel.json"

FORMAT = 10
"""10: `spad_bytes_per_lane`, and each copy node's; 9: `copy_nodes`; 8: runtime attributes streamed as record files, their `table_hash`, the build's `widths`;
7: `runtime_attributes`; 6: no `run`; 5 made `target` the machine file's keys and `file`, 4 named the
accelerator in `run`, 3 added `run`."""

COPY_NODES = "copy_nodes"
"""The TOGs a launch runs before the kernel's, in order, each {arg, bytes, spad_bytes_per_lane, held_buffer,
tog_so}: a copy node loads argument `arg` whole, and the kernel's TOG reads it as its buffer `held_buffer`."""

SPAD = "spad_bytes_per_lane"
"""The ELF's .spad section, the bytes per lane its work-items hold in the scratchpad at once: what every
dispatch of the kernel's TOG states as its `.spad`."""

MACHINE_CONFIG = "machine.yml"

TRAILING_GRID_PARAMS = ("gridX", "gridY", "gridZ", "pidX", "pidY", "pidZ")

PROGRAM_KEY_SHIFT = {"x": 0, "y": 21, "z": 42}

CYCLE_CALLS = 2
"""The cycle binary calls its one program this many times and gem5 times every call: call c of tile t
of n is row t + n * c of TOGSim's cycle and unit tables. The TOG computes on the last call
(variant CYCLE_CALLS - 1), since every tile but the first runs on a warm front end."""

CYCLE_CALL_GLOBAL = "__pytorchsim_cycle_call"
"""The cycle binary's call index, 0..CYCLE_CALLS-1: set by the wrapper before each call, passed by
every start marker to gem5 beside the tile id."""

NP_DTYPE = {
    "float64": "float64", "float32": "float32", "float16": "float16",
    "bfloat16": "uint16", "float8_e5m2": "uint8", "float8_e4m3fn": "uint8",
    "int64": "int64", "int32": "int32", "int16": "int16", "int8": "int8",
    "uint64": "uint64", "uint32": "uint32", "uint16": "uint16", "uint8": "uint8",
    "bool": "int8",
}

ITEMSIZE = {
    "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2, "float8_e5m2": 1, "float8_e4m3fn": 1,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1,
    "uint64": 8, "uint32": 4, "uint16": 2, "uint8": 1,
    "bool": 1,
}


_undeclared = set(NP_DTYPE) ^ set(ITEMSIZE)
if _undeclared:
    raise ImportError(f"NP_DTYPE and ITEMSIZE disagree about {sorted(_undeclared)}: "
                      f"both answer for every dtype a manifest may name")


class ObjectError(RuntimeError):
    pass


def numel(arg):
    """Elements in one manifest argument."""
    return math.prod(arg["shape"])


def nbytes(arg):
    """Bytes one manifest argument occupies as a .raw file."""
    return numel(arg) * ITEMSIZE[arg["dtype"]]


def grid3(grid):
    """The launch grid padded to three axes, as the wrapper expects it."""
    g = list(grid) + [1, 1, 1]
    return g[0], g[1], g[2]


def launch_shape(manifest):
    """The grid as a TOG is run with it, the head of shape_args: one extent per parallel axis, in
    the order of abi.parallel_axes, outermost first -- the order its grid loops read them in."""
    extent = dict(zip("xyz", grid3(manifest["grid"])))
    return [extent[a] for a in manifest["abi"]["parallel_axes"]]


def launch_scalars(scalars):
    """The entries of a manifest's `scalars` a TOG is handed after the grid extents: every integer
    parameter, in signature order. A float is never an extent, offset or bound."""
    return [s for s in scalars if s["ctype"] not in FLOAT_CTYPES]


def shape_args(manifest, values=None, grid=None):
    """What a TOG is run with (shape_args, tog_shape.txt): launch_shape (or `grid`), then the value
    of every launch_scalars parameter -- `values` by name, else the manifest's defaults."""
    ext = launch_shape(manifest) if grid is None else [int(g) for g in grid]
    return ext + [int(scalar_value(manifest, s, values))
                  for s in launch_scalars(manifest.get("scalars", []))]


def program_key_fits(grid):
    """Whether every program id of `grid` packs into its PROGRAM_KEY_SHIFT field -- the key
    the wrapper's dma_index_key and the TOG's togsim_dispatch both name a program by."""
    return all(g <= 1 << PROGRAM_KEY_SHIFT["y"] for g in grid3(grid))


def argv_order(manifest):
    """The .raw names in the order main() expects them on the command line."""
    return [a["name"] for a in manifest["args"]]


RUNTIME_ATTRIBUTE_DIR = "runtime_attribute"
"""Where, under a launch's runtime dir, the records go as files: each worker's record file, then what a
TOG's attributes read."""

RUNTIME_ATTRIBUTE_WRITE = "pytorchsim_runtime_attribute_write"
"""The wrapper's function a record is written by: (uint32 id, data, uint64 nbytes), data the record's
extents (rank i64) then its values row-major."""

RUNTIME_ATTRIBUTE_TABLE_HASH = "pytorchsim_runtime_attribute_table_hash"
"""The wrapper's uint64 the record file's header states: the hash of the lowered module's attribute table."""

RUNTIME_ATTRIBUTE_PROGRAM_ID = 0xFFFFFFFF
"""The id of the record the wrapper writes, with no data, before each program: the program marker."""

RECORD_FILE_MAGIC = b"TSRA"

PROGRAMS_FILE = "programs.bin"
TOGSIM_VERSION = (1, 0, 0)
"""The TOGSim version this compiler writes TOGs and runtime attribute records for. TOGSim takes a TOG or
a record file of its own major version and no newer than itself."""
TOGSIM_VERSION_NUMBER = TOGSIM_VERSION[0] << 16 | TOGSIM_VERSION[1] << 8 | TOGSIM_VERSION[2]
"""TOGSIM_VERSION as togsim_runtime.h encodes it (TOGSIM_VERSION), the u32 every record header carries."""
PROGRAMS_MAGIC, ATTRIBUTE_FILE_MAGIC = 0x50415354, 0x41525354


def record_file(worker):
    """The record file of spike worker `worker`, relative to the launch's runtime dir."""
    return os.path.join(RUNTIME_ATTRIBUTE_DIR, f"{worker}.rec")


def argv_records(manifest, worker):
    """The record file's path, which main() expects right after the tensor paths, or none."""
    return [record_file(worker)] if manifest.get("runtime_attributes") else []


def program_key(pid):
    """The key a program is known by: its ids packed at PROGRAM_KEY_SHIFT."""
    return sum(int(i) << PROGRAM_KEY_SHIFT[a] for a, i in zip("xyz", pid))


SCALAR_CTYPE = {"i32": "int32_t", "i64": "int64_t", "fp32": "float"}

FLOAT_CTYPES = ("float", "double", "_Float16")


def parameters(spec):
    """The kernel function's parameters before the grid, in signature order: ("tensor", k) for the
    k-th pointer (spec.args[k]) and ("scalar", k) for the k-th runtime_scalars entry. The one
    statement of where each sits, as the ELF calls the kernel and the TOG reads its arguments."""
    out, t, k = [], 0, 0
    for sig in spec.signature.values():
        if sig == "constexpr":
            continue
        if sig.startswith("*"):
            out.append(("tensor", t))
            t += 1
        else:
            out.append(("scalar", k))
            k += 1
    return out


def runtime_scalars(spec):
    """The kernel's scalar parameters in signature order, as (name, C type, default or None).

    The kernel takes them where `parameters` says, and main() reads their values off argv after
    the tensor paths; a launch that passes none (gem5's cycle run) gets the spec's
    `scalar_values`, else 0."""
    values = spec.extra.get("scalar_values", {})
    declared = dict(spec.extra.get("scalar_args", []))
    out = []
    for name, sig in spec.signature.items():
        if sig == "constexpr" or sig.startswith("*"):
            continue
        ctype = declared.get(name) or SCALAR_CTYPE.get(sig)
        if ctype is None:
            raise ValueError(
                f"kernel {spec.name}: no C type known for scalar '{name}: {sig}'")
        out.append((name, ctype, values.get(name)))
    return out


def scalar_entries(spec):
    """The manifest's `scalars`: runtime_scalars(spec) as {name, ctype, default}."""
    return [{"name": n, "ctype": c, "default": d} for n, c, d in runtime_scalars(spec)]


def scalar_value(manifest, s, values=None):
    """The value of scalar entry `s` for a launch: `values` by name, else the manifest's default."""
    v = (values or {}).get(s["name"], s["default"])
    if v is None:
        raise ObjectError(
            f"{manifest['name']}: scalar '{s['name']}' has no value for this launch "
            f"and the spec gave it no default")
    return v


def argv_tail(manifest, values=None):
    """The scalar values a launch appends to the tensor paths, as strings in parameter order:
    `values` by name, else the manifest's defaults. All or none: main() counts them."""
    return [repr(float(v)) if s["ctype"] in FLOAT_CTYPES else str(int(v))
            for s in manifest.get("scalars", []) for v in [scalar_value(manifest, s, values)]]


def writable(manifest):
    """The arguments the kernel writes -- out and inout."""
    return [a for a in manifest["args"] if a["role"] in ("out", "inout")]


def splittable(manifest):
    """Whether the launch grid may be cut into ranges across processes."""
    return not any(a["role"] == "inout" for a in manifest["args"])


def _prefetch_list(graph_prefetch):
    """KernelSpec.graph_prefetch as the manifest states it: a list, or None (a manifest without it too)
    for the compiler's own choice."""
    return None if graph_prefetch is None else list(graph_prefetch)


def disagreements(manifest, spec):
    """Every question BOTH sides answer, asked of both."""
    return [what for what, mine, theirs in (
        ("argv_order", argv_order(manifest), [a.name for a in spec.args]),
        ("scalars", [x["name"] for x in manifest.get("scalars", [])],
                    [n for n, _, _ in runtime_scalars(spec)]),
        ("grid3", grid3(manifest["grid"]), grid3(spec.grid)),
        ("numel", [numel(a) for a in manifest["args"]],
                  [a.numel for a in spec.args]),
        ("writable", [a["name"] for a in writable(manifest)],
                     [a.name for a in spec.outputs()]),
        ("graph_prefetch", [manifest.get("graph_prefetch")], [_prefetch_list(spec.graph_prefetch)]),
    ) if list(mine) != list(theirs)]


def refuse_if_it_disagrees(manifest, spec):
    """Refuse a manifest that answers differently from `spec`."""
    bad = disagreements(manifest, spec)
    if bad:
        raise ObjectError(
            f"{spec.name}: the manifest in this kernel object and the spec "
            f"disagree about {', '.join(bad)}. It was written by a different "
            f"compile; see disagreements().")


SPIKE_OPTIONS = ("spike_isa", "memory", "model", "machine_config")


def run_options(target, workdir, vcix_build, manifest=None, keys=None, **instead):
    """How a device runs the object in `workdir` compiled for `target`: Spike's --isa and -m, the model
    library in `vcix_build`, its --machine-config, the model's gem5 script; only `keys` of them when
    given, so a Spike run needs no gem5 machine. A truthy `instead` is taken unasked."""
    answers = {
        "spike_isa": target.spike_isa,
        "memory": lambda: ",".join(f"0x{base:x}:0x{size:x}" for base, size in target.memory()),
        "model": lambda: _model(target, vcix_build),
        "machine_config": lambda: artifact(workdir, "machine_config", manifest=manifest),
        "gem5_script": lambda: vcix_model.gem5_script(target.accel),
    }
    wanted = list(answers) if keys is None else list(keys)
    unknown = sorted((set(instead) | set(wanted)) - set(answers))
    if unknown:
        raise ValueError(f"run_options answers {', '.join(answers)}; not {', '.join(unknown)}")
    return {k: instead.get(k) or answers[k]() for k in wanted}


def _model(target, vcix_build):
    """`target`'s model in `vcix_build`, refused unless its units include systolic_array_unit."""
    path = vcix_model.find(vcix_build, target.accel)
    have = vcix_model.units(path)
    if target.togsim_unit not in have:
        raise vcix_model.ModelError(f"{target.file or target.name}: systolic_array_unit is "
                                    f"{target.togsim_unit or '(unsaid)'}, and the model "
                                    f"{os.path.basename(path)} holds {', '.join(have)}")
    return path


def source_names(spec, numbers):
    """The spec.args names of the kernel function's arguments `numbers`, as `parameters` places them; a
    number that is no tensor's is a ValueError."""
    params = parameters(spec)
    bad = [n for n in numbers if not 0 <= n < len(params) or params[n][0] != "tensor"]
    if bad:
        raise ValueError(f"{spec.name}: runtime attribute sources {bad} are no tensor argument of {params}")
    return [spec.args[params[n][1]].name for n in numbers]


def describe(spec, target, kernel_addr, artifacts, spad_bytes_per_lane, runtime_attributes=None, widths=None):
    """The manifest for one compiled kernel, as a dict: `spad_bytes_per_lane` its ELF's .spad (SPAD);
    `runtime_attributes` the lowered module's attribute table, its hash and each attribute's `sources` (tensor
    args); `widths` the spill decision the ELF was built at, which the cycle binary is held to."""
    out = {
        "format": FORMAT,
        "name": spec.name,
        SPAD: int(spad_bytes_per_lane),
        "grid": list(spec.grid),
        "target": dict(target.as_dict(), name=target.name, file=target.file),
        "args": [{"name": a.name, "role": a.role, "dtype": a.dtype,
                  "shape": list(a.shape)} for a in spec.args],
        "scalars": scalar_entries(spec),
        "abi": {"parallel_axes": list(spec.parallel_axes),
                "kernel_addr": list(kernel_addr)},
        "artifacts": {k: v for k, v in artifacts.items() if v},
        "graph_prefetch": _prefetch_list(spec.graph_prefetch),
    }
    if widths is not None:
        out["widths"] = widths
    if runtime_attributes:
        attributes = [dict(a, sources=source_names(spec, a["sources"])) for a in runtime_attributes["attributes"]]
        out["runtime_attributes"] = dict(runtime_attributes, attributes=attributes)
    return out


def write(workdir, spec, target, kernel_addr, artifacts, spad_bytes_per_lane, runtime_attributes=None,
          widths=None):
    """Write the manifest and the model's machine.yml into `workdir`; return the manifest's path."""
    target.write_model_yaml(os.path.join(workdir, MACHINE_CONFIG))
    artifacts = dict(artifacts, machine_config=MACHINE_CONFIG)
    path = os.path.join(workdir, MANIFEST)
    with open(path, "w") as fh:
        json.dump(describe(spec, target, kernel_addr, artifacts, spad_bytes_per_lane, runtime_attributes, widths),
                  fh, indent=2)
        fh.write("\n")
    return path


def copy_nodes(manifest):
    """The manifest's COPY_NODES, none when it states none (a build without the TOG stage)."""
    return list(manifest.get(COPY_NODES, ()))


def set_copy_nodes(workdir, nodes):
    """State `nodes` as the existing manifest's COPY_NODES, replacing what a previous TOG stage stated."""
    got = read(workdir)
    got[COPY_NODES] = list(nodes)
    with open(os.path.join(workdir, MANIFEST), "w") as fh:
        json.dump(got, fh, indent=2)
        fh.write("\n")


def add_artifact(workdir, kind, relpath):
    """Name one more output in an existing manifest, and return its path."""
    got = read(workdir)
    got.setdefault("artifacts", {})[kind] = relpath
    path = os.path.join(workdir, MANIFEST)
    with open(path, "w") as fh:
        json.dump(got, fh, indent=2)
        fh.write("\n")
    return path


def read(workdir):
    """The manifest in `workdir`."""
    path = os.path.join(workdir, MANIFEST)
    try:
        with open(path) as fh:
            got = json.load(fh)
    except OSError as e:
        raise ObjectError(
            f"{path}: {e.strerror}. A kernel object carries its own manifest -- "
            f"a directory without one was not produced by a compile that "
            f"reached the binary stage.")
    except json.JSONDecodeError as e:
        raise ObjectError(f"{path}: not valid JSON ({e})")
    if got.get("format") != FORMAT:
        raise ObjectError(
            f"{path}: this kernel object is stale -- its manifest is format "
            f"{got.get('format')!r} and this compiler reads and writes format {FORMAT}. "
            f"Rebuild it: compile the kernel again, into an empty workdir or cache entry.")
    return got


def artifact(workdir, kind, required=True, manifest=None):
    """One artifact's absolute path, by the name the manifest gives it."""
    got = read(workdir) if manifest is None else manifest
    rel = got.get("artifacts", {}).get(kind)
    if not rel:
        if not required:
            return None
        have = ", ".join(sorted(got.get("artifacts", {}))) or "none"
        raise ObjectError(
            f"{workdir}: this kernel object has no `{kind}`. It has: {have}.")
    return os.path.join(workdir, rel)
