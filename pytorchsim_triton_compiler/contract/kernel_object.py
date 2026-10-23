"""What the compiler hands a device, declared instead of guessed from filenames."""
import json
import math
import os

from . import vcix_model

MANIFEST = "kernel.json"

FORMAT = 7
"""7: `runtime_attributes`, the hidden output and the args each attribute is read from; 6: no `run`; a launch asks run_options() of the target it was compiled for. 5 made `target` the
machine file's keys and `file`, 4 named the accelerator in `run`, 3 added `run` and machine_config."""

MACHINE_CONFIG = "machine.yml"

TRAILING_GRID_PARAMS = ("gridX", "gridY", "gridZ", "pidX", "pidY", "pidZ")

PROGRAM_KEY_SHIFT = {"x": 0, "y": 21, "z": 42}

CYCLE_CALLS = 2
"""The cycle binary calls its one program this many times and gem5 times every call: call c of tile t
of n is row t + n * c of TOGSim's cycle and unit tables. The TOG computes on the last call
(variant CYCLE_CALLS - 1), since every tile but the first runs on a warm front end."""

CYCLE_CALL_GLOBAL = "__torchsim_cycle_call"
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


RUNTIME_ATTRIBUTE_ROW = "__torchsim_runtime_attribute_row"
"""The program's row of the hidden output: an int64 array the wrapper defines and copies into the
output after each program, which the kernel's records write."""

RUNTIME_ATTRIBUTE_HEAD = 2
"""Words at the head of a row: whether the program ran, and its key."""

RUNTIME_ATTRIBUTE_DIR = "runtime_attribute"
"""Where, under a launch's runtime dir, the records go as files: what a TOG's attributes read."""

HIDDEN_FILE = "runtime_attributes.raw"
"""The hidden output's file, beside the tensors' .raw files: every program's row."""

PROGRAMS_FILE = "programs.bin"
ATTRIBUTE_ABI = 17
_PROGRAMS_MAGIC, _RECORD_MAGIC = 0x50415354, 0x41525354


def argv_hidden(manifest):
    """The hidden output's file name, which main() expects right after the tensor paths, or none."""
    got = manifest.get("runtime_attributes")
    return [got["file"]] if got else []


def program_key(pid):
    """The key a program is known by: its ids packed at PROGRAM_KEY_SHIFT."""
    return sum(int(i) << PROGRAM_KEY_SHIFT[a] for a, i in zip("xyz", pid))


def read_rows(manifest, raw_path):
    """The hidden output at `raw_path` as an int64 array of one row per program; ObjectError when the
    run left none or another size."""
    import numpy as np
    got = manifest["runtime_attributes"]
    if not os.path.isfile(raw_path) or not os.path.getsize(raw_path):
        raise ObjectError(f"the kernel declares runtime attributes and its run wrote no {raw_path}")
    want = got["programs"] * got["row_words"]
    rows = np.fromfile(raw_path, dtype=np.int64)
    if rows.size != want:
        raise ObjectError(f"{raw_path} is {rows.size * 8} bytes, the manifest says {got['programs']} rows "
                          f"of {got['row_words']} words")
    return rows.reshape(got["programs"], got["row_words"])


def merge_rows(manifest, parts):
    """One hidden output from the workers' `parts` (each a full array of rows): each row from the one
    worker whose program ran there; ObjectError when two did."""
    import numpy as np
    out = np.zeros_like(parts[0])
    seen = np.zeros(parts[0].shape[0], dtype=bool)
    for rows in parts:
        ran = rows[:, 0] != 0
        both = np.flatnonzero(ran & seen)
        if both.size:
            raise ObjectError(f"program row {int(both[0])} ran in two program ranges")
        out[ran] = rows[ran]
        seen |= ran
    return out


def write_runtime_attributes(manifest, rows, out_dir):
    """Clear `out_dir` and write the run's records there: PROGRAMS_FILE (the keys that ran) and one
    attr_<key>_<id>.bin per record a program made, its extents and its values; ObjectError on any row
    that does not say one record per attribute at most, within its capacity."""
    import shutil
    import numpy as np
    got = manifest["runtime_attributes"]
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    keys, rows_of = [], {}
    for f, row in enumerate(rows):
        if row[0] not in (0, 1):
            raise ObjectError(f"program row {f} has ran={int(row[0])}")
        if not row[0]:
            continue
        key = int(row[1])
        if key in rows_of:
            raise ObjectError(f"program rows {rows_of[key]} and {f} both name key {key}")
        rows_of[key] = f
        keys.append(key)
        for a in got["attributes"]:
            _write_record(out_dir, key, a, row)
    if not keys:
        raise ObjectError("no program ran")
    with open(os.path.join(out_dir, PROGRAMS_FILE), "wb") as fh:
        np.array([_PROGRAMS_MAGIC, ATTRIBUTE_ABI], dtype=np.uint32).tofile(fh)
        np.array([len(keys)] + sorted(keys), dtype=np.uint64).tofile(fh)
    return sorted(keys)


def _write_record(out_dir, key, a, row):
    import numpy as np
    at, rank, cap = a["offset"], a["rank"], a["capacity"]
    count, fill = int(row[at]), int(row[at + 1])
    if count == 0:
        return
    if count > 1:
        raise ObjectError(f"key {key} recorded attribute {a['id']} {count} times")
    extents = [int(e) for e in row[at + 2:at + 2 + rank]]
    if any(e < 0 for e in extents) or any(e > c for e, c in zip(extents, cap)):
        raise ObjectError(f"key {key} attribute {a['id']} extents {extents} exceed its static capacity {cap}")
    if a.get("fill") is not None and fill != math.prod(extents[:a["fill"]]):
        raise ObjectError(f"key {key} attribute {a['id']} filled {fill} of "
                          f"{math.prod(extents[:a['fill']])} trip rows")
    bits = a.get("elem_bits", 64)
    words = row[at + 2 + rank:at + 2 + rank + -(-math.prod(cap) * bits // 64)]
    values = (np.ascontiguousarray(words).view({8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}[bits])
              [:math.prod(extents)].astype(np.int64))
    path = os.path.join(out_dir, f"attr_{key}_{a['id']}.bin")
    with open(path, "wb") as fh:
        np.array([_RECORD_MAGIC, ATTRIBUTE_ABI, a["id"], rank], dtype=np.uint32).tofile(fh)
        np.array([key], dtype=np.uint64).tofile(fh)
        np.array(extents + [int(v) for v in values], dtype=np.int64).tofile(fh)


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
    ) if list(mine) != list(theirs)]


def refuse_if_it_disagrees(manifest, spec):
    """Refuse a manifest that answers differently from `spec`."""
    bad = disagreements(manifest, spec)
    if bad:
        raise ObjectError(
            f"{spec.name}: the manifest in this kernel object and the spec "
            f"disagree about {', '.join(bad)}. It was written by a different "
            f"compile; see disagreements().")


def run_options(target, workdir, vcix_build, manifest=None, **instead):
    """How a device runs the object in `workdir` compiled for `target`: Spike's --isa and -m, the model
    library in `vcix_build`, its --machine-config, the model's gem5 script. A truthy `instead` is taken
    unasked."""
    answers = {
        "spike_isa": target.spike_isa,
        "memory": lambda: ",".join(f"0x{base:x}:0x{size:x}" for base, size in target.memory()),
        "model": lambda: vcix_model.find(vcix_build, target.accel),
        "machine_config": lambda: artifact(workdir, "machine_config", manifest=manifest),
        "gem5_script": lambda: vcix_model.gem5_script(target.accel),
    }
    unknown = sorted(set(instead) - set(answers))
    if unknown:
        raise ValueError(f"run_options answers {', '.join(answers)}; not {', '.join(unknown)}")
    return {k: instead.get(k) or answer() for k, answer in answers.items()}


def source_names(spec, numbers):
    """The spec.args names of the kernel function's arguments `numbers`, as `parameters` places them; a
    number that is no tensor's is a ValueError."""
    params = parameters(spec)
    bad = [n for n in numbers if not 0 <= n < len(params) or params[n][0] != "tensor"]
    if bad:
        raise ValueError(f"{spec.name}: runtime attribute sources {bad} are no tensor argument of {params}")
    return [spec.args[params[n][1]].name for n in numbers]


def describe(spec, target, kernel_addr, artifacts, runtime_attributes=None):
    """The manifest for one compiled kernel, as a dict; `runtime_attributes`, the lowered module's
    slot table, names the hidden output and each attribute's `sources`, the tensor args it is read from."""
    out = {
        "format": FORMAT,
        "name": spec.name,
        "grid": list(spec.grid),
        "target": dict(target.as_dict(), name=target.name, file=target.file),
        "args": [{"name": a.name, "role": a.role, "dtype": a.dtype,
                  "shape": list(a.shape)} for a in spec.args],
        "scalars": scalar_entries(spec),
        "abi": {"parallel_axes": list(spec.parallel_axes),
                "kernel_addr": list(kernel_addr)},
        "artifacts": {k: v for k, v in artifacts.items() if v},
    }
    if runtime_attributes:
        attributes = [dict(a, sources=source_names(spec, a["sources"])) for a in runtime_attributes["attributes"]]
        out["runtime_attributes"] = dict(runtime_attributes, attributes=attributes, file=HIDDEN_FILE,
                                         programs=math.prod(grid3(spec.grid)))
    return out


def write(workdir, spec, target, kernel_addr, artifacts, runtime_attributes=None):
    """Write the manifest and the model's machine.yml into `workdir`; return the manifest's path."""
    target.write_model_yaml(os.path.join(workdir, MACHINE_CONFIG))
    artifacts = dict(artifacts, machine_config=MACHINE_CONFIG)
    path = os.path.join(workdir, MANIFEST)
    with open(path, "w") as fh:
        json.dump(describe(spec, target, kernel_addr, artifacts, runtime_attributes), fh, indent=2)
        fh.write("\n")
    return path


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
