"""What the compiler hands a device, declared instead of guessed from filenames."""
import json
import math
import os

from . import vcix_model

MANIFEST = "kernel.json"

FORMAT = 6
"""6: no `run`; a launch asks run_options() of the target it was compiled for. 5 made `target` the
machine file's keys and `file`, 4 named the accelerator in `run`, 3 added `run` and machine_config."""

MACHINE_CONFIG = "machine.yml"

TRAILING_GRID_PARAMS = ("gridX", "gridY", "gridZ", "pidX", "pidY", "pidZ")

PROGRAM_KEY_SHIFT = {"x": 0, "y": 21, "z": 42}

NP_DTYPE = {
    "float64": "float64", "float32": "float32", "float16": "float16",
    "bfloat16": "uint16",
    "int64": "int64", "int32": "int32", "int16": "int16", "int8": "int8",
    "uint64": "uint64", "uint32": "uint32", "uint16": "uint16", "uint8": "uint8",
    "bool": "int8",
}

ITEMSIZE = {
    "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
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
    """The grid as a TOG is run with it (shape_args, tog_shape.txt): one extent per parallel axis,
    in the order of abi.parallel_axes, outermost first -- the order its grid loops read them in."""
    extent = dict(zip("xyz", grid3(manifest["grid"])))
    return [extent[a] for a in manifest["abi"]["parallel_axes"]]


def program_key_fits(grid):
    """Whether every program id of `grid` packs into its PROGRAM_KEY_SHIFT field -- the key
    the wrapper's dma_index_key and the TOG's togsim_dispatch both name a program by."""
    return all(g <= 1 << PROGRAM_KEY_SHIFT["y"] for g in grid3(grid))


def argv_order(manifest):
    """The .raw names in the order main() expects them on the command line."""
    return [a["name"] for a in manifest["args"]]


SCALAR_CTYPE = {"i32": "int32_t", "i64": "int64_t", "fp32": "float"}


def runtime_scalars(spec):
    """The kernel's scalar parameters in signature order, as (name, C type, default or None).

    They sit between the tensors and the grid in the ELF's parameter list, and main() reads
    their values off argv after the tensor paths; a launch that passes none (gem5's cycle
    run) gets the spec's `scalar_values`, else 0."""
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


def argv_tail(manifest, values=None):
    """The scalar values a launch appends to the tensor paths, as strings in parameter order:
    `values` by name, else the manifest's defaults. All or none: main() counts them."""
    out = []
    for s in manifest.get("scalars", []):
        v = (values or {}).get(s["name"], s["default"])
        if v is None:
            raise ObjectError(
                f"{manifest['name']}: scalar '{s['name']}' has no value for this launch "
                f"and the spec gave it no default")
        out.append(repr(float(v)) if s["ctype"] in ("float", "double", "_Float16")
                   else str(int(v)))
    return out


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


def describe(spec, target, kernel_addr, artifacts):
    """The manifest for one compiled kernel, as a dict."""
    return {
        "format": FORMAT,
        "name": spec.name,
        "grid": list(spec.grid),
        "target": dict(target.as_dict(), name=target.name, file=target.file),
        "args": [{"name": a.name, "role": a.role, "dtype": a.dtype,
                  "shape": list(a.shape)} for a in spec.args],
        "scalars": [{"name": n, "ctype": c, "default": d} for n, c, d in runtime_scalars(spec)],
        "abi": {"parallel_axes": list(spec.parallel_axes),
                "kernel_addr": list(kernel_addr)},
        "artifacts": {k: v for k, v in artifacts.items() if v},
    }


def write(workdir, spec, target, kernel_addr, artifacts):
    """Write the manifest and the model's machine.yml into `workdir`; return the manifest's path."""
    target.write_model_yaml(os.path.join(workdir, MACHINE_CONFIG))
    artifacts = dict(artifacts, machine_config=MACHINE_CONFIG)
    path = os.path.join(workdir, MANIFEST)
    with open(path, "w") as fh:
        json.dump(describe(spec, target, kernel_addr, artifacts), fh, indent=2)
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
