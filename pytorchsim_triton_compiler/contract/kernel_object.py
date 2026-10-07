"""What the compiler hands a device, declared instead of guessed from filenames."""
import json
import math
import os

MANIFEST = "kernel.json"

FORMAT = 1

TRAILING_GRID_PARAMS = ("gridX", "gridY", "gridZ", "pidX", "pidY", "pidZ")

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


def argv_order(manifest):
    """The .raw names in the order main() expects them on the command line."""
    return [a["name"] for a in manifest["args"]]


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


def describe(spec, target, kernel_addr, artifacts):
    """The manifest for one compiled kernel, as a dict."""
    return {
        "format": FORMAT,
        "name": spec.name,
        "grid": list(spec.grid),
        "target": dict(target.as_dict(), name=target.name),
        "args": [{"name": a.name, "role": a.role, "dtype": a.dtype,
                  "shape": list(a.shape)} for a in spec.args],
        "abi": {"parallel_axes": list(spec.parallel_axes),
                "kernel_addr": list(kernel_addr)},
        "artifacts": {k: v for k, v in artifacts.items() if v},
    }


def write(workdir, spec, target, kernel_addr, artifacts):
    """Write the manifest into `workdir`, and return its path."""
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
            f"{path}: manifest format {got.get('format')!r}, this reader knows "
            f"{FORMAT}. Recompile, or read it with the matching version.")
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
