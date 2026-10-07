"""pytorchsim_triton_compiler.lowering.passes -- every compiler pass, one file each, no sub-packages."""

PHASES = [
    ("tts -> transfers", [
        "torchsim-convert-unsupported-ops",
        "torchsim-convert-accesses-to-transfer",
    ]),
    ("layout made into shapes", [
        "torchsim-materialize-layouts",
    ]),
    ("fusion after layout", [
        "torchsim-fuse",
    ]),
    ("tensor -> memref", [
        "torchsim-bufferize",
    ]),
    ("the machine is known", [
        "torchsim-vectorize",
        "torchsim-legalize",
        "torchsim-lower-copies",
    ]),
]

PIPELINE = [name for _, names in PHASES for name in names]

FORBID_ENV = "TORCHSIM_COMPILE_FORBIDS"


def pass_name(mod):
    """A pass's name in MLIR's spelling -- `torchsim-vectorize` for the module
    `p05_vectorize`. The filename holds the words; nothing else writes them
    down, and `_stage4_module` translates back when it needs the file."""
    stem = getattr(mod, "__name__", mod).rsplit(".", 1)[-1]
    return "torchsim-" + stem.split("_", 1)[1].replace("_", "-")


def _file_stem(name):
    """`torchsim-vectorize` -> `vectorize`; a stem passed in comes back."""
    if name.startswith("torchsim-"):
        name = name[len("torchsim-"):]
    return name.replace("-", "_")


def canon_pass_name(name):
    """One meaning per pass however it is spelled, so a settings key written the
    old way turns a check ON rather than silently off -- which is what a plain
    string compare did the day the names became MLIR-spelled."""
    return "torchsim-" + _file_stem(name).replace("_", "-")


def _stage4_module(name):
    """The pass module whose file is `p<NN>_<name>.py`, or None. BY THE FILENAME,
    the one thing about these modules that is already load-bearing: a table would
    be a third place to keep in step."""
    import importlib
    import os
    stem = _file_stem(name)
    here = os.path.dirname(os.path.abspath(__file__))
    for f in sorted(os.listdir(here)):
        if (f[:1] == "p" and f.endswith(".py") and f[1:3].isdigit()
                and f[:-3].split("_", 1)[1:] == [stem]):
            return importlib.import_module("." + f[:-3], __name__)
    return None


def _split_attr(pattern):
    """("affine.for[torchsim.reduce_loop]") -> ("affine.for", "torchsim.reduce_loop"); else (p, None)."""
    if pattern.endswith("]") and "[" in pattern:
        head, _, attr = pattern[:-1].partition("[")
        return head, attr
    return pattern, None


def _forbid_matches(op_name, patterns, attrs=()):
    """Does this op match `patterns`? See FORBID_ENV above for the four forms.
    `attrs` is the op's attribute NAMES, consulted only by a pattern that asks
    for one -- an op with none matches the other three forms as it always did."""
    for p in patterns:
        if not p.startswith("-"):
            continue
        head, attr = _split_attr(p[1:])
        if head == op_name and (attr is None or attr in attrs):
            return False
    for p in patterns:
        if p.startswith("-"):
            continue
        head, attr = _split_attr(p)
        if attr is not None and attr not in attrs:
            continue
        if head.endswith(".*"):
            if op_name.startswith(head[:-1]):
                return True
        elif op_name == head:
            return True
    return False


def forbids(name):
    """(patterns, enforced-by-default, source file) declared by pass `name`, and
    ((), True, None) for a name that is not a stage-4 pass: a pass declaring
    nothing forbids nothing, which is the resting state."""
    mod = _stage4_module(name)
    if mod is None:
        return (), True, None
    return (tuple(getattr(mod, "FORBIDS", ())),
            bool(getattr(mod, "FORBIDS_DEFAULT", True)),
            getattr(mod, "__file__", None))


def forbids_enabled(name, extra=None):
    """Whether pass `name`'s declaration is checked on this run."""
    import os
    _, on, _ = forbids(name)
    want = canon_pass_name(name)
    settings = [(k, bool(v)) for k, v in (extra or {}).get("forbids", {}).items()]
    for item in os.environ.get(FORBID_ENV, "").split(","):
        k, _, v = item.strip().partition("=")
        if k:
            settings.append((k, v not in ("0", "false", "no")))
    for k, v in settings:
        if _stage4_module(k) is None:
            raise ValueError(f"forbids names {k!r}, which is no pass")
        if canon_pass_name(k) == want:
            on = v
    return on


def forbidden(name, module, extra=None):
    """[(op name, count, the first one, IN FULL)] `name` promised would not
    survive; empty when it did its job. THE OP AND NOT ONLY ITS LOCATION: stage 3
    prints generic form with no debug info, and the first line carries the types."""
    patterns, _, _ = forbids(name)
    if not patterns or not forbids_enabled(name, extra):
        return []
    from .lib_ir import walk
    found = {}
    for op in walk(module.operation):
        n = getattr(op, "name", None)
        if n is None:
            continue
        names = ()
        if any("[" in p for p in patterns):
            try:
                names = {op.attributes[i].name for i in range(len(op.attributes))}
            except (TypeError, ValueError, IndexError):
                names = ()
        if not _forbid_matches(n, patterns, names):
            continue
        if n not in found:
            head = " ".join(str(op).strip().splitlines()[0].split())
            found[n] = [0, f"{head[:180]} at {op.location}"]
        found[n][0] += 1
    return sorted(((n, c, where) for n, (c, where) in found.items()),
                  key=lambda f: -f[1])


import sys as _sys
from ... import toolchain as _toolchain

_toolchain.activate_bindings(_sys.path)

from . import p08_lower_matmul, p09_lower_to_vcix, p10_lower_to_xlu  # noqa: E402
from .p12_lower_to_llvm import run_standard_lowering  # noqa: E402,F401

POST_OPT_PASSES = [
    p08_lower_matmul,
    p09_lower_to_vcix,
    p10_lower_to_xlu,
]


def _write_atomic(out_path, text):
    """Write `text` to `out_path` through a temp file in the same directory.
    A CONCURRENT COMPILE MUST NEVER SEE A TRUNCATED FILE -- mlir-opt would parse
    one to an empty module and drop the kernel; rename on one directory is atomic."""
    import os
    import tempfile
    d = os.path.dirname(os.path.abspath(out_path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, out_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def run_module_passes(in_path, out_path, passes, dump=None, **opts):
    """Parse `in_path` once and run, in order, each pass whose MARKERS the module holds when its
    turn comes; print once to `out_path`. `dump(name, module)` after each pass that ran."""
    with open(in_path) as f:
        text = f.read()
    if not any(mk in text for p in passes for mk in p.MARKERS):
        if out_path != in_path:
            import shutil
            shutil.copyfile(in_path, out_path)
        return False

    from mlir.ir import Context, Module, Location
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        module = Module.parse(text)
        for p in passes:
            if not any(mk in text for mk in p.MARKERS):
                continue
            p.run(module, **opts)
            nm = pass_name(p)
            left = forbidden(nm, module, opts.get("forbids_extra"))
            if left:
                shown = ", ".join(f"{n} x{c}" for n, c, _ in left)
                raise RuntimeError(
                    f"{nm} declared these would not survive it and they did: "
                    f"{shown}. First one: {left[0][2]}")
            if dump is not None:
                dump(nm, module)
            text = str(module)
        out = text

    _write_atomic(out_path, out)
    return True
