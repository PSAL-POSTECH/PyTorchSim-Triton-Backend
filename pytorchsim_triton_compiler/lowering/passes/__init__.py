"""pytorchsim_triton_compiler.lowering.passes -- every compiler pass, a file <layer><serial>_<name>
(layer a common, b device, c target; serials count up by one), machine-only passes under lowering/ext/, and
transform.py and lower.py that run them in order. Shared here: a pass's name and what it FORBIDS."""
import sys as _sys

from ... import toolchain as _toolchain

_toolchain.activate_bindings(_sys.path)

FORBID_ENV = "PYTORCHSIM_COMPILE_FORBIDS"


def _is_pass_file(stem):
    return stem[:1] in "abc" and stem[1:3].isdigit() and stem[3:4] == "_"


def pass_name(mod):
    """A pass's MLIR name -- `pytorchsim-vectorize` for the module `b07_vectorize`; a conversion
    (`convert-...`) or an upstream dialect's pass (`arith-...`) keeps its file's name as is, and any other
    module states its NAME."""
    stem = getattr(mod, "__name__", mod).rsplit(".", 1)[-1]
    if _is_pass_file(stem):
        name = stem[4:].replace("_", "-")
        head = name.split("-", 1)[0]
        return name if head == "convert" or _is_upstream_dialect(head) else "pytorchsim-" + name
    return mod.NAME


def _is_upstream_dialect(name):
    import importlib.util
    return importlib.util.find_spec("mlir.dialects." + name) is not None


def _pass_module(name):
    import importlib
    import os
    want = name
    here = os.path.dirname(os.path.abspath(__file__))
    for f in sorted(os.listdir(here)):
        if f.endswith(".py") and _is_pass_file(f[:-3]) and pass_name(f[:-3]) == want:
            return importlib.import_module("." + f[:-3], __name__)
    return None


def _split_attr(pattern):
    if pattern.endswith("]") and "[" in pattern:
        head, _, attr = pattern[:-1].partition("[")
        return head, attr
    return pattern, None


def _forbid_matches(op_name, patterns, attrs=()):
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
    """(patterns, enforced-by-default, source file) declared by pass `name`, and ((), True, None)
    for a name that is not a pass: a pass declaring nothing forbids nothing, which is the
    resting state."""
    mod = _pass_module(name)
    if mod is None:
        return (), True, None
    return (tuple(getattr(mod, "FORBIDS", ())),
            bool(getattr(mod, "FORBIDS_DEFAULT", True)),
            getattr(mod, "__file__", None))


def forbids_enabled(name, extra=None):
    """Whether pass `name`'s declaration is checked on this run."""
    import os
    _, on, _ = forbids(name)
    want = name
    settings = [(k, bool(v)) for k, v in (extra or {}).get("forbids", {}).items()]
    for item in os.environ.get(FORBID_ENV, "").split(","):
        k, _, v = item.strip().partition("=")
        if k:
            settings.append((k, v not in ("0", "false", "no")))
    for k, v in settings:
        if _pass_module(k) is None:
            raise ValueError(f"forbids names {k!r}, which is no pass")
        if k == want:
            on = v
    return on


def forbidden(name, module, extra=None):
    """[(op name, count, the first one, IN FULL)] `name` promised would not survive; empty when it
    did its job."""
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
