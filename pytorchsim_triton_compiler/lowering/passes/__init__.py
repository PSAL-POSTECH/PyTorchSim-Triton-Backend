"""pytorchsim_triton_compiler.lowering.passes -- every compiler pass, one file each (p00..p13), and
the two steps that run them in order: transform.py (p00..p07) and lower.py (p08..p13). What lives
here is what both share: a pass's MLIR-spelled name, and the ops it declares it removes (FORBIDS)."""
import sys as _sys

from ... import toolchain as _toolchain

_toolchain.activate_bindings(_sys.path)

FORBID_ENV = "TORCHSIM_COMPILE_FORBIDS"


def pass_name(mod):
    """A pass's name in MLIR's spelling -- `torchsim-vectorize` for the module `p05_vectorize`, or
    the NAME a module declares."""
    stem = getattr(mod, "__name__", mod).rsplit(".", 1)[-1]
    return getattr(mod, "NAME", None) or "torchsim-" + stem.split("_", 1)[1].replace("_", "-")


def _file_stem(name):
    for prefix in ("tritonpytorchsim-", "torchsim-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("-", "_")


def canon_pass_name(name):
    """One meaning per pass however it is spelled, so a settings key written the old way turns a
    check ON rather than silently off -- which is what a plain string compare did the day the names
    became MLIR-spelled."""
    return "torchsim-" + _file_stem(name).replace("_", "-")


def _pass_module(name):
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
    want = canon_pass_name(name)
    settings = [(k, bool(v)) for k, v in (extra or {}).get("forbids", {}).items()]
    for item in os.environ.get(FORBID_ENV, "").split(","):
        k, _, v = item.strip().partition("=")
        if k:
            settings.append((k, v not in ("0", "false", "no")))
    for k, v in settings:
        if _pass_module(k) is None:
            raise ValueError(f"forbids names {k!r}, which is no pass")
        if canon_pass_name(k) == want:
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
