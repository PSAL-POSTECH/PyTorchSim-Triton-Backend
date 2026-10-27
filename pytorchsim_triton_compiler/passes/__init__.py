"""What every pass shares: its MLIR name (pass_name), what it FORBIDS, and TargetPasses, where a pipeline
runs the target's own. The passes are one module each, by kind as MLIR's are: conversion/, transforms/,
analysis/ (the checks); utils/ what they share; transform.py and lower.py the order they run in; the
target's passes in targets/<target>/."""
import sys as _sys

from .. import toolchain as _toolchain

_toolchain.activate_bindings(_sys.path)

FORBID_ENV = "PYTORCHSIM_COMPILE_FORBIDS"

# The packages that hold passes, each module one pass: its name is pass_name's of the module's.
PASS_PACKAGES = ("conversion", "transforms", "analysis")


def pass_name(mod):
    """A pass's MLIR name, the one rule for every pass module, common, check or a target's: a module of
    conversion/ is `convert-<name>` (`conversion.tts_to_pytorchsim` is `convert-tts-to-pytorchsim`), one
    named convert_* keeps it, any other `pytorchsim-<name>` (`vectorize` is `pytorchsim-vectorize`)."""
    full = getattr(mod, "__name__", mod)
    name = full.rsplit(".", 1)[-1].replace("_", "-")
    if ".conversion." in f".{full}":
        return "convert-" + name
    return name if name.startswith("convert-") else "pytorchsim-" + name


def _pass_module(name):
    import importlib
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for package in PASS_PACKAGES:
        for f in sorted(os.listdir(os.path.join(here, package))):
            if f.endswith(".py") and f != "__init__.py" and pass_name(f"{package}.{f[:-3]}") == name:
                return importlib.import_module(f".{package}.{f[:-3]}", __package__)
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
    return forbidden_ops(patterns, module)


def forbidden_ops(patterns, module):
    """[(op name, count, the first one, IN FULL)] of the ops under `module` that `patterns` match."""
    from .utils.ir_utils import walk
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


class TargetPasses:
    """Where a pipeline runs the target's own passes (layer B): Target.<attr>'s, in the target's order,
    then a check that they left none of FORBIDS, the common ops every pass after them assumes gone."""

    def __init__(self, attr, forbids):
        self.attr, self.FORBIDS = attr, tuple(forbids)

    def passes(self):
        from ..targets import target
        return tuple(getattr(target(), self.attr))

    def __str__(self):
        return f"the target's {self.attr}"


def expand(passes):
    """`passes` with each TargetPasses replaced by the target's passes, followed by itself, the check."""
    out = []
    for p in passes:
        if isinstance(p, TargetPasses):
            out += list(p.passes()) + [p]
        else:
            out.append(p)
    return out


def names(passes):
    """The MLIR names of `passes`, a TargetPasses as the target's when one is known, else as itself."""
    out = []
    for p in passes:
        if not isinstance(p, TargetPasses):
            out.append(pass_name(p))
            continue
        from ..contract.target import TargetError
        try:
            out += [pass_name(q) for q in p.passes()]
        except TargetError:
            out.append(f"<{p}>")
    return out
