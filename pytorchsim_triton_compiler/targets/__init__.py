"""Every target the compiler lowers for, one directory each, laid out as Triton's third_party/<backend>/:
targets/<target>/ its Python (SUBTARGETS in __init__.py, its Target subclass in compiler.py),
include/ and lib/ its dialect, CMakeLists.txt its build. This package names no target; it finds them."""
import importlib
import inspect
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGETS = {}


def names():
    """Every target: each directory here that is a package and states its build (CMakeLists.txt)."""
    return tuple(sorted(n for n in os.listdir(_HERE)
                        if os.path.isfile(os.path.join(_HERE, n, "__init__.py"))
                        and os.path.isfile(os.path.join(_HERE, n, "CMakeLists.txt"))))


def subtargets():
    """{accelerator: target}: the accelerators each target's __init__.py states in SUBTARGETS."""
    return {accel: name for name in names()
            for accel in importlib.import_module(f"{__name__}.{name}").SUBTARGETS}


def target_name(subtarget):
    """The target that lowers for `subtarget` (Subtarget.target: the machine file's `target`, else the one
    whose SUBTARGETS holds its accelerator), refused when there is none."""
    from ..contract.target import is_accel
    from .compiler import PassGateError
    accel = next((e for e in subtarget.isa if is_accel(e)), None)
    if accel is None:
        raise PassGateError(f"{subtarget.name}'s isa ({', '.join(subtarget.isa)}) names no accelerator "
                            f"(pytorchsim_<name>), so no target lowers for it")
    if subtarget.target is None:
        raise PassGateError(f"{subtarget.name}'s accelerator {accel} states no target and no target lists it "
                            f"in SUBTARGETS: they list {', '.join(subtargets())}")
    if subtarget.target not in names():
        raise PassGateError(f"{subtarget.name}'s target {subtarget.target!r} is none the compiler lowers: "
                            f"targets {', '.join(names())}")
    return subtarget.target


def _concrete(module, base):
    """The one concrete subclass of `base` `module` defines (Triton's _find_concrete_subclasses)."""
    from .compiler import PassGateError
    found = [c for c in vars(module).values()
             if inspect.isclass(c) and issubclass(c, base) and not inspect.isabstract(c)]
    if len(found) != 1:
        raise PassGateError(f"{module.__name__} defines {len(found)} concrete {base.__name__} subclasses, "
                            f"not one: {found}")
    return found[0]


def target(subtarget=None):
    """`subtarget`'s target (the machine's, unsaid): its compiler.py's Target subclass, made once."""
    if subtarget is None:
        from ..driver import machine
        subtarget = machine.subtarget()
    name = target_name(subtarget)
    if name not in _TARGETS:
        from .compiler import Target
        _TARGETS[name] = _concrete(importlib.import_module(f"{__name__}.{name}.compiler"), Target)()
    return _TARGETS[name]
