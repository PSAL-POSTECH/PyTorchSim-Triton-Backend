"""A target is its pack: a pipeline list opens a Slot, and the pack of the target's accelerator family
(lowering/ext/<family>/, its PROVIDES) names the module that fills each one. A slot that lowers ops, or is
exactly_one, must be filled -- a pack missing one is refused when it is loaded, before any stage runs; a
slot that lowers nothing (a schedule) may stay empty. What the common passes read of a provider before it
runs is its slot's contract (CONTRACTS). resolve() is the list a target runs; the pipeline files stay the
one statement of the order."""
import importlib
import pkgutil
from typing import Protocol

from . import pass_name
from ...contract.target import ACCELERATORS, TargetError, is_accel

EXT_PACKAGE = __name__.rsplit(".", 2)[0] + ".ext"


class PassGateError(TargetError):
    """The target's pack cannot run the pipelines as declared."""


class Slot:
    """A place in a pipeline list that one pack module fills. One that `lowers` ops, or is exactly_one,
    is required: every pack fills it. Any other may stay empty."""

    def __init__(self, name, lowers=(), exactly_one=False):
        self.name = name
        self.lowers = tuple(lowers)
        self.exactly_one = exactly_one

    @property
    def required(self):
        return bool(self.lowers) or self.exactly_one

    def __repr__(self):
        return (f"Slot({self.name!r}" + (f", lowers={self.lowers!r}" if self.lowers else "")
                + (", exactly_one=True" if self.exactly_one else "") + ")")


class DmaProvider(Protocol):
    """lower-dma: beside run(), what p11, p12aa, the ELF wrapper and the TOG read of it."""
    DESCRIPTOR_DIMS: int   # the axes one descriptor walks; p11 cuts a transfer to this many
    INDEX_KEY: int         # the func7 of the instruction that names the program to the model

    def insn(self, func7, rs1, rs2) -> str: ...        # the DMA instruction as inline asm spells it
    def walk(self, op, vectorlane): ...               # (dims, DRAM strides, lead) a transfer's descriptor walks
    def sram_footprints(self, op, vectorlane, interval_of): ...  # what a transfer touches in each lane's bank


class VectorFunctionsProvider(Protocol):
    """lower-vector-functions: the math functions the unit computes as they are, math.<name>."""
    NATIVE_FUNCTIONS: frozenset


#: slot name -> the Protocol its provider satisfies beside SLOT, FORBIDS and run().
CONTRACTS = {
    "lower-dma": DmaProvider,
    "lower-vector-functions": VectorFunctionsProvider,
}


def members(proto):
    """The names a module satisfying `proto` has: its annotated attributes and its methods."""
    names = set(getattr(proto, "__annotations__", {}))
    names |= {n for n, v in vars(proto).items() if callable(v) and not n.startswith("_")}
    return sorted(names)


def check(pipelines):
    """The slots the pipeline lists open, name -> (list title, Slot): refuses a slot opened twice, and one
    opened before the machine is known (rule 7), since a pack is the machine's."""
    opened = {}
    for title, items, machine_known in pipelines:
        for item in items:
            if not isinstance(item, Slot):
                continue
            if item.name in opened:
                raise PassGateError(f"slot {item.name} is opened by both {opened[item.name][0]} and {title}")
            if not machine_known:
                raise PassGateError(f"slot {item.name} is opened in {title}, before the machine is known")
            opened[item.name] = (title, item)
    return opened


_OPENED = {}


def opened():
    """The slots the shipped pipelines open, checked once."""
    if not _OPENED:
        from . import lower, transform
        _OPENED.update(check(transform.pipelines() + lower.pipelines()))
    return _OPENED


def family_of(target):
    """The pack `target` runs: the family of the accelerator its isa names (contract.target.ACCELERATORS)."""
    accel = next((e for e in target.isa if is_accel(e)), None)
    if accel is None:
        raise PassGateError(f"{target.name}'s isa ({', '.join(target.isa)}) names no accelerator "
                            f"(pytorchsim_<name>), so no pack lowers for it")
    if accel not in ACCELERATORS:
        raise PassGateError(f"{target.name}'s accelerator {accel} belongs to no family: "
                            f"ACCELERATORS knows {', '.join(ACCELERATORS)}")
    return ACCELERATORS[accel]


def _module(pkg, ref):
    """The provider `ref` names in a pack's PROVIDES: a module, `.name` under the pack, or
    `<package>.<name>` under lowering/ext/ (a generic provider, rvv_generic's)."""
    if not isinstance(ref, str):
        return ref
    return importlib.import_module(pkg.__name__ + ref if ref.startswith(".") else f"{EXT_PACKAGE}.{ref}")


def load(family, openings=None):
    """The pack of `family`: slot name -> the module filling it, as its PROVIDES states, checked: every
    named slot is one the pipelines open (`openings`, the shipped ones unsaid), every module declares
    that SLOT and FORBIDS what the slot lowers and has what its contract names, and every required slot
    is filled."""
    openings = opened() if openings is None else openings
    pkg = importlib.import_module(f"{EXT_PACKAGE}.{family}")
    provides = getattr(pkg, "PROVIDES", None)
    if not isinstance(provides, dict):
        raise PassGateError(f"{pkg.__name__} declares no PROVIDES: the slots it fills, each with the "
                            f"module that fills it")
    pack = {}
    for name, ref in provides.items():
        if name not in openings:
            raise PassGateError(f"{pkg.__name__} fills slot {name}, which no pipeline opens; open: "
                                f"{', '.join(sorted(openings)) or 'none'}")
        mod = _module(pkg, ref)
        if getattr(mod, "SLOT", None) != name:
            raise PassGateError(f"{pkg.__name__} fills {name} with {pass_name(mod)}, whose SLOT is "
                                f"{getattr(mod, 'SLOT', None)!r}")
        _, slot = openings[name]
        left = [o for o in slot.lowers if o not in getattr(mod, "FORBIDS", ())]
        if left:
            raise PassGateError(f"{pass_name(mod)} fills {name}, which lowers {', '.join(left)}, and does "
                                f"not FORBID them")
        proto = CONTRACTS.get(name)
        missing = [m for m in members(proto) if not hasattr(mod, m)] if proto else []
        if missing:
            raise PassGateError(f"{pass_name(mod)} fills {name} without {', '.join(missing)}, which "
                                f"{proto.__name__} names")
        pack[name] = mod
    if not isinstance(getattr(pkg, "LAYOUT_RULES", None), str) or not pkg.LAYOUT_RULES:
        raise PassGateError(f"{pkg.__name__} declares no LAYOUT_RULES: the name of the rules the "
                            f"layout anchor asks for its dots' forms and the reduces its lanes fold")
    unfilled = sorted(n for n, (_, s) in openings.items() if s.required and n not in pack)
    if unfilled:
        raise PassGateError(f"{pkg.__name__} fills no {', '.join(unfilled)}: a slot that lowers ops or is "
                            f"exactly_one is filled by every pack, with its own pass or a generic one")
    return pack


_PACKS = {}


def pack(target):
    """`target`'s pack, loaded and checked once per family."""
    family = family_of(target)
    if family not in _PACKS:
        _PACKS[family] = load(family)
    return _PACKS[family]


def layout_rules(target):
    """The name of the LayoutRules `target`'s pack declares, which tritongpu-torchsim-anchor-layouts
    looks up in the extension (triton_extension/lib/LayoutRules/)."""
    pack(target)
    return importlib.import_module(f"{EXT_PACKAGE}.{family_of(target)}").LAYOUT_RULES


def packs():
    """family -> pack, for every package under lowering/ext/ that declares PROVIDES."""
    root = importlib.import_module(EXT_PACKAGE)
    for info in pkgutil.iter_modules(root.__path__):
        if not info.ispkg or not hasattr(importlib.import_module(f"{EXT_PACKAGE}.{info.name}"), "PROVIDES"):
            continue
        if info.name not in _PACKS:
            _PACKS[info.name] = load(info.name)
        yield info.name, _PACKS[info.name]


def providers():
    """Every module any shipped pack fills a slot with, in pass-name order."""
    seen = {id(m): m for _, p in packs() for m in p.values()}
    return sorted(seen.values(), key=pass_name)


def resolve(pipeline, target):
    """`pipeline` with each Slot replaced by the pass `target`'s pack fills it with, or left out when an
    optional slot is unfilled; every plain pass kept."""
    filled = pack(target)
    out = []
    for item in pipeline:
        if not isinstance(item, Slot):
            out.append(item)
        elif item.name in filled:
            out.append(filled[item.name])
    return out


def spelled(pipeline, target=None):
    """The pass names of `pipeline` as `target` runs it; without one, a slot as [name]."""
    if target is not None:
        return [pass_name(m) for m in resolve(pipeline, target)]
    return [f"[{i.name}]" if isinstance(i, Slot) else pass_name(i) for i in pipeline]
