"""Pass gating by the target's extensions: a pipeline list opens a Slot, a module under lowering/ext/
that declares SLOT fills it, and REQUIRES (names all in the target's features) says on which machines.
resolve() is the list a target runs; the pipeline files stay the one statement of the order."""
import importlib
import pkgutil

from . import pass_name
from ...contract.target import features

EXT_PACKAGE = __name__.rsplit(".", 2)[0] + ".ext"


class PassGateError(RuntimeError):
    pass


class Slot:
    """A place in a pipeline list that the passes declaring SLOT = name fill. One that `lowers` ops
    takes at most one pass and may stay empty, refusing in its place a module that holds one of
    them; exactly_one when the target must have one and only one."""

    def __init__(self, name, lowers=(), exactly_one=False):
        self.name = name
        self.lowers = tuple(lowers)
        self.exactly_one = exactly_one

    def __repr__(self):
        return (f"Slot({self.name!r}" + (f", lowers={self.lowers!r}" if self.lowers else "")
                + (", exactly_one=True" if self.exactly_one else "") + ")")


class Unfilled:
    """A slot that lowers ops and has no pass on this target, where it stands: it runs nothing, and
    refuses a module holding one of those ops, naming the slot and the op."""

    def __init__(self, slot, target, offered=()):
        self.slot, self.target, self.offered = slot, target, offered
        self.__name__ = self.NAME = f"[{slot.name}: no pass]"
        self.MARKERS = slot.lowers

    def run(self, module, **_):
        from .lib_ir import walk
        held = sorted({op.name for op in walk(module.operation)} & set(self.slot.lowers))
        if held:
            said = "; ".join(f"{pass_name(m)} REQUIRES {', '.join(requires(m))}"
                             for m in self.offered) or "no pass declares it"
            raise PassGateError(f"the module holds {', '.join(held)}, which slot {self.slot.name} "
                                f"lowers, and no pass fills that slot on {self.target.name}, whose "
                                f"isa is {', '.join(self.target.isa)}: {said}")
        return module


def requires(mod):
    """The extensions a pass needs, () for one every target runs."""
    return tuple(getattr(mod, "REQUIRES", ()))


def active(mod, target):
    """Whether `target`'s features hold every name `mod` REQUIRES."""
    return set(requires(mod)) <= features(target)


_FOUND = {}


def providers(package=EXT_PACKAGE):
    """Every module under `package` that declares a SLOT, in pass-name order, imported once."""
    if package not in _FOUND:
        root = importlib.import_module(package)
        found = []
        for info in pkgutil.walk_packages(root.__path__, package + "."):
            mod = importlib.import_module(info.name)
            if hasattr(mod, "SLOT"):
                found.append(mod)
        _FOUND[package] = sorted(found, key=pass_name)
    return _FOUND[package]


def check(pipelines, found):
    """Refuse what no target could run as declared: a provider of a slot no list opens, a slot
    opened twice, REQUIRES in a list that runs before the machine is known (rule 7), and a
    provider that does not FORBID every op its slot lowers."""
    opened = {}
    for title, items, machine_known in pipelines:
        for item in items:
            if isinstance(item, Slot):
                if item.name in opened:
                    raise PassGateError(f"slot {item.name} is opened by both {opened[item.name][0]} "
                                        f"and {title}")
                opened[item.name] = (title, machine_known, item)
            elif requires(item) and not machine_known:
                raise PassGateError(f"{pass_name(item)} REQUIRES {', '.join(requires(item))} but "
                                    f"runs in {title}, before the machine is known")
    for mod in found:
        if mod.SLOT not in opened:
            raise PassGateError(f"{pass_name(mod)} fills slot {mod.SLOT}, which no pipeline "
                                f"opens; open: {', '.join(sorted(opened)) or 'none'}")
        title, machine_known, slot = opened[mod.SLOT]
        if requires(mod) and not machine_known:
            raise PassGateError(f"{pass_name(mod)} REQUIRES {', '.join(requires(mod))} and fills "
                                f"{mod.SLOT} in {title}, which runs before the machine is known")
        left = [o for o in slot.lowers if o not in getattr(mod, "FORBIDS", ())]
        if left:
            raise PassGateError(f"{pass_name(mod)} fills {mod.SLOT}, which lowers "
                                f"{', '.join(left)}, and does not FORBID them")


def _pipelines():
    from . import lower, transform
    return transform.pipelines() + lower.pipelines()


_CHECKED = []


def registered():
    """The providers under lowering/ext/, checked once against every pipeline list."""
    found = providers()
    if not _CHECKED:
        check(_pipelines(), found)
        _CHECKED.append(True)
    return found


def resolve(pipeline, target, found=None):
    """`pipeline` with each Slot replaced by the providers whose REQUIRES `target` meets, in
    pass-name order, or by Unfilled when a slot that lowers ops gets none; each plain pass kept
    when it does. Refuses an exactly_one slot that gets none, and several in a slot taking one."""
    found = registered() if found is None else found
    out = []
    for item in pipeline:
        if not isinstance(item, Slot):
            if active(item, target):
                out.append(item)
            continue
        offered = [m for m in found if m.SLOT == item.name]
        fill = [m for m in offered if active(m, target)]
        if (item.exactly_one or item.lowers) and len(fill) > 1:
            raise PassGateError(f"slot {item.name} takes exactly one pass, and {target.name} "
                                f"meets the REQUIRES of {', '.join(pass_name(m) for m in fill)}")
        if item.exactly_one and not fill:
            said = "; ".join(f"{pass_name(m)} REQUIRES {', '.join(requires(m))}" for m in offered)
            raise PassGateError(f"no pass fills slot {item.name} on {target.name}, whose isa is "
                                f"{', '.join(target.isa)}: " + (said or "no pass declares it"))
        out.extend(fill or ([Unfilled(item, target, offered)] if item.lowers else []))
    return out


def spelled(pipeline, target=None):
    """The pass names of `pipeline` as `target` runs it; without one, a slot as [name]."""
    if target is not None:
        return [pass_name(m) for m in resolve(pipeline, target)]
    return [f"[{i.name}]" if isinstance(i, Slot) else pass_name(i) for i in pipeline]
