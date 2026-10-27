"""The vector widths a build was lowered at -- the kernel LMUL, the per-loop LMULs legalize took and the
per-stream LMULs the matmul lowering took, as the spill retry decided them -- stamped on the module, so
every binary built from one decision can be shown to be built from that one."""
import json

WIDTHS = "pytorchsim.widths"
"""The module attribute: {"lmul", "loop_lmul": {loop: LMUL}, "dot_lmul": {dot: LMUL}}, as JSON."""


class WidthsDiffer(RuntimeError):
    """Two binaries of one kernel would be built at different widths."""


def decision(lmul, loop_lmul=None, dot_lmul=None):
    """The widths as stamped: every key a string, so a stamp read back compares equal."""
    return {"lmul": int(lmul), "loop_lmul": {str(k): int(v) for k, v in sorted((loop_lmul or {}).items())},
            "dot_lmul": {str(k): int(v) for k, v in sorted((dot_lmul or {}).items())}}


def stamp(module, **fields):
    """`fields` of the decision merged into `module`'s stamp."""
    import mlir.ir as ir
    attrs = module.operation.attributes
    got = json.loads(ir.StringAttr(attrs[WIDTHS]).value) if WIDTHS in attrs else {}
    got.update(fields)
    attrs[WIDTHS] = ir.StringAttr.get(json.dumps(got, sort_keys=True))


def stamp_file(path, **fields):
    """stamp() on the module at `path`, written back."""
    from mlir.ir import Context, Location, Module
    from .utils.ir_utils import module_facts
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        with open(path) as f:
            module = Module.parse(f.read())
        stamp(module, **fields)
        with open(path, "w") as f:
            f.write(module.operation.get_asm(enable_debug_info=True))
        module_facts(path, module)


def read(path):
    """The widths the module at `path` states, or None."""
    from .utils.ir_utils import module_facts
    attrs = module_facts(path)[0]
    return json.loads(attrs[WIDTHS]) if WIDTHS in attrs else None


def refuse_unless_same(stated):
    """`stated`, {what: widths}, all one decision: WidthsDiffer naming the first two that differ, or
    one that states none."""
    unstated = [w for w, got in stated.items() if got is None]
    if unstated:
        raise WidthsDiffer(f"{unstated[0]} states no {WIDTHS}: it was lowered before the widths were "
                           f"stamped; build it again from transform")
    (first, want), *rest = stated.items()
    for what, got in rest:
        if got != want:
            raise WidthsDiffer(f"{what} is built at widths {got} and {first} at {want}: one kernel's "
                               f"binaries must come from one spill decision")
    return want
