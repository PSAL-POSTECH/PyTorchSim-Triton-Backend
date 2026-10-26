"""convert-to-vcix: every unit op of the target kind written as the VCIX instruction it is, by the encoder the
kind states for it in TargetKind.TO_VCIX, where it stands."""
from .target_kind import target_kind
from .lib_ir import walk
from . import pass_name

NAME = pass_name(__name__)


def __getattr__(name):
    """MARKERS: the unit ops the target kind encodes."""
    if name == "MARKERS":
        return tuple(sorted(target_kind().TO_VCIX))
    raise AttributeError(name)


def run(module, **_):
    from mlir.ir import InsertionPoint
    encoders = target_kind().TO_VCIX
    for op in [o for o in walk(module.operation) if o.name in encoders]:
        with InsertionPoint(op), op.location:
            encoders[op.name](op)
