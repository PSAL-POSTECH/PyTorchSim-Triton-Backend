"""convert-to-vcix: every unit op of the target written as the VCIX instruction it is, by the encoder the
target states for it in Target.TO_VCIX, where it stands."""
from ...targets import target
from ..utils.ir_utils import walk
from .. import pass_name

NAME = pass_name(__name__)


def __getattr__(name):
    """MARKERS: the unit ops the target encodes."""
    if name == "MARKERS":
        return tuple(sorted(target().TO_VCIX))
    raise AttributeError(name)


def run(module, **_):
    from mlir.ir import InsertionPoint
    encoders = target().TO_VCIX
    for op in [o for o in walk(module.operation) if o.name in encoders]:
        with InsertionPoint(op), op.location:
            encoders[op.name](op)
