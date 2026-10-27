"""Small, dependency-light helpers shared across the MLIR passes."""
from ..passes.utils.ir_utils import walk_ops  # noqa: F401
from ..passes.utils.vcix import i64  # noqa: F401


def _ir():
    import mlir.ir as ir
    return ir


def i32(v):
    """`i32` IntegerAttr for `v` (uses the active MLIR context)."""
    ir = _ir()
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), int(v))


def i64_array(vals):
    """ArrayAttr of `i64` IntegerAttrs for `vals`."""
    ir = _ir()
    i = ir.IntegerType.get_signless(64)
    return ir.ArrayAttr.get([ir.IntegerAttr.get(i, int(v)) for v in vals])


def str_attr(v):
    """StringAttr of `str(v)`."""
    ir = _ir()
    return ir.StringAttr.get(str(v))


def _attrs(op):
    return getattr(op, "operation", op).attributes


def attr_int(op, key, default=None):
    """Integer value of `op`'s `key` attribute, or `default` if absent."""
    ir = _ir()
    a = _attrs(op)
    return ir.IntegerAttr(a[key]).value if key in a else default


def attr_i64_array(op, key, default=None):
    """`op`'s `key` integer array as a Python list, or `default` if absent (pass `default=[]` for
    the "missing -> empty" convention)."""
    a = _attrs(op)
    if key not in a:
        return default
    attr = a[key]
    return [int(attr[i]) for i in range(len(attr))]
