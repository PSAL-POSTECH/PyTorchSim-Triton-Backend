"""tritonpytorchsim-read-broadcast-row-once: a buffer every writer of which is a transfer with DRAM
stride 0 along an axis holds the same value at every index of that axis, so an affine read of it
reads index 0 there; reads that differed only in that index become one value."""
from mlir.ir import AffineConstantExpr, AffineMap, AffineMapAttr, ArrayAttr, FlatSymbolRefAttr, IntegerAttr

from .lib_ir import producer, walk

NAME = "tritonpytorchsim-read-broadcast-row-once"
TRANSFER = "torchsim.transfer"
GET_GLOBAL = "memref.get_global"
AFFINE_READS = ("affine.vector_load", "affine.load")
READS = AFFINE_READS + ("memref.load",)


def _aliases(module, buf):
    """Every value naming the storage `buf` names: itself, or each get_global of its symbol."""
    op = producer(buf)
    if op is None or op.name != GET_GLOBAL:
        return [buf]
    sym = FlatSymbolRefAttr(op.attributes["name"]).value
    return [o.results[0] for o in walk(module.operation)
            if o.name == GET_GLOBAL and FlatSymbolRefAttr(o.attributes["name"]).value == sym]


def _equal_axes(values):
    """The axes along which every writer of the storage writes one value, or an empty set."""
    axes = None
    for v in values:
        for use in v.uses:
            op = use.owner
            if op.name in READS and use.operand_number == 0:
                continue
            if op.name != TRANSFER or use.operand_number != 2:
                return set()
            stride = [IntegerAttr(a).value for a in ArrayAttr(op.attributes["dram_stride"])]
            zero = {d for d, s in enumerate(stride) if s == 0}
            axes = zero if axes is None else axes & zero
    return axes or set()


def _is_zero(expr):
    return isinstance(expr, AffineConstantExpr) and expr.value == 0


def _read_index_zero(read, axes):
    """Rewrite `read`'s map so each axis in `axes` reads index 0."""
    m = AffineMapAttr(read.attributes["map"]).value
    exprs = list(m.results)
    if all(_is_zero(exprs[d]) for d in axes):
        return
    for d in axes:
        exprs[d] = AffineConstantExpr.get(0)
    read.attributes["map"] = AffineMapAttr.get(AffineMap.get(m.n_dims, m.n_symbols, exprs))


def run(module, **_opts):
    """Every storage written only by stride-0 transfers: its affine reads take index 0 on those axes."""
    seen = []
    for op in walk(module.operation):
        if op.name != TRANSFER or any(op.operands[2] == s for s in seen):
            continue
        values = _aliases(module, op.operands[2])
        seen.extend(values)
        axes = _equal_axes(values)
        if not axes:
            continue
        for v in values:
            for use in list(v.uses):
                if use.owner.name in AFFINE_READS and use.operand_number == 0:
                    _read_index_zero(use.owner, axes)
    return module
