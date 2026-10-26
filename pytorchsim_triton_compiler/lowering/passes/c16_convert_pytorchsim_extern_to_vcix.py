"""convert-pytorchsim-extern-to-vcix: a kernel's pytorchsim_extern.vcix on tiles written as the VCIX op its
form names, once per register of each lane's rows: its operand's rows read and its dst's rows written at
the register's width, with the opcode and imm the kernel stated and the fields it leaves free 0; an iv or
v.i takes the register's length as vl, a v.iv none. Nothing here knows what the instruction does or which
unit it names. One whose form and arity the VCIX dialect has no op for, or whose tiles a lane's register
rows do not cover, is refused."""
from ...contract.vcix_form import FORMS
from ...dialects.pytorchsim_extern import VcixOp
from . import pass_name

NAME = pass_name(__name__)

FORBIDS = (VcixOp.OPERATION_NAME,)

# Each form the VCIX dialect has an op for: (the field it leaves free, whether it takes vl).
_OPS = {"iv": ("rd", True), "v.i": ("rs2", True), "v.iv": (None, False)}


def run(module, *, target, **_):
    """Every kernel's vcix on tiles written where it stands."""
    from mlir.ir import MemRefType
    from .lib_ir import walk
    from .target_kind import target_kind
    view_op = target_kind(target).BANK_VIEW
    for op in [o for o in walk(module.operation) if isinstance(o.opview, VcixOp)]:
        if any(isinstance(v.type, MemRefType) for v in op.operands):
            _convert(op, target.vector_lanes, target.vlen_bits, view_op)
    return module


def _convert(op, lanes, vlen, view_op):
    """`vcix.<form>` over each lane's rows of `op`'s tiles, `vlen` bits at a time, in place of `op`."""
    import mlir.ir as ir
    from .lib_bank_layout import bank_view, view_lanes
    from .lib_ir import build_trip_loop, elem_bits, operand_segments
    form = ir.StringAttr(op.attributes["form"]).value
    opcode, imm = (ir.IntegerAttr(op.attributes[k]).value for k in ("opcode", "imm"))
    n_in, n_dst = operand_segments(op)
    if form not in _OPS or FORMS.get(form) != (n_in, n_dst) or len(op.results):
        raise NotImplementedError(f"{NAME}: a kernel's vcix.{form} on buffers with {n_in} operand(s) and "
                                  f"{n_dst} dst: the VCIX dialect has no op of that form and arity: {op}")
    tiles = list(op.operands)
    for t in tiles:
        if view_lanes(t) != lanes:
            raise NotImplementedError(
                f"{NAME}: a kernel's vcix.{form} on {t.type}: the tile does not hold a row in each of the "
                f"machine's {lanes} lanes, so the instruction reads or writes lanes it does not cover: {op}")
    elt = ir.MemRefType(tiles[0].type).element_type
    with ir.InsertionPoint(op), op.location:
        banks = [bank_view(t, lanes, view_op) for t in tiles]
        depth = ir.MemRefType(banks[0].type).shape[1]
        nr = min(vlen // elem_bits(elt), depth)
        if depth % nr or any(ir.MemRefType(b.type).shape[1] != depth for b in banks):
            raise NotImplementedError(f"{NAME}: {depth} rows per lane are not a whole number of "
                                      f"{nr}-wide registers, or its tiles differ in depth: {op}")
        vec = ir.VectorType.get([nr], elt)
        if depth == nr:
            _one(None, banks, n_in, vec, form, opcode, imm)
        else:
            iv, inside = build_trip_loop(ir.InsertionPoint.current, depth // nr)
            with inside:
                _one(iv, banks, n_in, vec, form, opcode, imm)
    op.erase()


def _one(iv, banks, n_in, vec, form, opcode, imm):
    """One register of rows at trip `iv` (None: the only one): read, the VCIX op, written."""
    import mlir.ir as ir
    from mlir.dialects import arith
    from .lib_ir import const_index, transfer_read, transfer_write, zero_of
    from .lib_vcix import i64, vcix
    nr = ir.VectorType(vec).shape[0]
    at = const_index(0) if iv is None else arith.MulIOp(iv, const_index(nr)).result
    elt = ir.VectorType(vec).element_type
    operands = [transfer_read(vec, b, [const_index(0), at], zero_of(elt)) for b in banks[:n_in]]
    free, sized = _OPS[form]
    attrs = {"opcode": i64(opcode), "imm": i64(imm)}
    if free:
        attrs[free] = i64(0)
    if sized:
        operands.append(arith.ConstantOp(ir.IntegerType.get_signless(64), i64(nr)).result)
    dsts = banks[n_in:]
    got = vcix(f"vcix.{form}", operands, [vec] * len(dsts), attrs)
    for r, b in zip(got.results, dsts):
        transfer_write(r, b, [const_index(0), at])
