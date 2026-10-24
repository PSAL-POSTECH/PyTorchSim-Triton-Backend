"""tritonpytorchsim-lower-kernel-vcix-tpuv6e: a kernel's torchsim.vcix -> its encoding issued over
its tiles' rows on vcix_tpuv6e (libtpuv6e.so): a v.iv reads its rows directly, any other form reads
and writes them as the cross-lane unit's 32-bit pushes and pops do."""

from ....contract.vcix_form import FORMS
from ...passes.lib_bank_layout import bank_view, view_lanes
from ...passes.lib_ir import const_index, elem_bits, operand_segments, transfer_read, transfer_write
from ...passes.lib_ir import walk, zero_of
from ...passes.lib_vcix import i64 as a64, vcix
from ...passes.lower import LOWER_KERNEL_VCIX
from .lib_xlu import build_pop_rows, build_push_rows, build_row_loop, crossing_bits, row_at
from .lib_xlu import rows_per_vector

import mlir.ir as ir  # noqa: E402

NAME = "tritonpytorchsim-lower-kernel-vcix-tpuv6e"

REQUIRES = ("pytorchsim_tpu_family",)

SLOT = LOWER_KERNEL_VCIX.name

FORBIDS = LOWER_KERNEL_VCIX.lowers

(KERNEL_VCIX,) = FORBIDS

MARKERS = FORBIDS


def run(module, *, target, **_):
    """Every kernel's vcix lowered where it stands."""
    vectorlane, vlen = target.vector_lanes, target.vlen_bits
    for op in [o for o in walk(module.operation) if o.name == KERNEL_VCIX]:
        _lower_kernel_vcix(op, vectorlane, vlen)
    return module


def _lower_kernel_vcix(op, lanes, vlen):
    """`vcix.<form>` with the kernel's opcode and imm over its tiles' rows in order -- read from
    its operand, written into its dst, or both -- as a crossing reads and writes them."""
    from mlir.dialects import arith
    form = ir.StringAttr(op.attributes["form"]).value
    opcode, imm = (ir.IntegerAttr(op.attributes[k]).value for k in ("opcode", "imm"))
    n_in, n_dst = operand_segments(op)
    if FORMS.get(form) != (n_in, n_dst) or len(op.results):
        raise NotImplementedError(f"a kernel's vcix.{form} on buffers with {n_in} operand(s) and "
                                  f"{n_dst} dst: the form takes {FORMS.get(form)}: {op}")
    tiles = list(op.operands)
    elt = ir.MemRefType(tiles[0].type).element_type
    bits = elem_bits(elt)
    if form != "v.iv" and bits > 32:
        raise NotImplementedError(
            f"a kernel's vcix.{form} on {bits}-bit elements: the unit carries 32 bits at a time, "
            f"and a kernel's push and pop are separate instructions, so the halves cannot "
            f"alternate as a crossing's do: {op}")
    for t in tiles:
        if view_lanes(t) != lanes:
            raise NotImplementedError(
                f"a kernel's vcix.{form} on {t.type}: the tile does not hold a row in each of the "
                f"machine's {lanes} lanes, so the instruction reads or writes lanes it does not "
                f"cover: {op}")
    i64 = ir.IntegerType.get_signless(64)
    with ir.InsertionPoint(op):
        banks = [bank_view(t, lanes) for t in tiles]
        depth = ir.MemRefType(banks[0].type).shape[1]
        nr = rows_per_vector(1, depth, vlen // crossing_bits(elt))
        if depth % nr or any(ir.MemRefType(b.type).shape[1] != depth for b in banks):
            raise NotImplementedError(f"{depth} rows per lane are not a whole number of "
                                      f"{nr}-wide registers, or its tiles differ in depth: {op}")
        vec = ir.VectorType.get([nr], elt)
        if form == "v.iv":
            _build_viv_rows(banks[0], banks[1], depth, vec, opcode, imm, zero_of(elt))
        elif n_dst:
            rvl = arith.ConstantOp(i64, a64(nr)).result
            build_pop_rows(banks[0], 0, 1, depth, vec, rvl, 0, bits, 1, opcode, imm, None)
        else:
            rvl = arith.ConstantOp(i64, a64(nr)).result
            build_push_rows(banks[0], 0, 1, depth, vec, rvl, 0, bits, opcode, imm, zero_of(elt))
    op.erase()


def _build_viv_rows(src, dst, rows, vec, opcode, imm, pad):
    """`rows` rows of `src`, `vec` at a time, each through one vcix.v.iv into the same rows of
    `dst`."""
    nr = ir.VectorType(vec).shape[0]

    def one(i):
        at = row_at(0, i, nr)
        v = transfer_read(vec, src, [const_index(0), at], pad)
        got = vcix("vcix.v.iv", [v], [vec], {"opcode": a64(opcode), "imm": a64(imm)}).results[0]
        transfer_write(got, dst, [const_index(0), at])
    build_row_loop(rows // nr, one)
