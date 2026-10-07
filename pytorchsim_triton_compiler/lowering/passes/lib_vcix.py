"""lib_vcix: the VCIX dialect ops three emitters mint -- lower_matmul's array sequence,
lower_to_vcix's compute, lower_to_xlu's pushes and pops -- and the word vectors they travel in."""
import sys

from ... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from .lib_ir import elem_bits  # noqa: E402


def i64(v):
    """`i64` IntegerAttr for `v`."""
    import mlir.ir as ir
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), int(v))


VCIX_OPS = ("vcix.iv", "vcix.i", "vcix.v.i", "vcix.v.iv")


def unregistered_ops(ctx, names=VCIX_OPS):
    """Which of `names` this LLVM's MLIR does not define -- empty is correct.
    A stock build answers with ALL FOUR, v.iv included: upstream registers the
    dialect for TRANSLATION only, so the bindings can mint none of them."""
    return [n for n in names if not ctx.is_registered_operation(n)]


def vcix(name, operands, result_tys, attrs):
    """Emit one VCIX operation as a `vcix.*` DIALECT op, operands in ODS order.
    THE DIALECT OWNS THE ENCODING -- the intrinsic, its argument order and its
    xlen come from the dialect's own translation, so no name is spelt here."""
    ops = list(operands)
    missing = unregistered_ops(ops[0].context, (name,))
    if missing:
        raise RuntimeError(
            f"{name} is not an op in this LLVM's MLIR. TORCHSIM_COMPILE_LLVM_PATH must name "
            f"the LLVM pinned in setup/versions.env, whose branch carries them, which is "
            f"where the VCIX side-effecting ops are -- see toolchain.LLVM_BIN.")
    return ir.Operation.create(name, results=list(result_tys), operands=ops,
                               attributes=dict(attrs))


def word_vector(vec_ty):
    """The vector type a VCIX push/pop TRAVELS in: same shape and
    WIDTH, integer element. THE INTRINSICS ARE TYPED ON THE REGISTER, NOT THE
    DATA -- a push hands raw bits over at the current SEW, so it is bitcast."""
    vt = ir.VectorType(vec_ty)
    return ir.VectorType.get(list(vt.shape),
                             ir.IntegerType.get_signless(elem_bits(vt.element_type)))


def bitcast(value, ty):
    """arith.bitcast `value` to `ty`. ALWAYS a real cast, never a no-op:
    `ty` is the same-width INTEGER vector and the value is always the float one,
    because lower_matmul refused an integer matmul above all three call sites."""
    from mlir.dialects import arith
    return arith.BitcastOp(ty, value).result


def lane_number(rvl, width):
    """Each lane's own number, `width` elements of it: the VCIX op of opcode 0."""
    i64_ty = ir.IntegerType.get_signless(64)
    return vcix("vcix.v.i", [rvl], [ir.VectorType.get([width], i64_ty)],
                {"opcode": i64(0), "rs2": i64(0), "imm": i64(0)}).results[0]
