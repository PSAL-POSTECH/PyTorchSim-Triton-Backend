"""vcix: the VCIX dialect ops (Xsfvcp, any model) that the passes under targets/ mint, and
the word vectors they travel in; what an opcode means is the model's, stated in its own package."""
import sys

from ... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from ...contract.vcix_form import FORMS  # noqa: E402
from .ir_utils import elem_bits  # noqa: E402


def i64(v):
    """`i64` IntegerAttr for `v`."""
    import mlir.ir as ir
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), int(v))


VCIX_OPS = tuple("vcix." + form for form in FORMS)


def unregistered_ops(ctx, names=VCIX_OPS):
    """Which of `names` this LLVM's MLIR does not define -- empty is correct."""
    return [n for n in names if not ctx.is_registered_operation(n)]


def vcix(name, operands, result_tys, attrs):
    """Emit one VCIX operation as a `vcix.*` DIALECT op, operands in ODS order."""
    ops = list(operands)
    missing = unregistered_ops(ops[0].context, (name,))
    if missing:
        raise RuntimeError(
            f"{name} is not an op in this LLVM's MLIR. PYTORCHSIM_COMPILE_LLVM_PATH must name "
            f"the LLVM pinned in setup/versions.env, whose branch carries them, which is "
            f"where the VCIX side-effecting ops are -- see toolchain.LLVM_BIN.")
    return ir.Operation.create(name, results=list(result_tys), operands=ops,
                               attributes=dict(attrs))


def word_vector(vec_ty):
    """The vector type a VCIX push/pop TRAVELS in: same shape and WIDTH, integer element."""
    vt = ir.VectorType(vec_ty)
    return ir.VectorType.get(list(vt.shape),
                             ir.IntegerType.get_signless(elem_bits(vt.element_type)))


def bitcast(value, ty):
    """arith.bitcast `value` to `ty`."""
    from mlir.dialects import arith
    return arith.BitcastOp(ty, value).result


def unit_vector(h):
    """The elements a vcix op's vector holds for `h` of them: a power of two (llc widens no other
    width of a vcix op's operand or result: "Do not know how to widen this operator's operand!"),
    its vl operand saying `h`."""
    return 1 << (h - 1).bit_length()


def to_unit(v, h):
    """`v`, h words, as the unit's vector: the words past h are never read (vl is h)."""
    from mlir.ir import DenseI64ArrayAttr, Operation, VectorType
    w = unit_vector(h)
    if w == h:
        return v
    ty = VectorType.get([w], VectorType(v.type).element_type)
    return Operation.create("vector.shuffle", results=[ty], operands=[v, v],
                            attributes={"mask": DenseI64ArrayAttr.get(list(range(h)) + [0] * (w - h))}).results[0]


def from_unit(v, h):
    """The first h elements of the unit's vector `v`."""
    from mlir.dialects import vector
    from mlir.ir import VectorType
    if unit_vector(h) == h:
        return v
    return vector.ExtractStridedSliceOp(VectorType.get([h], VectorType(v.type).element_type), v,
                                        [0], [h], [1]).result
