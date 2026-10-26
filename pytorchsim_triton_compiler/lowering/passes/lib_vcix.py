"""lib_vcix: the VCIX dialect ops (Xsfvcp, any model) that the passes under lowering/ext/ mint, and
the word vectors they travel in; what an opcode means is the model's, stated in its own package."""
import sys

from ... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from ...contract.vcix_form import FORMS  # noqa: E402
from .lib_ir import elem_bits  # noqa: E402


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

