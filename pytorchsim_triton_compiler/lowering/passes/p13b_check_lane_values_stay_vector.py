"""tritonpytorchsim-check-lane-values-stay-vector: no op TO_LLVM leaves takes a vector and gives an
integer or float scalar. A lane's value lives in its vector register; the one scalar register file
every lane shares would keep lane 0's alone. Read off the types, whatever the op is called."""

from .lib_ir import walk

NAME = "tritonpytorchsim-check-lane-values-stay-vector"


class LaneValueMadeScalar(Exception):
    """TO_LLVM left an op that makes a lane's own value a scalar, which the one scalar register
    file all lanes share holds for lane 0 only."""


def run(module, **_):
    """Refuse `module` if an op of it makes a vector's element a scalar."""
    made = ops_making_a_vector_scalar(module)
    if made:
        raise LaneValueMadeScalar(
            f"TO_LLVM left {len(made)} op(s) that make a lane's value a scalar, which every "
            f"lane's one shared scalar register file holds for lane 0 only. First: "
            f"{' '.join(str(made[0]).split())[:200]} at {made[0].location}")
    return module


def ops_making_a_vector_scalar(module):
    """Every op of `module` with a vector operand and an integer or float scalar result."""
    from mlir.ir import FloatType, IntegerType, VectorType
    return [op for op in walk(module.operation)
            if any(isinstance(x.type, VectorType) for x in op.operands)
            and any(isinstance(r.type, (IntegerType, FloatType)) for r in op.results)]
