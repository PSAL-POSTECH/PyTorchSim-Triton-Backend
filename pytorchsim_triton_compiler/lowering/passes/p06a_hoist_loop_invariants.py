"""tritonpytorchsim-hoist-loop-invariants: what an affine loop computes the same on every trip is
computed once before it; a read only when nothing in the loop may write what it reads. The pass is
C++ (dialects/src/lib/HoistLoopInvariants.cpp), where mlir::AliasAnalysis answers that."""

from ...dialects import torchsim  # noqa: F401

NAME = "tritonpytorchsim-hoist-loop-invariants"


def run(module, **_opts):
    """The pass over every function."""
    from mlir.passmanager import PassManager
    PassManager.parse(f"builtin.module(func.func({NAME}))", module.context).run(module.operation)
    return module
