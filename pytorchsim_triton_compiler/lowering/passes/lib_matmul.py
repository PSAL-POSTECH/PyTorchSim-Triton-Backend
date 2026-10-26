"""lib_matmul: what a pytorchsim.dot is whatever unit it runs on -- its ops, its m, n and k, the
refusals any lowering of it makes, and KEPT_APART, which a lowering stamps on a loop beside a dot
it left a loop."""
import sys

from ... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from .lib_ir import DOT_OP, array_kind, op_of  # noqa: E402
from .lib_indexing_maps import MATMUL_K, MATMUL_M, MATMUL_N, matmul_axis  # noqa: E402

MATMUL_OPS = (DOT_OP,)

# The name a matmul's stream carries as its location, its line the matmul's number in program order:
# llc's regalloc remarks name the dot a loop of its stream spills in, and the retry narrows that dot
# (the stage's dot_lmul), apart from the loops legalize numbers (b08_legalize_vectors.LOOP_FILE).
DOT_FILE = "pytorchsim-dot"


class ArrayRefusal(Exception):
    """The array cannot take this matmul."""


def refuse_batched(op):
    """A batched dot: the loop nest a matmul lowering walks has no batch axis."""
    if array_kind(op) == "batch_matmul":
        raise ArrayRefusal(f"a batched pytorchsim.dot: this lowering walks n, k and m and has no batch "
                           f"loop, so it would compute the first batch only: {op}")


def refuse_accumulator_in_its_loop(op):
    innermost = None
    parent = op.operation.parent
    while parent is not None:
        if parent.name in ("affine.for", "scf.for") and innermost is None:
            innermost = parent
        parent = parent.parent

    if innermost is not None:
        owner = op.operands[2].owner
        owner = op_of(owner)
        anc = getattr(owner, "parent", None)
        while anc is not None:
            if anc == innermost:
                raise NotImplementedError(
                    "the matmul accumulator is allocated inside its own loop -- "
                    "that loop is not a reduction axis")
            anc = anc.parent


def get_mnk(op):
    mtA, mtB = ir.MemRefType(op.operands[0].type), ir.MemRefType(op.operands[1].type)
    return (mtA.shape[matmul_axis(op, 0, MATMUL_M)],
            mtB.shape[matmul_axis(op, 1, MATMUL_N)],
            mtA.shape[matmul_axis(op, 0, MATMUL_K)])


KEPT_APART = "pytorchsim.kept_apart"
