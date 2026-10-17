"""arith-int-range-narrowing: integer and index arithmetic whose operands and result the integer
range analysis proves fit a narrower integer width the function already computes vectors at is
computed at that width -- an index vector built from a loop's induction variable and vector.step
is no longer computed at the index width (XLEN) only to be cast down to the i32 the kernel compares
it with. Only widths some vector of the function already has are offered, so the pass brings in no
new element width (no new extend/truncate chain, no new vtype); a function with no narrower
integer vector is left as it is. The pass is upstream MLIR (ArithIntRangeOpts.cpp)."""

NAME = "arith-int-range-narrowing"


def _vector_int_widths(func):
    """The widths, below the index width (XLEN), of `func`'s integer vectors, i1 masks aside."""
    from mlir.ir import IntegerType, VectorType
    from ...contract.target import XLEN
    from .lib_ir import values
    got = set()
    for v in values(func.operation):
        if not isinstance(v.type, VectorType):
            continue
        ty = v.type.element_type
        if isinstance(ty, IntegerType) and 1 < ty.width < XLEN:
            got.add(ty.width)
    return sorted(got)


def run(module, **_opts):
    """The pass over every function, offered that function's own narrower integer widths."""
    from mlir.passmanager import PassManager
    from .lib_ir import funcs
    for func in funcs(module):
        widths = _vector_int_widths(func)
        if not widths:
            continue
        PassManager.parse(f"func.func({NAME}{{int-bitwidths-supported="
                          f"{','.join(map(str, widths))}}})",
                          module.context).run(func.operation)
    return module
