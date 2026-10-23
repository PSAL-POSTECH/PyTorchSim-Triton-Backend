"""Every affine.for bound map lib_ir.loop_bounds reads, and the TOG builder's refusal of a loop holding a
DMA whose end it cannot read: a bound is a constant, or a constant dim/symbol operand, or None.
    python3 tests/runtime/loop_bounds_cases.py        exit 0 iff every case gets what it names"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pytorchsim_triton_compiler  # noqa: E402,F401
from pytorchsim_triton_compiler.dialects import torchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import loop_bounds, trip_count  # noqa: E402
from pytorchsim_triton_compiler.lowering.tog.passes.build_tog import (  # noqa: E402
    TogBuilder, _rewrite_loop_steps, ir)


def _loop(check, lb, ub, lb_ops, ub_ops, body=None):
    """`check(loop)` of one affine.for whose bounds are ints or `(dims, symbols, results)` maps,
    with operands the given constants or, for "arg", the function's index argument."""
    from mlir.dialects import affine, arith, func
    from mlir.ir import (AffineConstantExpr, AffineDimExpr, AffineMap, AffineSymbolExpr,
                         Context, F32Type, IndexType, InsertionPoint, IntegerAttr, Location,
                         MemRefType, Module)
    with Context(), Location.unknown():
        m = Module.create()
        f32, idx = F32Type.get(), IndexType.get()
        dram = MemRefType.get([64], f32)
        sram = MemRefType.get([8, 8], f32, memory_space=IntegerAttr.get(idx, 1))
        with InsertionPoint(m.body):
            fn = func.FuncOp("kernel", ([dram, sram, idx], []))
        blk = fn.add_entry_block()
        args = list(blk.arguments)

        def val(v):
            return args[2] if v == "arg" else arith.ConstantOp(idx, IntegerAttr.get(idx, v)).result

        def bound(b):
            if isinstance(b, int):
                return b
            dims, syms, results = b
            exprs = (AffineDimExpr.get, AffineSymbolExpr.get, AffineConstantExpr.get)
            return AffineMap.get(dims, syms, [r(*exprs) for r in results])

        with InsertionPoint(blk):
            loop = affine.AffineForOp(bound(lb), bound(ub), 1,
                                      lower_bound_operands=[val(v) for v in lb_ops],
                                      upper_bound_operands=[val(v) for v in ub_ops])
            with InsertionPoint(loop.body):
                if body:
                    body(args, val)
                affine.AffineYieldOp([])
            func.ReturnOp([])
        m.operation.verify()
        return check(loop)


def _bounds(*spec):
    return _loop(lambda loop: (loop_bounds(loop), trip_count(loop)), *spec)


def _traced(*spec):
    def dma(args, val):
        torchsim.TransferOp(None, args[0], val(0), args[1], val(0), val(0), [], [], "MVIN", 0,
                            [8, 1], async_=True)

    def trace(loop):
        b = TogBuilder()
        b.visit_operation(loop, None)
        _rewrite_loop_steps(b)
        once = b.loop_nodes[0].op
        once.operation.verify()
        return (len(b.loop_nodes), str(ir.AffineMapAttr(once.operation.attributes["upperBoundMap"])))
    return _loop(trace, *spec, body=dma)


def D(i):
    return lambda d, s, c: d(i)


def S(i):
    return lambda d, s, c: s(i)


def C(n):
    return lambda d, s, c: c(n)


CASES = {
    "constant bounds": (lambda: _bounds(0, 8, [], []), ((0, 8, 1), 8)),
    "dim operand": (lambda: _bounds((1, 0, [D(0)]), 10, [2], []), ((2, 10, 1), 8)),
    "symbol operand": (lambda: _bounds(0, (0, 1, [S(0)]), [], [8]), ((0, 8, 1), 8)),
    "second dim operand": (lambda: _bounds((2, 0, [D(1)]), 10, [1, 3], []), ((3, 10, 1), 7)),
    "offset map": (lambda: _bounds(0, (1, 0, [lambda d, s, c: d(0) + c(4)]), [], [4]),
                   ((0, None, 1), None)),
    "min map": (lambda: _bounds(0, (1, 0, [D(0), C(6)]), [], [8]), ((0, None, 1), None)),
    "end not a constant": (lambda: _traced(0, (1, 0, [D(0)]), [], ["arg"]),
                           (1, "affine_map<() -> (1)>")),
}


def main():
    failed = 0
    for name, (case, want) in CASES.items():
        try:
            got = case()
        except Exception as e:
            got = f"{type(e).__name__}: {e}"
        ok = (want in got) if isinstance(want, str) and isinstance(got, str) else got == want
        failed += not ok
        print(f"{name:20} {'PASS' if ok else 'FAIL'}  got {str(got)[:110]}  want {want}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
