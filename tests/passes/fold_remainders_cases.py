"""pytorchsim-simplify-index-arithmetic's remainders drop the terms their divisor divides only where that
keeps the value, after its nsw marking as in the pipeline.
    python3 tests/passes/fold_remainders_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from mlir.dialects import arith  # noqa: E402
from pytorchsim_triton_compiler.passes.transforms import simplify_index_arithmetic
from pytorchsim_triton_compiler.passes.utils.ir_utils import producer, walk  # noqa: E402



def _rem(m, cls):
    return next(o for o in walk(m.operation) if isinstance(o.opview, cls))


def _fold(m):
    simplify_index_arithmetic.run(m, arg_numels={})


def a_carried_value_is_not_nonnegative_by_the_loop_bound():
    """`(a + 8) srem 8` with `a` carried down from 0 by 4 a lap: a = -4 gives 4, `a srem 8` -4.
    Breaks if an affine.for's iter_args are taken as non-negative by its lower bound."""
    m = ir.Module.parse("""
func.func @f(%init: i32, %o: memref<4xi32>, %gx: i32, %gy: i32, %gz: i32, %px: i32, %py: i32, %pz: i32) {
  %c8 = arith.constant 8 : i32
  %c4 = arith.constant 4 : i32
  %r = affine.for %i = 0 to 4 iter_args(%a = %init) -> i32 {
    %s = arith.addi %a, %c8 : i32
    %m = arith.remsi %s, %c8 : i32
    affine.store %m, %o[%i] : memref<4xi32>
    %n = arith.subi %a, %c4 : i32
    affine.yield %n : i32
  }
  return
}""")
    _fold(m)
    lhs = producer(_rem(m, arith.RemSIOp).operands[0])
    assert lhs is not None and isinstance(lhs.opview, arith.AddIOp), lhs


def the_induction_variable_still_folds():
    """`(i + 8) srem 8` with i from 0 is `i srem 8`. Breaks if the iter_args fix also stops the IV."""
    m = ir.Module.parse("""
func.func @f(%o: memref<4xindex>, %gx: i32, %gy: i32, %gz: i32, %px: i32, %py: i32, %pz: i32) {
  %c8 = arith.constant 8 : index
  affine.for %i = 0 to 4 {
    %s = arith.addi %i, %c8 : index
    %m = arith.remsi %s, %c8 : index
    affine.store %m, %o[%i] : memref<4xindex>
  }
  return
}""")
    _fold(m)
    rem = _rem(m, arith.RemSIOp)
    assert isinstance(rem.operands[0], ir.BlockArgument), rem


def an_unsigned_widening_is_rebuilt_unsigned():
    """`extui(u) + 7k urem 7` is `extui(u) urem 7`: u = 0x80000000 gives 2, `extsi(u) urem 7` 0.
    Breaks if _cast_to widens a term peeled through extui with extsi."""
    m = ir.Module.parse("""
func.func @g(%u: i32, %k32: i32, %o: memref<1xi64>) {
  %c7 = arith.constant 7 : i64
  %c0 = arith.constant 0 : index
  %k = arith.extui %k32 : i32 to i64
  %km = arith.muli %k, %c7 : i64
  %ue = arith.extui %u : i32 to i64
  %s = arith.addi %ue, %km : i64
  %m = arith.remui %s, %c7 : i64
  memref.store %m, %o[%c0] : memref<1xi64>
  return
}""")
    _fold(m)
    lhs = producer(_rem(m, arith.RemUIOp).operands[0])
    assert lhs is not None and isinstance(lhs.opview, arith.ExtUIOp), lhs


CASES = [a_carried_value_is_not_nonnegative_by_the_loop_bound,
         the_induction_variable_still_folds,
         an_unsigned_widening_is_rebuilt_unsigned]


def main():
    failed = 0
    for case in CASES:
        try:
            with ir.Context() as ctx, ir.Location.unknown():
                ctx.allow_unregistered_dialects = True
                case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
