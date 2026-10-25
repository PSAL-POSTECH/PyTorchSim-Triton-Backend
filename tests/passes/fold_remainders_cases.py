"""pytorchsim-fold-remainders drops the terms its divisor divides only where that keeps the value.
    python3 tests/passes/fold_remainders_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import b10_fold_remainders  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import producer, walk  # noqa: E402


def _rem(m, name):
    return next(o for o in walk(m.operation) if o.name == name)


def a_carried_value_is_not_nonnegative_by_the_loop_bound():
    """`(a + 8) srem 8` with `a` carried down from 0 by 4 a lap: a = -4 gives 4, `a srem 8` -4.
    Breaks if _nonneg takes an affine.for's iter_args as non-negative by its lower bound."""
    m = ir.Module.parse("""
func.func @f(%init: i32, %o: memref<4xi32>) {
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
    b10_fold_remainders.run(m)
    lhs = producer(_rem(m, "arith.remsi").operands[0])
    assert lhs is not None and lhs.name == "arith.addi", lhs


def the_induction_variable_still_folds():
    """`(i + 8) srem 8` with i from 0 is `i srem 8`. Breaks if the iter_args fix also stops the IV."""
    m = ir.Module.parse("""
func.func @f(%o: memref<4xindex>) {
  %c8 = arith.constant 8 : index
  affine.for %i = 0 to 4 {
    %s = arith.addi %i, %c8 : index
    %m = arith.remsi %s, %c8 : index
    affine.store %m, %o[%i] : memref<4xindex>
  }
  return
}""")
    b10_fold_remainders.run(m)
    rem = _rem(m, "arith.remsi")
    assert isinstance(rem.operands[0], ir.BlockArgument), rem


def an_unsigned_widening_is_rebuilt_unsigned():
    """`extui(u) + 7k urem 7` is `extui(u) urem 7`: u = 0x80000000 gives 2, `extsi(u) urem 7` 0.
    Breaks if _cast_to widens a term peeled through extui with extsi."""
    m = ir.Module.parse("""
func.func @g(%u: i32, %k: i64, %o: memref<1xi64>) {
  %c7 = arith.constant 7 : i64
  %c0 = arith.constant 0 : index
  %km = arith.muli %k, %c7 : i64
  %ue = arith.extui %u : i32 to i64
  %s = arith.addi %ue, %km : i64
  %m = arith.remui %s, %c7 : i64
  memref.store %m, %o[%c0] : memref<1xi64>
  return
}""")
    b10_fold_remainders.run(m)
    lhs = producer(_rem(m, "arith.remui").operands[0])
    assert lhs is not None and lhs.name == "arith.extui", lhs


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
