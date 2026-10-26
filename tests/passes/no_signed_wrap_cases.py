"""pytorchsim-simplify-index-arithmetic marks an add or multiply nsw only by the index premise or a proof, and
its narrowing offers only the widths a function reads its index vectors at.
    python3 tests/passes/no_signed_wrap_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from mlir.dialects import arith  # noqa: E402
from pytorchsim_triton_compiler.dialects import pytorchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import (  # noqa: E402
    b09_simplify_index_arithmetic)
from pytorchsim_triton_compiler.lowering.passes.lib_ir import funcs, walk  # noqa: E402

pytorchsim.register()

BUF = "memref<128xf32, 1>"
_STORE = ('"pytorchsim.transfer"(%d, {at}, %a, %c0, %c0) <{{async = true, direction = "store", '
          'dram_arg = 0 : i64, dram_stride = [1], '
          'operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}> '
          f': (memref<4096xf32>, index, {BUF}, index, index) -> ()')


def _muls(m):
    return [o for o in walk(m.operation) if isinstance(o.opview, arith.MulIOp)]


def a_program_id_times_its_block_is_nsw_by_the_premise():
    """`pid * 512` in i32 is an index: nsw by kernel_spec.OFFSET_INDEX_BITS, though the id's range alone
    lets it wrap. Breaks if the premise stops reaching i32 index arithmetic."""
    m = ir.Module.parse(f"""
func.func @f(%d: memref<4096xf32>) {{
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : i32
  %a = memref.alloc() : {BUF}
  %p = "pytorchsim.program_id"() <{{axis = 0 : i32}}> : () -> i32
  %o = arith.muli %p, %c512 : i32
  %i = arith.index_cast %o : i32 to index
  {_STORE.format(at="%i")}
  return
}}""")
    b09_simplify_index_arithmetic.run(m, arg_numels={0: 4096})
    assert all(pytorchsim.has_no_signed_wrap(o) for o in _muls(m)), m


def a_loaded_value_added_to_an_index_is_data_and_not_marked():
    """`pid + x` with x loaded from a buffer is data, not an index: no nsw, though the program id is.
    Breaks if the premise reaches integer values the kernel loads."""
    m = ir.Module.parse(f"""
func.func @f(%d: memref<4096xf32>, %b: memref<1xi32, 1>) {{
  %c0 = arith.constant 0 : index
  %a = memref.alloc() : {BUF}
  %p = "pytorchsim.program_id"() <{{axis = 0 : i32}}> : () -> i32
  %x = memref.load %b[%c0] : memref<1xi32, 1>
  %s = arith.addi %p, %x : i32
  memref.store %s, %b[%c0] : memref<1xi32, 1>
  return
}}""")
    b09_simplify_index_arithmetic.run(m, arg_numels={0: 4096})
    adds = [o for o in walk(m.operation) if isinstance(o.opview, arith.AddIOp)]
    assert adds and not any(pytorchsim.has_no_signed_wrap(o) for o in adds), m


def a_multiply_whose_product_may_wrap_is_not_marked_by_its_bounded_result():
    """`u * 2^40` (u an unsigned i32 widened) lands in a 4096-element argument: its operands are non-negative
    and its result bounded, but a wrapped product lands there too. Breaks if a multiply is marked so."""
    m = ir.Module.parse(f"""
func.func @f(%d: memref<4096xf32>, %u: i32) {{
  %c0 = arith.constant 0 : index
  %big = arith.constant 1099511627776 : index
  %a = memref.alloc() : {BUF}
  %x = arith.index_castui %u : i32 to index
  %i = arith.muli %x, %big : index
  {_STORE.format(at="%i")}
  return
}}""")
    b09_simplify_index_arithmetic.run(m, arg_numels={0: 4096})
    assert not any(pytorchsim.has_no_signed_wrap(o) for o in _muls(m)), m


def an_i8_data_vector_does_not_offer_i8_to_lane_arithmetic():
    """A function with i8 data and lane numbers cast to i32 is offered i32 alone. Breaks if b09 offers
    every integer vector width the function has (lane arithmetic drawn down to i8)."""
    m = ir.Module.parse("""
func.func @f(%b: memref<16xi8, 1>, %o: memref<16xi32, 1>) {
  %c0 = arith.constant 0 : index
  %l = "pytorchsim.lane_id"() <{lanes = 128 : i64}> : () -> vector<16xindex>
  %w = arith.index_cast %l : vector<16xindex> to vector<16xi32>
  vector.store %w, %o[%c0] : memref<16xi32, 1>, vector<16xi32>
  %v = vector.load %b[%c0] : memref<16xi8, 1>, vector<16xi8>
  vector.store %v, %b[%c0] : memref<16xi8, 1>, vector<16xi8>
  return
}""")
    got = b09_simplify_index_arithmetic._vector_int_widths(funcs(m)[0])
    assert got == [32], got


CASES = [a_program_id_times_its_block_is_nsw_by_the_premise,
         a_loaded_value_added_to_an_index_is_data_and_not_marked,
         a_multiply_whose_product_may_wrap_is_not_marked_by_its_bounded_result,
         an_i8_data_vector_does_not_offer_i8_to_lane_arithmetic]


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
