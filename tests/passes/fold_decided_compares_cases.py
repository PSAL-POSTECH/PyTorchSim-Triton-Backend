"""tritonpytorchsim-fold-decided-compares folds a compare exactly when the ranges it reads decide
it: each case is one function whose compares feed `test.sink`, and what every sink then reads.
    python3 tests/passes/fold_decided_compares_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.contract import kernel_object  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p03a_fold_decided_compares as fold  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import const_int, walk  # noqa: E402

GRID_ARGS = ", ".join(f"%{p}: i32" for p in kernel_object.TRAILING_GRID_PARAMS)

OVER = """  %g = linalg.generic {{indexing_maps = [{map}, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}} ins(%t : {tin}) outs(%o : {tout}) {{
  ^bb0(%in: f32, %out: f32):
{body}
    linalg.yield %in : f32
  }} -> {tout}
  return %g : {tout}"""


def _sinks(body, args="", grid=None, ret=""):
    """What each `test.sink` of `body` reads after the pass: True, False, or None for a compare
    that stayed."""
    from mlir.ir import Context, Location, Module
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        m = Module.parse(f"func.func @k({args}){ret} {{\n{body}\n}}\n")
        fold.run(m, arg_ranges=None if grid is None else kernel_object.program_id_ranges(grid))
        assert m.operation.verify()
        out = []
        for op in walk(m.operation):
            if op.name == "test.sink":
                c = const_int(op.operands[0])
                out.append(None if c is None else bool(c))
        return out


def _over(body, tin="tensor<4x8xf32>", tout="tensor<4x8xf32>",
          map="affine_map<(d0, d1) -> (d0, d1)>", args="", grid=None):
    """`_sinks` of a body that runs inside one linalg.generic over `tin` -> `tout`."""
    head = f"%t: {tin}, %o: {tout}" + (", " + args if args else "")
    return _sinks(OVER.format(map=map, tin=tin, tout=tout, body=body), head, grid, f" -> {tout}")


def constants_decide_every_predicate():
    preds = {"eq": False, "ne": True, "slt": True, "sle": True, "sgt": False, "sge": False,
             "ult": True, "ule": True, "ugt": False, "uge": False}
    body = "  %a = arith.constant 3 : i32\n  %b = arith.constant 5 : i32\n" + "".join(
        f'  %q{p} = arith.cmpi {p}, %a, %b : i32\n  "test.sink"(%q{p}) : (i1) -> ()\n'
        for p in preds) + "  return"
    assert _sinks(body) == list(preds.values())


def equal_points_decide_eq_and_overlap_does_not():
    assert _over("""    %i = linalg.index 0 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %e = arith.cmpi eq, %c5, %c5 : index
    "test.sink"(%e) : (i1) -> ()
    %n = arith.cmpi ne, %i, %c5 : index
    "test.sink"(%n) : (i1) -> ()
    %u = arith.cmpi eq, %i, %c2 : index
    "test.sink"(%u) : (i1) -> ()""") == [True, True, None]


def a_loop_variable_ends_at_its_last_step_not_at_its_bound():
    assert _sinks("""  %c0 = arith.constant 0 : i32
  %c512 = arith.constant 512 : i32
  %c2048 = arith.constant 2048 : i32
  %c1537 = arith.constant 1537 : i32
  %c1536 = arith.constant 1536 : i32
  scf.for %i = %c0 to %c2048 step %c512 : i32 {
    %a = arith.cmpi slt, %i, %c1537 : i32
    "test.sink"(%a) : (i1) -> ()
    %b = arith.cmpi slt, %i, %c1536 : i32
    "test.sink"(%b) : (i1) -> ()
    %c = arith.cmpi sge, %i, %c0 : i32
    "test.sink"(%c) : (i1) -> ()
  }
  return""") == [True, None, True]


def a_bound_the_step_does_not_divide_still_ends_at_the_last_step():
    assert _sinks("""  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c20 = arith.constant 20 : index
  %c18 = arith.constant 18 : index
  %c19 = arith.constant 19 : index
  scf.for %i = %c3 to %c20 step %c5 {
    %a = arith.cmpi sle, %i, %c18 : index
    "test.sink"(%a) : (i1) -> ()
    %b = arith.cmpi slt, %i, %c18 : index
    "test.sink"(%b) : (i1) -> ()
    %c = arith.cmpi ult, %i, %c19 : index
    "test.sink"(%c) : (i1) -> ()
  }
  return""") == [True, None, True]


def a_loop_whose_bounds_are_not_constants_is_unknown():
    assert _sinks("""  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c9 = arith.constant 9 : index
  scf.for %i = %c0 to %n step %c1 {
    %a = arith.cmpi sge, %i, %c0 : index
    "test.sink"(%a) : (i1) -> ()
  }
  scf.for %j = %c9 to %c0 step %c1 {
    %b = arith.cmpi sge, %j, %c0 : index
    "test.sink"(%b) : (i1) -> ()
  }
  %r = scf.for %k = %c0 to %c9 step %c1 iter_args(%x = %c0) -> (index) {
    %c = arith.cmpi sge, %x, %c0 : index
    "test.sink"(%c) : (i1) -> ()
    scf.yield %k : index
  }
  return""", "%n: index") == [None, None, None]


def an_index_runs_the_extent_its_map_names():
    assert _over("""    %i = linalg.index 0 : index
    %j = linalg.index 1 : index
    %c4 = arith.constant 4 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %a = arith.cmpi slt, %i, %c4 : index
    "test.sink"(%a) : (i1) -> ()
    %b = arith.cmpi slt, %j, %c4 : index
    "test.sink"(%b) : (i1) -> ()
    %c = arith.cmpi slt, %j, %c8 : index
    "test.sink"(%c) : (i1) -> ()
    %d = arith.cmpi sge, %j, %c8 : index
    "test.sink"(%d) : (i1) -> ()
    %e = arith.cmpi slt, %j, %c7 : index
    "test.sink"(%e) : (i1) -> ()""") == [True, None, True, False, None]


def an_index_is_read_through_a_permuted_map():
    assert _over("""    %i = linalg.index 0 : index
    %c4 = arith.constant 4 : index
    %a = arith.cmpi slt, %i, %c4 : index
    "test.sink"(%a) : (i1) -> ()""", tin="tensor<8x4xf32>", tout="tensor<4x?xf32>",
                 map="affine_map<(d0, d1) -> (d1, d0)>") == [True]


def an_index_over_a_dynamic_extent_is_unknown():
    assert _over("""    %j = linalg.index 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %a = arith.cmpi slt, %j, %c8 : index
    "test.sink"(%a) : (i1) -> ()
    %b = arith.cmpi sge, %j, %c0 : index
    "test.sink"(%b) : (i1) -> ()""", tin="tensor<4x?xf32>", tout="tensor<4x?xf32>") == [None, None]


def the_tile_mask_of_a_looped_reduction_is_always_true():
    assert _over("""    %j = linalg.index 1 : index
    %j32 = arith.index_cast %j : index to i32
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %c32 = arith.constant 32 : i32
    %c30 = arith.constant 30 : i32
    scf.for %r = %c0 to %c32 step %c8 : i32 {
      %at = arith.addi %r, %j32 : i32
      %a = arith.cmpi slt, %at, %c32 : i32
      "test.sink"(%a) : (i1) -> ()
      %b = arith.cmpi slt, %at, %c30 : i32
      "test.sink"(%b) : (i1) -> ()
    }""") == [True, None]


def a_sum_or_product_past_the_type_is_unknown():
    assert _over("""    %j = linalg.index 1 : index
    %j8 = arith.index_cast %j : index to i8
    %j32 = arith.index_cast %j : index to i32
    %z8 = arith.constant 0 : i8
    %z32 = arith.constant 0 : i32
    %s8 = arith.addi %j8, %j8 : i8
    %a = arith.cmpi sge, %s8, %z8 : i8
    "test.sink"(%a) : (i1) -> ()
    %s32 = arith.addi %j32, %j32 : i32
    %b = arith.cmpi sge, %s32, %z32 : i32
    "test.sink"(%b) : (i1) -> ()
    %p8 = arith.muli %j8, %j8 : i8
    %c = arith.cmpi sge, %p8, %z8 : i8
    "test.sink"(%c) : (i1) -> ()
    %p32 = arith.muli %j32, %j32 : i32
    %d = arith.cmpi sge, %p32, %z32 : i32
    "test.sink"(%d) : (i1) -> ()""", tin="tensor<4x101xf32>", tout="tensor<4x101xf32>") == [
        None, True, None, True]


def a_cast_that_may_truncate_is_unknown():
    assert _over("""    %j = linalg.index 1 : index
    %j8 = arith.index_cast %j : index to i8
    %z8 = arith.constant 0 : i8
    %a = arith.cmpi sge, %j8, %z8 : i8
    "test.sink"(%a) : (i1) -> ()
    %j16 = arith.index_cast %j : index to i16
    %z16 = arith.constant 0 : i16
    %b = arith.cmpi sge, %j16, %z16 : i16
    "test.sink"(%b) : (i1) -> ()""", tin="tensor<4x300xf32>", tout="tensor<4x300xf32>") == [
        None, True]


def an_index_is_held_to_its_least_width():
    wide = "tensor<4x4294967296xf32>"
    assert _over("""    %j = linalg.index 1 : index
    %i = linalg.index 0 : index
    %c0 = arith.constant 0 : index
    %a = arith.cmpi sge, %j, %c0 : index
    "test.sink"(%a) : (i1) -> ()
    %b = arith.cmpi sge, %i, %c0 : index
    "test.sink"(%b) : (i1) -> ()""", tin=wide, tout=wide) == [None, True]


def an_unsigned_compare_puts_the_negatives_above():
    assert _over("""    %j = linalg.index 1 : index
    %j32 = arith.index_cast %j : index to i32
    %m1 = arith.constant -1 : i32
    %m4 = arith.constant -4 : i32
    %m9 = arith.constant -9 : i32
    %c8 = arith.constant 8 : i32
    %a = arith.cmpi ult, %j32, %m1 : i32
    "test.sink"(%a) : (i1) -> ()
    %b = arith.cmpi slt, %j32, %m1 : i32
    "test.sink"(%b) : (i1) -> ()
    %around = arith.addi %j32, %m4 : i32
    %c = arith.cmpi ult, %around, %c8 : i32
    "test.sink"(%c) : (i1) -> ()
    %d = arith.cmpi slt, %around, %c8 : i32
    "test.sink"(%d) : (i1) -> ()
    %below = arith.addi %j32, %m9 : i32
    %e = arith.cmpi ugt, %below, %c8 : i32
    "test.sink"(%e) : (i1) -> ()
    %f = arith.cmpi ult, %below, %m1 : i32
    "test.sink"(%f) : (i1) -> ()
    %h = arith.cmpi uge, %j32, %c8 : i32
    "test.sink"(%h) : (i1) -> ()""") == [True, False, None, True, True, True, False]


def a_negative_factor_turns_the_range_over():
    assert _over("""    %j = linalg.index 1 : index
    %m2 = arith.constant -2 : index
    %c0 = arith.constant 0 : index
    %m14 = arith.constant -14 : index
    %p = arith.muli %j, %m2 : index
    %a = arith.cmpi sle, %p, %c0 : index
    "test.sink"(%a) : (i1) -> ()
    %b = arith.cmpi sge, %p, %m14 : index
    "test.sink"(%b) : (i1) -> ()
    %c = arith.cmpi sgt, %p, %m14 : index
    "test.sink"(%c) : (i1) -> ()""") == [True, True, None]


def what_is_not_bounded_is_left_alone():
    assert _sinks("""  %c0 = arith.constant 0 : i32
  %a = arith.cmpi sge, %x, %c0 : i32
  "test.sink"(%a) : (i1) -> ()
  %d = arith.subi %c0, %c0 : i32
  %b = arith.cmpi eq, %d, %c0 : i32
  "test.sink"(%b) : (i1) -> ()
  %t = arith.constant true
  %c = arith.cmpi eq, %t, %t : i1
  "test.sink"(%c) : (i1) -> ()
  %v = arith.constant dense<[1, 2]> : vector<2xi32>
  %w = arith.cmpi slt, %v, %v : vector<2xi32>
  "test.sink.vector"(%w) : (vector<2xi1>) -> ()
  return""", "%x: i32") == [None, None, None]


def a_program_id_is_inside_the_grid_it_was_compiled_for():
    body = """    %i = linalg.index 0 : index
    %i32 = arith.index_cast %i : index to i32
    %c256 = arith.constant 256 : i32
    %c2048 = arith.constant 2048 : i32
    %at = arith.muli %pidX, %c256 : i32
    %x = arith.addi %at, %i32 : i32
    %a = arith.cmpi slt, %x, %c2048 : i32
    "test.sink"(%a) : (i1) -> ()
    %y = arith.addi %pidY, %i32 : i32
    %b = arith.cmpi slt, %y, %c256 : i32
    "test.sink"(%b) : (i1) -> ()
    %c = arith.cmpi slt, %gridX, %c2048 : i32
    "test.sink"(%c) : (i1) -> ()"""
    shape = dict(tin="tensor<256x8xf32>", tout="tensor<256x8xf32>", args=GRID_ARGS)
    assert _over(body, grid=(8,), **shape) == [True, True, None]
    assert _over(body, grid=(9,), **shape) == [None, True, None]
    assert _over(body, grid=(8, 2), **shape) == [True, None, None]
    assert _over(body, grid=None, **shape) == [None, None, None]


CASES = [constants_decide_every_predicate, equal_points_decide_eq_and_overlap_does_not,
         a_loop_variable_ends_at_its_last_step_not_at_its_bound,
         a_bound_the_step_does_not_divide_still_ends_at_the_last_step,
         a_loop_whose_bounds_are_not_constants_is_unknown, an_index_runs_the_extent_its_map_names,
         an_index_is_read_through_a_permuted_map, an_index_over_a_dynamic_extent_is_unknown,
         the_tile_mask_of_a_looped_reduction_is_always_true,
         a_sum_or_product_past_the_type_is_unknown, a_cast_that_may_truncate_is_unknown,
         an_index_is_held_to_its_least_width, an_unsigned_compare_puts_the_negatives_above,
         a_negative_factor_turns_the_range_over, what_is_not_bounded_is_left_alone,
         a_program_id_is_inside_the_grid_it_was_compiled_for]


def main():
    bad = 0
    for case in CASES:
        try:
            case()
            print(f"ok    {case.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
