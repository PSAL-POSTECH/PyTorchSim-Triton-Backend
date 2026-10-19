"""torchsim-peel-reduction-init starts a loop-carried reduction from its first trip's value when it starts from
its combiner's identity, and leaves every other loop as it is: a start that is not the identity, a combination
the loop also reads, a loop that fills masked lanes with that identity inside, and a loop of one trip.
    python3 tests/passes/peel_reduction_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p06c_peel_reduction_init as peel  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import loop_bounds, walk  # noqa: E402


def _reduce(start, combine, elem="f32", n=1024, extra=""):
    """A loop over `n` elements, 64 a trip, that combines each trip's load into a value carried from `start`."""
    return f'''func.func @f(%m: memref<{n}x{elem}>, %o: memref<64x{elem}>) -> vector<64x{elem}> {{
  %s = arith.constant dense<{start}> : vector<64x{elem}>
  %r = affine.for %i = 0 to {n} step 64 iter_args(%acc = %s) -> (vector<64x{elem}>) {{
    %x = affine.vector_load %m[%i] : memref<{n}x{elem}>, vector<64x{elem}>
    %y = {combine} %x, %acc : vector<64x{elem}>
    {extra}
    affine.yield %y : vector<64x{elem}>
  }}
  return %r : vector<64x{elem}>
}}'''


def _peeled(body):
    """(each loop's start bound, how many arith.constant ops are left) after the pass."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse("module {\n" + body + "\n}\n")
        peel.run(m)
        assert m.operation.verify()
        ops = list(walk(m.operation))
        return ([loop_bounds(o)[0] for o in ops if o.name == "affine.for"],
                sum(o.name == "arith.constant" for o in ops))


def a_max_from_minus_infinity_starts_from_its_first_trip():
    starts, consts = _peeled(_reduce("0xFF800000", "arith.maximumf"))
    assert starts == [64] and consts == 1, (starts, consts)   # the one constant left is the index 0


def an_integer_sum_from_zero_starts_from_its_first_trip():
    starts, _ = _peeled(_reduce("0", "arith.addi", elem="i32"))
    assert starts == [64], starts


def a_start_that_is_not_the_identity_is_left():
    starts, consts = _peeled(_reduce("1.0", "arith.addf"))
    assert starts == [0] and consts == 1, (starts, consts)


def a_combination_the_loop_also_reads_is_left():
    starts, _ = _peeled(_reduce("0xFF800000", "arith.maximumf",
                                extra="affine.vector_store %y, %o[0] : memref<64xf32>, vector<64xf32>"))
    assert starts == [0], starts


def a_reduction_that_fills_with_its_identity_inside_is_left():
    body = _reduce("0.0", "arith.addf").replace(
        "%y = arith.addf %x, %acc", "%f = arith.constant dense<0.0> : vector<64xf32>\n"
        "    %k = arith.cmpf olt, %x, %f : vector<64xf32>\n"
        "    %z = arith.select %k, %x, %f : vector<64xi1>, vector<64xf32>\n"
        "    %y = arith.addf %z, %acc")
    starts, _ = _peeled(body)
    assert starts == [0], starts


def a_loop_of_one_trip_is_left():
    starts, _ = _peeled(_reduce("0xFF800000", "arith.maximumf", n=64))
    assert starts == [0], starts


CASES = [a_max_from_minus_infinity_starts_from_its_first_trip,
         an_integer_sum_from_zero_starts_from_its_first_trip,
         a_start_that_is_not_the_identity_is_left,
         a_combination_the_loop_also_reads_is_left,
         a_reduction_that_fills_with_its_identity_inside_is_left,
         a_loop_of_one_trip_is_left]


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
