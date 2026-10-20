"""lower_vector_functions settles a causal mask's selects before the SFU ops are placed: a loop whose
trips settle a select alike is cut into one loop per run of such trips, each select folded to what it
picks there; a running max of a value made outside the loop is folded once before it; a select by a
compare of i32 vectors whose difference fits in an i32 is made of the difference's sign; and a
one-element vector spread over a register group is spread over one register and concatenated. A select
whose compare can overflow keeps its mask.
    python3 tests/passes/mask_selects_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import lower_vector_functions as lvf  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

# Row 3's causal mask over 8 trips of one column each: trips 0..3 keep x, 4..7 take -inf; the max
# carried along them starts at -inf.
_MASKED_MAX = """func.func @f(%m: memref<64xf32>, %o: memref<8xf32>) {
  %row = arith.constant dense<3> : vector<8xi32>
  %no = arith.constant dense<0xFF800000> : vector<8xf32>
  %r = affine.for %i = 0 to 64 step 8 iter_args(%acc = %no) -> (vector<8xf32>) {
    %x = affine.vector_load %m[%i] : memref<64xf32>, vector<8xf32>
    %t = arith.index_cast %i : index to i32
    %col = vector.broadcast %t : i32 to vector<8xi32>
    %keep = arith.cmpi sge, %row, %col : vector<8xi32>
    %y = arith.select %keep, %x, %no : vector<8xi1>, vector<8xf32>
    %n = arith.maxnumf %y, %acc : vector<8xf32>
    affine.yield %n : vector<8xf32>
  }
  affine.vector_store %r, %o[0] : memref<8xf32>, vector<8xf32>
  return
}"""


# A select by a compare of a lane number against `B`: its difference fits in an i32 for a constant,
# not for a value of the function's.
_LANE_SELECT = """func.func @f(%m: memref<8xf32>, %v: vector<8xi32>) {
  %l = "torchsim.vlane_idx"() : () -> vector<8xindex>
  %li = arith.index_cast %l : vector<8xindex> to vector<8xi32>
  %c = arith.constant dense<5> : vector<8xi32>
  %x = affine.vector_load %m[0] : memref<8xf32>, vector<8xf32>
  %no = arith.constant dense<0.0> : vector<8xf32>
  %k = arith.cmpi slt, %li, B : vector<8xi32>
  %y = arith.select %k, %x, %no : vector<8xi1>, vector<8xf32>
  affine.vector_store %y, %m[0] : memref<8xf32>, vector<8xf32>
  return
}"""


# One element spread over 64 f32, eight registers of VLEN 256.
_SPREAD = """func.func @f(%m: memref<1xf32>, %o: memref<64xf32>) {
  %x = affine.vector_load %m[0] : memref<1xf32>, vector<1xf32>
  %y = vector.broadcast %x : vector<1xf32> to vector<64xf32>
  affine.vector_store %y, %o[0] : memref<64xf32>, vector<64xf32>
  return
}"""


def _run(src):
    """`src` through lower_vector_functions on the v6e, in the current context; its ops. The module
    must verify after: a cut loop's offset counter feeds affine loads, which take only a dim (the
    next stage parses the module again and fails on anything else)."""
    with ir.Location.unknown():
        m = ir.Module.parse("module {\n" + src + "\n}\n")
        lvf.run(m, target=machine.target())
        assert m.operation.verify()
        return [o for o in walk(m.operation)], m


def a_masked_running_max_is_cut_by_trip_and_its_selects_folded():
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        ops, _ = _run(_MASKED_MAX)
        names = [o.name for o in ops]
        loops = [o for o in ops if o.name == "affine.for"]
        assert len(loops) == 2, names
        assert "arith.select" not in names, names
        # The trips that take -inf fold -inf into the max: that loop carries the max as it is.
        carried = [o for o in loops
                   if not any(x.name == "arith.maxnumf" for x in walk(o))]
        assert len(carried) == 1, names


def a_select_by_a_lane_number_against_a_constant_is_made_of_signs():
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        ops, _ = _run(_LANE_SELECT.replace("B", "%c"))
        names = [o.name for o in ops]
        assert "arith.select" not in names, names
        assert "arith.shrsi" in names, names


def a_select_whose_difference_may_overflow_keeps_its_mask():
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        ops, _ = _run(_LANE_SELECT.replace("B", "%v"))
        names = [o.name for o in ops]
        assert "arith.select" in names and "arith.shrsi" not in names, names


def one_element_spread_over_a_group_is_spread_over_one_register_and_concatenated():
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        ops, _ = _run(_SPREAD)
        spreads = [o for o in ops if o.name == "vector.broadcast"]
        assert len(spreads) == 1 and ir.VectorType(spreads[0].results[0].type).shape == [8], spreads
        assert sum(o.name == "vector.shuffle" for o in ops) == 3, [o.name for o in ops]


CASES = [a_masked_running_max_is_cut_by_trip_and_its_selects_folded,
         a_select_by_a_lane_number_against_a_constant_is_made_of_signs,
         a_select_whose_difference_may_overflow_keeps_its_mask,
         one_element_spread_over_a_group_is_spread_over_one_register_and_concatenated]


if __name__ == "__main__":
    bad = 0
    for case in CASES:
        try:
            case()
            print(f"ok    {case.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    sys.exit(1 if bad else 0)
