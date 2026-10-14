"""A reduce whose masked trips contribute its identity runs only the live trips. A GEMV with K = 768
in R0_BLOCK 1024 ran all 1024 positions with a compare and a select on each (1x768x3072: 18546 TOGSim
cycles, 12358 with the reduce stopped at K and the select gone).
    python3 tests/passes/masked_reduce_extent_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.lowering.passes import p05_vectorize, p06_legalize  # noqa: E402

_MOD = """
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {{
  memref.global @a : memref<8x64xbf16, 1>
  memref.global @b : memref<8x64xbf16, 1>
  memref.global @o : memref<8xf32, 1>
  func.func @k(%n: i32) {{
    %cst = arith.constant {fill} : f32
    %c48 = arith.constant 48 : i32
    %c44 = arith.constant 44 : i32
    %c5 = arith.constant 5 : i32
    %a = memref.get_global @a : memref<8x64xbf16, 1>
    %b = memref.get_global @b : memref<8x64xbf16, 1>
    %o = memref.get_global @o : memref<8xf32, 1>
    linalg.generic {{indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "reduction"]}}
        ins(%a, %b : memref<8x64xbf16, 1>, memref<8x64xbf16, 1>) outs(%o : memref<8xf32, 1>) {{
    ^bb0(%x: bf16, %y: bf16, %acc: f32):
      %i0 = linalg.index 0 : index
      %i1 = linalg.index 1 : index
      %xf = arith.extf %x : bf16 to f32
      %yf = arith.extf %y : bf16 to f32
      %p = arith.mulf %xf, %yf : f32
      %r = arith.index_cast %i1 : index to i32
      %live = arith.cmpi slt, %r, {bound} : i32
      %l = arith.index_cast %i0 : index to i32
      %row = arith.cmpi slt, %l, %c5 : i32
      %both = arith.andi %live, %row : i1
      %s = arith.select {pred}, %p, %cst : f32
      %sum = arith.addf %s, %acc : f32
      linalg.yield %sum : f32
    }}
    return
  }}
}}
"""


def _reduce_loop(fill="0.0", bound="%c48", pred="%live"):
    """The upper bound and step of the one loop carrying the reduce after p05 + p06 at 8 index-wide
    elements per instruction, and how many selects are left in it."""
    from mlir.ir import AffineMapAttr, IntegerAttr, Module
    from pytorchsim_triton_compiler.lowering.passes import lib_ir
    module = Module.parse(_MOD.format(fill=fill, bound=bound, pred=pred))
    p05_vectorize.run(module, lanes=128)
    p06_legalize.run(module, lanes=128, vlen=256, lmul=2)
    loops = [o for o in lib_ir.walk(module.operation)
             if o.name == "affine.for" and len(o.results)]
    assert len(loops) == 1, module
    ub = AffineMapAttr(loops[0].attributes["upperBoundMap"]).value
    step = IntegerAttr(loops[0].attributes["step"]).value
    selects = [o for o in lib_ir.walk(loops[0]) if o.name == "arith.select"]
    return f"{ub} step {step}", len(selects), module


def a_reduce_masked_below_a_constant_stops_there_without_a_select():
    ub, n, m = _reduce_loop()
    assert ub == "() -> (48) step 8" and n == 0, (ub, n)


def a_conjunction_keeps_the_select_for_its_other_terms():
    ub, n, m = _reduce_loop(pred="%both")
    assert ub == "() -> (48) step 8" and n == 1, (ub, n)


def a_bound_between_width_multiples_ends_past_it_and_keeps_the_select():
    ub, n, m = _reduce_loop(bound="%c44")
    assert ub == "() -> (48) step 8" and n == 1, (ub, n)


def a_fill_that_is_not_the_identity_runs_every_trip():
    ub, n, m = _reduce_loop(fill="1.0")
    assert ub == "() -> (64) step 8" and n == 1, (ub, n)


def a_bound_known_only_at_run_time_runs_every_trip():
    ub, n, m = _reduce_loop(bound="%n")
    assert ub == "() -> (64) step 8" and n == 1, (ub, n)


CASES = [a_reduce_masked_below_a_constant_stops_there_without_a_select,
         a_conjunction_keeps_the_select_for_its_other_terms,
         a_bound_between_width_multiples_ends_past_it_and_keeps_the_select,
         a_fill_that_is_not_the_identity_runs_every_trip,
         a_bound_known_only_at_run_time_runs_every_trip]


def main():
    from mlir.ir import Context, Location
    failed = 0
    for case in CASES:
        try:
            with Context() as ctx, Location.unknown():
                ctx.allow_unregistered_dialects = True
                case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
