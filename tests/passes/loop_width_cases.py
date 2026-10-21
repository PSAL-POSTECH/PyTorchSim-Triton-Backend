"""A spill narrows the place llc names, not the kernel: legalize numbers every loop (a runtime scf
one too) by its location, a loop's width follows the LMUL a retry gave it or a loop around it, and
the retry narrows every place that spills at once, to the width its spilled register units say fits,
puts back a width that relieved none, and halves the kernel's LMUL only when no place can go lower.
python3 tests/passes/loop_width_cases.py; exit 0 iff every case holds"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import pipeline, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p06_legalize  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import funcs, walk  # noqa: E402

VLEN, LMUL = 256, 8

# loop 1 holds loop 2, a vector loop; loop 3 is a vector loop of its own
TWO = """func.func @f(%m: memref<4x512xf32, 1>) {
  affine.for %r = 0 to 4 {
    affine.for %c = 0 to 512 step 512 {
      %x = affine.vector_load %m[%r, %c] : memref<4x512xf32, 1>, vector<512xf32>
      affine.vector_store %x, %m[%r, %c] : memref<4x512xf32, 1>, vector<512xf32>
    } {torchsim.vector_loop}
  }
  affine.for %c = 0 to 512 step 512 {
    %x = affine.vector_load %m[%c, %c] : memref<4x512xf32, 1>, vector<512xf32>
    affine.vector_store %x, %m[%c, %c] : memref<4x512xf32, 1>, vector<512xf32>
  } {torchsim.vector_loop}
  return
}"""


def _widths(loop_lmul):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(TWO)
        f = next(iter(funcs(m)))
        p06_legalize._number_loops(f)
        lines = sorted(ir.FileLineColLoc(lp.location).start_line for lp in walk(f.operation)
                       if lp.name == "affine.for")
        widths = [w.v for w in p06_legalize._choose_widths(f, VLEN, LMUL, loop_lmul)]
        return lines, widths


def every_loop_is_numbered_by_its_location():
    lines, _ = _widths({})
    assert lines == [1, 2, 3], lines


def a_narrowed_loop_narrows_itself_and_what_it_holds_only():
    assert _widths({})[1] == [64, 64]           # LMUL 8: 64 f32
    assert _widths({1: 4})[1] == [32, 64]       # loop 1 at 4: the vector loop in it, not loop 3
    assert _widths({3: 2})[1] == [64, 16]


RUNTIME = """func.func @f(%m: memref<4x512xf32, 1>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %r = %c0 to %n step %c1 {
    affine.for %c = 0 to 512 step 512 {
      %x = affine.vector_load %m[%c, %c] : memref<4x512xf32, 1>, vector<512xf32>
      affine.vector_store %x, %m[%c, %c] : memref<4x512xf32, 1>, vector<512xf32>
    } {torchsim.vector_loop}
  }
  return
}"""


def a_runtime_loop_is_numbered_and_bounds_the_vector_loop_in_it():
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(RUNTIME)
        f = next(iter(funcs(m)))
        p06_legalize._number_loops(f)
        named = {o.name: ir.FileLineColLoc(o.location).start_line for o in walk(f.operation)
                 if o.name in ("scf.for", "affine.for")}
        assert named == {"scf.for": 1, "affine.for": 2}, named
        widths = [w.v for w in p06_legalize._choose_widths(f, VLEN, LMUL, {1: 2})]
        assert widths == [16], widths


def _ctx(**kw):
    c = SimpleNamespace(lmul=LMUL, loop_lmul={}, dot_lmul={}, dot_widths={}, spill_kept=set(),
                        spill_last=None)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _spill(loops=None, dots=None, units=None):
    return SimpleNamespace(loop_spills=loops or {}, dot_spills=dots or {}, spilled_units=units or {})


def every_place_that_spills_is_halved_together():
    ctx = _ctx(dot_widths={7: 8})
    pipeline._narrow_for(ctx, _spill(loops={2: 5}, dots={7: 40}))
    assert ctx.dot_lmul == {7: 4} and ctx.loop_lmul == {2: 4} and ctx.lmul == LMUL, ctx


def a_place_over_by_its_spilled_units_goes_where_they_fit():
    ctx = _ctx(dot_widths={7: 8, 9: 8})
    pipeline._narrow_for(ctx, _spill(dots={7: 3, 9: 3}, units={("dot", 7): 24, ("dot", 9): 40}))
    assert ctx.dot_lmul == {7: 4, 9: 2}, ctx     # 8 * 32 / 56 -> 4, 8 * 32 / 72 -> 2


def a_narrowing_that_relieved_none_goes_back_and_is_kept():
    ctx = _ctx(dot_widths={7: 4}, dot_lmul={7: 4}, spill_last=[(("dot", 7), 8, 40)])
    pipeline._narrow_for(ctx, _spill(loops={2: 5}, dots={7: 40}))
    assert ctx.dot_lmul[7] == 8 and ("dot", 7) in ctx.spill_kept and ctx.loop_lmul == {2: 4}, ctx


def a_spill_no_place_names_halves_the_widest_streams():
    ctx = _ctx(dot_widths={1: 4, 2: 8, 3: 8})
    got = pipeline._narrow_for(ctx, _spill())
    assert got[0] == "lower" and ctx.dot_lmul == {2: 4, 3: 4} and ctx.lmul == LMUL, (got, ctx)


def no_place_to_narrow_halves_the_kernel():
    ctx = _ctx()
    pipeline._narrow_for(ctx, _spill())
    assert ctx.lmul == 4 and ctx.loop_lmul == {}, ctx


CASES = [every_loop_is_numbered_by_its_location, a_narrowed_loop_narrows_itself_and_what_it_holds_only,
         a_runtime_loop_is_numbered_and_bounds_the_vector_loop_in_it,
         every_place_that_spills_is_halved_together, a_place_over_by_its_spilled_units_goes_where_they_fit,
         a_narrowing_that_relieved_none_goes_back_and_is_kept,
         a_spill_no_place_names_halves_the_widest_streams, no_place_to_narrow_halves_the_kernel]

if __name__ == "__main__":
    bad = 0
    for c in CASES:
        try:
            c()
            print(f"ok   {c.__name__}")
        except Exception as e:
            bad += 1
            print(f"FAIL {c.__name__}: {type(e).__name__}: {str(e)[:300]}")
    sys.exit(1 if bad else 0)
