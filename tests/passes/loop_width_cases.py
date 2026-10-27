"""A spill narrows the place llc names, not the kernel: legalize numbers every loop (a runtime scf
one too) by its location, a loop's width follows the LMUL a retry gave it or a loop around it, and
the retry narrows every place that spills at once, to the width its spilled register units say fits,
puts back a width that relieved none, and halves the kernel's LMUL only when no place can go lower. A
loop starts at the widest LMUL whose live registers fit the register file, and the retry, judged by
the spills llc counts (no simulator), goes on while they stand and stops after max_retries() builds on
the widths that spilled least.
python3 tests/passes/loop_width_cases.py; exit 0 iff every case holds"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.driver import compiler  # noqa: E402
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from mlir.dialects import affine, scf, vector  # noqa: E402
from pytorchsim_triton_compiler.passes.transforms import legalize_vectors  # noqa: E402
from pytorchsim_triton_compiler.passes.utils.ir_utils import funcs, loop_bounds, walk  # noqa: E402

VLEN, LMUL = 256, 8

# loop 1 holds loop 2, a vector loop; loop 3 is a vector loop of its own
TWO = """func.func @f(%m: memref<4x512xf32, 1>) {
  affine.for %r = 0 to 4 {
    affine.for %c = 0 to 512 step 512 {
      %x = affine.vector_load %m[%r, %c] : memref<4x512xf32, 1>, vector<512xf32>
      affine.vector_store %x, %m[%r, %c] : memref<4x512xf32, 1>, vector<512xf32>
    } {pytorchsim.vector_loop}
  }
  affine.for %c = 0 to 512 step 512 {
    %x = affine.vector_load %m[%c, %c] : memref<4x512xf32, 1>, vector<512xf32>
    affine.vector_store %x, %m[%c, %c] : memref<4x512xf32, 1>, vector<512xf32>
  } {pytorchsim.vector_loop}
  return
}"""


def _widths(loop_lmul):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(TWO)
        f = next(iter(funcs(m)))
        legalize_vectors._number_loops(f)
        lines = sorted(ir.FileLineColLoc(lp.location).start_line for lp in walk(f.operation)
                       if isinstance(lp.opview, affine.AffineForOp))
        widths = [w.v for w in legalize_vectors._choose_widths(f, VLEN, LMUL, loop_lmul)]
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
    } {pytorchsim.vector_loop}
  }
  return
}"""


def a_runtime_loop_is_numbered_and_bounds_the_vector_loop_in_it():
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(RUNTIME)
        f = next(iter(funcs(m)))
        legalize_vectors._number_loops(f)
        named = {type(o.opview): ir.FileLineColLoc(o.location).start_line for o in walk(f.operation)
                 if isinstance(o.opview, (scf.ForOp, affine.AffineForOp))}
        assert named == {scf.ForOp: 1, affine.AffineForOp: 2}, named
        widths = [w.v for w in legalize_vectors._choose_widths(f, VLEN, LMUL, {1: 2})]
        assert widths == [16], widths


def _one_loop(step):
    """A vector loop of `step` f32 carrying a sum out to a reader, at LMUL 8 (cap 64)."""
    return f"""func.func @f(%m: memref<{step}xf32, 1>) {{
  %z = arith.constant dense<0.0> : vector<{step}xf32>
  %r = affine.for %c = 0 to {step} step {step} iter_args(%a = %z) -> (vector<{step}xf32>) {{
    %x = affine.vector_load %m[%c] : memref<{step}xf32, 1>, vector<{step}xf32>
    %s = arith.addf %x, %a : vector<{step}xf32>
    affine.yield %s : vector<{step}xf32>
  }} {{pytorchsim.vector_loop}}
  "test.use"(%r) : (vector<{step}xf32>) -> ()
  return
}}"""


def _legalized(step):
    """The legalized loops' (start, end, step), the reader's operand as (shape, element type) and the
    number of vector.shuffle ops; the reader is the op before the return, its element type told by
    whether it is f32."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(_one_loop(step))
        f = next(iter(funcs(m)))
        for w in legalize_vectors._choose_widths(f, VLEN, LMUL):
            legalize_vectors._apply_width(w)
        loops = [loop_bounds(lp) for lp in walk(f.operation) if isinstance(lp.opview, affine.AffineForOp)]
        use = list(f.regions[0].blocks[0].operations)[-2]
        used = ir.VectorType(use.operands[0].type)
        shuffles = sum(isinstance(o.opview, vector.ShuffleOp) for o in walk(f.operation))
        return loops, (used.shape, used.element_type == ir.F32Type.get()), shuffles


def a_width_no_divisor_fits_runs_at_the_cap_and_one_tail_carries_the_sum_on():
    loops, used, shuffles = _legalized(127)
    assert loops == [(0, 64, 64), (64, 127, 63)], loops
    assert used == ([64], True) and shuffles == 2, (used, shuffles)


def a_divisor_as_few_trips_as_the_cap_takes_no_tail():
    loops, used, shuffles = _legalized(96)
    assert loops == [(0, 96, 48)], loops
    assert used == ([48], True) and shuffles == 0, (used, shuffles)


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
    compiler._narrow_for(ctx, _spill(loops={2: 5}, dots={7: 40}))
    assert ctx.dot_lmul == {7: 4} and ctx.loop_lmul == {2: 4} and ctx.lmul == LMUL, ctx


def a_place_over_by_its_spilled_units_goes_where_they_fit():
    ctx = _ctx(dot_widths={7: 8, 9: 8})
    compiler._narrow_for(ctx, _spill(dots={7: 3, 9: 3}, units={("dot", 7): 24, ("dot", 9): 40}))
    assert ctx.dot_lmul == {7: 4, 9: 2}, ctx     # 8 * 32 / 56 -> 4, 8 * 32 / 72 -> 2


def a_narrowing_that_relieved_none_goes_back_and_is_kept():
    ctx = _ctx(dot_widths={7: 4}, dot_lmul={7: 4}, spill_last=[(("dot", 7), 8, 40)])
    compiler._narrow_for(ctx, _spill(loops={2: 5}, dots={7: 40}))
    assert ctx.dot_lmul[7] == 8 and ("dot", 7) in ctx.spill_kept and ctx.loop_lmul == {2: 4}, ctx


def a_spill_no_place_names_halves_the_widest_streams():
    ctx = _ctx(dot_widths={1: 4, 2: 8, 3: 8})
    got = compiler._narrow_for(ctx, _spill())
    assert got[0] == "lower" and ctx.dot_lmul == {2: 4, 3: 4} and ctx.lmul == LMUL, (got, ctx)


def no_place_to_narrow_halves_the_kernel():
    ctx = _ctx()
    compiler._narrow_for(ctx, _spill())
    assert ctx.lmul == 4 and ctx.loop_lmul == {}, ctx

# five f32 vectors live at once before the first add: 5 x 8 registers at LMUL 8 is over 32, 5 x 4 fits
FIVE = """func.func @f(%m: memref<5x512xf32, 1>) {
  affine.for %c = 0 to 512 step 512 {
    %a = affine.vector_load %m[0, %c] : memref<5x512xf32, 1>, vector<512xf32>
    %b = affine.vector_load %m[1, %c] : memref<5x512xf32, 1>, vector<512xf32>
    %d = affine.vector_load %m[2, %c] : memref<5x512xf32, 1>, vector<512xf32>
    %e = affine.vector_load %m[3, %c] : memref<5x512xf32, 1>, vector<512xf32>
    %g = affine.vector_load %m[4, %c] : memref<5x512xf32, 1>, vector<512xf32>
    %s1 = arith.addf %a, %b : vector<512xf32>
    %s2 = arith.addf %s1, %d : vector<512xf32>
    %s3 = arith.addf %s2, %e : vector<512xf32>
    %s4 = arith.addf %s3, %g : vector<512xf32>
    affine.vector_store %s4, %m[0, %c] : memref<5x512xf32, 1>, vector<512xf32>
  } {pytorchsim.vector_loop}
  return
}"""


def a_loop_whose_live_values_overfill_the_registers_starts_narrower():
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(FIVE)
        f = next(iter(funcs(m)))
        legalize_vectors._number_loops(f)
        widths = [w.v for w in legalize_vectors._choose_widths(f, VLEN, LMUL, {})]
    assert widths == [32], widths               # LMUL 4: 32 f32, not LMUL 8's 64


def _judged(spills_per_build):
    """The retry's answers for builds that spill `spills_per_build` in loop 2, in turn."""
    ctx = _ctx(retry_best=None, retries=0, loops_settled=False)
    said = []
    for n in spills_per_build:
        ctx.spills = _spill(loops={2: n})
        said.append(compiler._judge_spills(ctx))
        if ctx.loops_settled:
            break
    return ctx, said


def a_narrowing_that_spills_no_fewer_goes_on_and_the_fewest_spills_stand():
    ctx, said = _judged([10, 10, 12, 11])
    assert ctx.loops_settled and ctx.loop_lmul == {} and said[-1][0] == "transform", (ctx, said)
    assert "3 retries spent: the widths that spilled 10 stand" in said[-1][1], said


def the_retry_stops_after_max_retries_on_the_widths_that_spilled_least():
    assert compiler.max_retries() == 3              # the ISA's LMUL 8 halves three times to 1
    ctx, said = _judged([40, 30, 20, 10, 5])
    assert ctx.loops_settled and ctx.retries == 3, (ctx, said)
    assert "3 retries spent" in said[-1][1] and len(said) == 4, said


CASES = [every_loop_is_numbered_by_its_location, a_narrowed_loop_narrows_itself_and_what_it_holds_only,
         a_runtime_loop_is_numbered_and_bounds_the_vector_loop_in_it,
         a_width_no_divisor_fits_runs_at_the_cap_and_one_tail_carries_the_sum_on,
         a_divisor_as_few_trips_as_the_cap_takes_no_tail,
         every_place_that_spills_is_halved_together, a_place_over_by_its_spilled_units_goes_where_they_fit,
         a_narrowing_that_relieved_none_goes_back_and_is_kept,
         a_spill_no_place_names_halves_the_widest_streams, no_place_to_narrow_halves_the_kernel,
         a_loop_whose_live_values_overfill_the_registers_starts_narrower,
         a_narrowing_that_spills_no_fewer_goes_on_and_the_fewest_spills_stand,
         the_retry_stops_after_max_retries_on_the_widths_that_spilled_least]

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
