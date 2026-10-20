"""A spill narrows the loops llc names, not the kernel: legalize numbers every loop by its
location, a loop's width follows the LMUL a retry gave it or a loop around it, and the retry halves
the loops llc's remarks name -- the kernel's LMUL only when it names none, or none can go lower.
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


def _ctx():
    return SimpleNamespace(lmul=LMUL, loop_lmul={}, dot_lmul={}, dot_widths={})


def _spill(loops=(), dots=(), outside=0):
    return SimpleNamespace(loops=list(loops), dots=list(dots), outside=outside)


def a_spill_in_named_loops_halves_those_loops():
    ctx = _ctx()
    got = pipeline._narrow_for(ctx, _spill(loops=[2, 5]))
    assert got[0] == "transform" and ctx.loop_lmul == {2: 4, 5: 4} and ctx.lmul == LMUL, (got, ctx)


def a_spill_in_no_named_loop_halves_the_kernel():
    ctx = _ctx()
    pipeline._narrow_for(ctx, _spill())
    assert ctx.lmul == 4 and ctx.loop_lmul == {}, ctx


def a_named_loop_at_one_falls_back_to_the_kernel():
    ctx = _ctx()
    ctx.loop_lmul = {5: 1}
    pipeline._narrow_for(ctx, _spill(loops=[5]))
    assert ctx.lmul == 4, ctx


def a_spill_in_a_named_stream_halves_that_stream_only_from_lower():
    ctx = _ctx()
    ctx.dot_widths = {1: 4, 2: 8}
    got = pipeline._narrow_for(ctx, _spill(dots=[2]))
    assert got[0] == "lower" and ctx.dot_lmul == {2: 4} and ctx.loop_lmul == {} and ctx.lmul == LMUL, \
        (got, ctx)


def a_spill_no_loop_or_stream_names_halves_the_widest_streams():
    ctx = _ctx()
    ctx.dot_widths = {1: 4, 2: 8, 3: 8}
    got = pipeline._narrow_for(ctx, _spill(outside=22))
    assert got[0] == "lower" and ctx.dot_lmul == {2: 4, 3: 4} and ctx.lmul == LMUL, (got, ctx)


def a_spill_no_loop_or_stream_names_with_every_stream_at_one_halves_the_kernel():
    ctx = _ctx()
    ctx.dot_widths = {1: 1}
    got = pipeline._narrow_for(ctx, _spill(outside=3))
    assert got[0] == "transform" and ctx.lmul == 4 and ctx.dot_lmul == {}, (got, ctx)


CASES = [every_loop_is_numbered_by_its_location, a_narrowed_loop_narrows_itself_and_what_it_holds_only,
         a_spill_in_named_loops_halves_those_loops, a_spill_in_no_named_loop_halves_the_kernel,
         a_named_loop_at_one_falls_back_to_the_kernel,
         a_spill_in_a_named_stream_halves_that_stream_only_from_lower,
         a_spill_no_loop_or_stream_names_halves_the_widest_streams,
         a_spill_no_loop_or_stream_names_with_every_stream_at_one_halves_the_kernel]

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
