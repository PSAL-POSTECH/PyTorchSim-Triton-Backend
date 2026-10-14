"""torchsim-legalize leaves no constant or broadcast it stopped reading: the step-wide value a loop
cut replaced, and the constants of a scalar transcendental chain it reissued. On main both survive
unused into 09-custom.mlir (attention's vector<1> -> vector<128> broadcast beside its vector<64>).
    python3 tests/passes/legalize_dead_ops_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p06_legalize  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

T = machine.target()

CUT = """module {
  memref.global @row_spad_128lane : memref<128xf32, 1>
  memref.global @tile_spad_128lane : memref<128x128xf32, 1>
  func.func @f() {
    %r = memref.get_global @row_spad_128lane : memref<128xf32, 1>
    %t = memref.get_global @tile_spad_128lane : memref<128x128xf32, 1>
    affine.for %l = 0 to 128 {
      affine.for %i = 0 to 1 {
        %s = affine.vector_load %r[%i] {torchsim.lane_splat} : memref<128xf32, 1>, vector<1xf32>
        %b = vector.broadcast %s : vector<1xf32> to vector<128xf32>
        %m = affine.for %j = 0 to 128 step 128 iter_args(%a = %b) -> (vector<128xf32>) {
          %x = affine.vector_load %t[%i, %j] : memref<128x128xf32, 1>, vector<128xf32>
          %y = arith.maxnumf %x, %a : vector<128xf32>
          affine.yield %y : vector<128xf32>
        } {torchsim.reduce_loop, torchsim.vector_loop}
        %v = vector.reduction <maxnumf>, %m : vector<128xf32> into f32
        %w = vector.broadcast %v : f32 to vector<1xf32>
        affine.vector_store %w, %r[%i] : memref<128xf32, 1>, vector<1xf32>
      }
    } {torchsim.vlane_loop = true}
    return
  }
}
"""

TRANSCENDENTAL = """module {
  memref.global @row_spad_128lane : memref<128xf32, 1>
  func.func @f() {
    %c = arith.constant dense<8.0> : vector<1xf32>
    %r = memref.get_global @row_spad_128lane : memref<128xf32, 1>
    affine.for %l = 0 to 128 {
      affine.for %i = 0 to 1 {
        %s = affine.vector_load %r[%i] : memref<128xf32, 1>, vector<1xf32>
        %d = arith.divf %s, %c : vector<1xf32>
        %q = math.rsqrt %d : vector<1xf32>
        affine.vector_store %q, %r[%i] : memref<128xf32, 1>, vector<1xf32>
      } {torchsim.vector_loop}
    } {torchsim.vlane_loop = true}
    return
  }
}
"""


def _unread_after_legalize(text):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(text)
        p06_legalize.run(m, lanes=T.vector_lanes, vlen=T.vlen_bits, lmul=8)
        assert m.operation.verify()
        return [f"{o.name} -> {o.results[0].type}" for o in walk(m.operation)
                if o.name in ("arith.constant", "vector.broadcast")
                and not any(True for _ in o.results[0].uses)]


def a_loop_cut_leaves_no_step_wide_broadcast_it_replaced():
    got = _unread_after_legalize(CUT)
    assert not got, got


def a_reissued_scalar_transcendental_leaves_no_constant_of_its_chain():
    got = _unread_after_legalize(TRANSCENDENTAL)
    assert not got, got


CASES = [a_loop_cut_leaves_no_step_wide_broadcast_it_replaced,
         a_reissued_scalar_transcendental_leaves_no_constant_of_its_chain]

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
