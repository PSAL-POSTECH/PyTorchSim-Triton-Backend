"""torchsim-legalize leaves no constant or broadcast it stopped reading -- the step-wide value a loop
cut replaced, the constants of a reissued scalar chain -- and hands its retiring only values of ops
still in the function. python3 tests/passes/legalize_dead_ops_cases.py; exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p06_legalize  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime"))

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

CHAIN = """module {
  memref.global @row_spad_128lane : memref<128xf32, 1>
  func.func @f() {
    %n = arith.constant 128.0 : f32
    %eps = arith.constant 1.0e-5 : f32
    %r = memref.get_global @row_spad_128lane : memref<128xf32, 1>
    affine.for %l = 0 to 128 {
      affine.for %i = 0 to 1 {
        %v = affine.load %r[%i] : memref<128xf32, 1>
        %a = arith.divf %v, %n : f32
        %b = arith.addf %a, %eps : f32
        %q = math.exp %b : f32
        %w = vector.broadcast %q : f32 to vector<1xf32>
        affine.vector_store %w, %r[%i] : memref<128xf32, 1>, vector<1xf32>
      }
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


def a_reissued_scalar_chain_retires_only_values_of_ops_still_there():
    from pytorchsim_triton_compiler.lowering.passes.lib_ir import op_of
    kept, gone, box = p06_legalize._retire, [], {}

    def spy(vals):
        live = set(walk(box["m"].operation))
        gone.extend(op_of(v.owner).name for v in vals
                    if hasattr(op_of(v.owner), "name") and op_of(v.owner) not in live)
        return kept(vals)
    p06_legalize._retire = spy
    try:
        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        with ctx, ir.Location.unknown():
            box["m"] = ir.Module.parse(CHAIN)
            p06_legalize.run(box["m"], lanes=T.vector_lanes, vlen=T.vlen_bits, lmul=8)
            assert box["m"].operation.verify()
    finally:
        p06_legalize._retire = kept
    assert not gone, gone


def _legalized_ops(text, as_allocs):
    """The op names p06 leaves, but a buffer's own (get_global, alloc, dealloc, global)."""
    from alloc_twin import to_allocs
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(text)
        if as_allocs:
            to_allocs(m)
        p06_legalize.run(m, lanes=T.vector_lanes, vlen=T.vlen_bits, lmul=8)
        assert m.operation.verify()
        life = ("memref.get_global", "memref.alloc", "memref.dealloc", "memref.global")
        return [o.name for o in walk(m.operation) if o.name not in life]


def a_reduction_stored_into_an_allocated_buffer_legalizes_as_a_globals():
    """Breaks if p06 answers a reduction stored into an alloc differently from the same reduction
    stored into a get_global (#230): the alloc twin of CUT must leave the global form's ops."""
    want = _legalized_ops(CUT, False)
    got = _legalized_ops(CUT, True)
    assert got == want, [x for x in got if x not in want] or got


def a_reissued_scalar_chain_is_one_element_wide_and_stored_as_is():
    """The twins are vector<1>, so the store takes the exp's twin itself: no lane-0 slice of a
    wider twin and no re-broadcast."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(CHAIN)
        p06_legalize.run(m, lanes=T.vector_lanes, vlen=T.vlen_bits, lmul=8)
        assert m.operation.verify()
        ops = list(walk(m.operation))
        exp = [o for o in ops if o.name == "math.exp"]
        store = [o for o in ops if o.name == "affine.vector_store"]
        assert len(exp) == 1 and ir.VectorType(exp[0].results[0].type).shape == [1], exp
        assert store[-1].operands[0] == exp[0].results[0], store[-1]


CASES = [a_loop_cut_leaves_no_step_wide_broadcast_it_replaced,
         a_reissued_scalar_transcendental_leaves_no_constant_of_its_chain,
         a_reissued_scalar_chain_retires_only_values_of_ops_still_there,
         a_reduction_stored_into_an_allocated_buffer_legalizes_as_a_globals,
         a_reissued_scalar_chain_is_one_element_wide_and_stored_as_is]

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
