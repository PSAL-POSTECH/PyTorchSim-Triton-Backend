"""TO_LLVM's postcondition (p13b): no op taking a vector to an integer or float scalar survives it,
read off the types; without the lane broadcast lowering a hoisted broadcast leaves one.
    python3 tests/passes/lane_scalar_guard_cases.py        exit 0 iff every case holds"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lower  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import (  # noqa: E402
    p13b_check_lane_values_stay_vector as p13b)

T = machine.target()

HOISTED = """func.func @f(%x: memref<4x64xf32, 1>, %m: memref<4xf32, 1>, %o: memref<4x64xf32, 1>) {
  affine.for %r = 0 to 4 {
    %s = affine.vector_load %m[%r] : memref<4xf32, 1>, vector<1xf32>
    %b = vector.broadcast %s : vector<1xf32> to vector<64xf32>
    affine.for %c = 0 to 4 {
      %v = affine.vector_load %x[%r, 0] : memref<4x64xf32, 1>, vector<64xf32>
      %y = arith.subf %v, %b : vector<64xf32>
      affine.vector_store %y, %o[%r, 0] : memref<4x64xf32, 1>, vector<64xf32>
    }
  }
  return
}
"""

EVERY_FORM = """func.func @f(%v: vector<1xf32>, %w: vector<4xf32>, %u: vector<4xi32>, %i: i64,
    %k: vector<1xi32>) -> (f32, f32, f32, i32, f32, vector<1xf32>, i32, f32) {
  %a = llvm.extractelement %v[%i : i64] : vector<1xf32>
  %b = vector.extract %w[0] : f32 from vector<4xf32>
  %c = vector.reduction <add>, %w : vector<4xf32> into f32
  %d = "llvm.intr.vector.reduce.add"(%u) : (vector<4xi32>) -> i32
  %e = llvm.extractelement %w[%i : i64] : vector<4xf32>
  %f = vector.extract_strided_slice %w {offsets = [1], sizes = [1], strides = [1]}
      : vector<4xf32> to vector<1xf32>
  %g = llvm.bitcast %k : vector<1xi32> to i32
  %h = "some.unnamed_op"(%w) : (vector<4xf32>) -> f32
  return %a, %b, %c, %d, %e, %f, %g, %h : f32, f32, f32, i32, f32, vector<1xf32>, i32, f32
}
"""


def _to_llvm(text):
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "k.mlir"), os.path.join(d, "l.mlir")
        with open(src, "w") as f:
            f.write(text)
        lower.to_llvm(src, out, target=T)
        with open(out) as f:
            return f.read()


def _lane_scalars(text):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        return [o.name for o in p13b.ops_making_a_vector_scalar(ir.Module.parse(text))]


def a_hoisted_broadcast_leaves_to_llvm_with_no_lane_value_made_scalar():
    got = _lane_scalars(_to_llvm(HOISTED))
    assert not got, got


def without_the_lane_broadcast_lowering_the_guard_refuses_the_extract():
    from pytorchsim_triton_compiler.lowering.passes import p12_lower_lane_broadcasts as p12
    kept = list(lower.TO_LLVM)
    lower.TO_LLVM[:] = [p for p in kept if p is not p12]
    try:
        _to_llvm(HOISTED)
    except p13b.LaneValueMadeScalar as e:
        assert "llvm.extractelement" in str(e) and "vector<1xf32>" in str(e), str(e)
        return
    finally:
        lower.TO_LLVM[:] = kept
    raise AssertionError("TO_LLVM without the lane broadcast lowering was not refused")


def the_guard_reads_types_so_a_wide_extract_and_an_unknown_op_are_refused_too():
    """Breaks if the guard goes back to a list of op names: the wide extract %e, the bitcast %g and
    the op no list could know, %h, all leave a scalar; %f stays a vector and is spared."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(EVERY_FORM)
        got = [o.name for o in p13b.ops_making_a_vector_scalar(m)]
    assert got == ["llvm.extractelement", "vector.extract", "vector.reduction",
                   "llvm.intr.vector.reduce.add", "llvm.extractelement", "llvm.bitcast",
                   "some.unnamed_op"], got


def the_guard_ends_to_llvm():
    assert lower.TO_LLVM[-1] is p13b, lower.TO_LLVM


CASES = [a_hoisted_broadcast_leaves_to_llvm_with_no_lane_value_made_scalar,
         without_the_lane_broadcast_lowering_the_guard_refuses_the_extract,
         the_guard_reads_types_so_a_wide_extract_and_an_unknown_op_are_refused_too,
         the_guard_ends_to_llvm]

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
