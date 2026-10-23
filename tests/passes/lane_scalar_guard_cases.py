"""TO_LLVM's postcondition (p13b), off types: no op takes a vector to a scalar (a hoisted broadcast without
p12 leaves one); a scratchpad pointer is in no vector, two-memory access, scalar write or sub-byte vector.
    python3 tests/passes/lane_scalar_guard_cases.py        exit 0 iff every case holds"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from case_runner import run  # noqa: E402
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


ARGS = ("%p: !llvm.ptr<1>, %q: !llvm.ptr<1>, %d: !llvm.ptr, %g: vector<2x!llvm.ptr<1>>, "
        "%k: vector<8xi1>, %k2: vector<2xi1>, %n: i64")


def _refusal(body):
    """The exception p13b.run raises on `body` as an llvm.func's, or None."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse("llvm.func @ext(!llvm.ptr<1>, !llvm.ptr<1>, i64)\n"
                            f"llvm.func @f({ARGS}) {{\n{body}\n  llvm.return\n}}\n")
        try:
            p13b.run(m)
        except Exception as e:  # noqa: BLE001
            return e
    return None


def _refused_as(body, why):
    got = _refusal(body)
    assert got is not None, f"not refused: {body.strip()}"
    assert getattr(got, "reason", None) == getattr(p13b, why), (why, type(got).__name__, str(got)[:200])


def a_memory_intrinsic_on_the_scratchpad_is_refused():
    """Breaks if p13b stops reading the effects stated on a second pointer: memcpy and memmove either
    way, and the inline form, move bytes no value carries."""
    for body in ['  "llvm.intr.memcpy"(%p, %d, %n) <{isVolatile = false}> : '
                 '(!llvm.ptr<1>, !llvm.ptr, i64) -> ()',
                 '  "llvm.intr.memcpy"(%d, %p, %n) <{isVolatile = false}> : '
                 '(!llvm.ptr, !llvm.ptr<1>, i64) -> ()',
                 '  "llvm.intr.memmove"(%p, %q, %n) <{isVolatile = false}> : '
                 '(!llvm.ptr<1>, !llvm.ptr<1>, i64) -> ()',
                 '  "llvm.intr.memcpy.inline"(%p, %d) <{isVolatile = false, len = 8 : i64}> : '
                 '(!llvm.ptr<1>, !llvm.ptr) -> ()']:
        _refused_as(body, "TWO_MEMORIES")


def a_sub_byte_vector_through_a_scratchpad_pointer_is_refused():
    """Breaks if p13b stops reading element widths: <8 x i1> stored, loaded, or masked-stored under an
    i1 mask is bit-packed where the buffer holds a byte per element."""
    for body in ["  %v = llvm.load %d : !llvm.ptr -> vector<8xi1>\n"
                 "  llvm.store %v, %p : vector<8xi1>, !llvm.ptr<1>",
                 "  %v = llvm.load %p : !llvm.ptr<1> -> vector<16xi1>",
                 "  %v = llvm.load %d : !llvm.ptr -> vector<8xi1>\n"
                 "  llvm.intr.masked.store %v, %p, %k {alignment = 1 : i32} : "
                 "vector<8xi1>, vector<8xi1> into !llvm.ptr<1>",
                 '  %v = "some.read"(%p) : (!llvm.ptr<1>) -> vector<8xi4>']:
        _refused_as(body, "SUB_BYTE")


def a_vector_of_scratchpad_pointers_is_refused():
    """Breaks if p13b stops reading pointer vectors: a gather, a scatter, or the address arithmetic
    that makes one gives each lane an address of its own."""
    gather = ("  %z = llvm.mlir.zero : vector<2xf32>\n"
              "  %v = llvm.intr.masked.gather %g, %k2, %z {alignment = 4 : i32} : "
              "(vector<2x!llvm.ptr<1>>, vector<2xi1>, vector<2xf32>) -> vector<2xf32>")
    for body in [gather,
                 gather + "\n  llvm.intr.masked.scatter %v, %g, %k2 {alignment = 4 : i32} : "
                          "vector<2xf32>, vector<2xi1> into vector<2x!llvm.ptr<1>>",
                 "  %i = llvm.mlir.constant(dense<[0, 1]> : vector<2xi64>) : vector<2xi64>\n"
                 "  %a = llvm.getelementptr %p[%i] : (!llvm.ptr<1>, vector<2xi64>) -> "
                 "vector<2x!llvm.ptr<1>>, f32"]:
        _refused_as(body, "INDEXED")


def a_scalar_write_through_a_scratchpad_pointer_is_refused():
    """Breaks if p13b stops reading what a stated write carries: a scalar, a pointer, or a memset's
    byte stored through ptr<1> reaches lane 0's bank alone."""
    for body in ["  %s = llvm.load %d : !llvm.ptr -> f32\n  llvm.store %s, %p : f32, !llvm.ptr<1>",
                 "  %s = llvm.load %d : !llvm.ptr -> i1\n  llvm.store volatile %s, %p : i1, !llvm.ptr<1>",
                 "  llvm.store %q, %p : !llvm.ptr<1>, !llvm.ptr<1>",
                 "  %b = llvm.mlir.constant(7 : i8) : i8\n"
                 '  "llvm.intr.memset"(%p, %b, %n) <{isVolatile = false}> : (!llvm.ptr<1>, i8, i64) -> ()',
                 "  %b = llvm.mlir.constant(7 : i8) : i8\n"
                 '  "llvm.intr.memset.inline"(%p, %b) <{isVolatile = false, len = 8 : i64}> : '
                 "(!llvm.ptr<1>, i8) -> ()"]:
        _refused_as(body, "SCALAR_WRITE")


CASES = [a_hoisted_broadcast_leaves_to_llvm_with_no_lane_value_made_scalar,
         without_the_lane_broadcast_lowering_the_guard_refuses_the_extract,
         the_guard_reads_types_so_a_wide_extract_and_an_unknown_op_are_refused_too,
         the_guard_ends_to_llvm,
         a_memory_intrinsic_on_the_scratchpad_is_refused,
         a_sub_byte_vector_through_a_scratchpad_pointer_is_refused,
         a_vector_of_scratchpad_pointers_is_refused,
         a_scalar_write_through_a_scratchpad_pointer_is_refused]

if __name__ == "__main__":
    sys.exit(run(CASES))
