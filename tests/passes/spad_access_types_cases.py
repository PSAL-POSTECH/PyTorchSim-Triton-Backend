"""TO_LLVM's postcondition (p13b) on scratchpad pointers, read off types and stated effects: no vector
of them, no access through one and another pointer at once, no scalar write, no sub-byte vectors only.
    python3 tests/passes/spad_access_types_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import (  # noqa: E402
    p13b_check_lane_values_stay_vector as p13b)

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


def what_a_lane_bank_takes_is_spared():
    """Vector and masked accesses of whole-byte elements, scalar loads, the vp forms, anything off the
    scratchpad, and a call passing a buffer's two pointers (it states no effect) pass."""
    body = ("  %v = llvm.load %p {alignment = 4 : i64} : !llvm.ptr<1> -> vector<8xf32>\n"
            "  llvm.store %v, %q : vector<8xf32>, !llvm.ptr<1>\n"
            "  %z = llvm.mlir.zero : vector<8xf32>\n"
            "  %m = llvm.intr.masked.load %p, %k, %z {alignment = 4 : i32} : "
            "(!llvm.ptr<1>, vector<8xi1>, vector<8xf32>) -> vector<8xf32>\n"
            "  llvm.intr.masked.store %m, %q, %k {alignment = 4 : i32} : "
            "vector<8xf32>, vector<8xi1> into !llvm.ptr<1>\n"
            "  %s = llvm.load %p : !llvm.ptr<1> -> f32\n"
            "  %b = llvm.load %p : !llvm.ptr<1> -> i1\n"
            "  %c = llvm.load %p : !llvm.ptr<1> -> vector<1xi8>\n"
            "  %e = llvm.mlir.constant(8 : i32) : i32\n"
            '  llvm.call_intrinsic "llvm.vp.store"(%v, %q, %k, %e) : '
            "(vector<8xf32>, !llvm.ptr<1>, vector<8xi1>, i32) -> ()\n"
            "  %w = llvm.load %d : !llvm.ptr -> vector<8xi1>\n"
            "  llvm.store %w, %d : vector<8xi1>, !llvm.ptr\n"
            '  "llvm.intr.memcpy"(%d, %d, %n) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()\n'
            "  %a = llvm.getelementptr %p[%n] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32\n"
            "  llvm.store %p, %d : !llvm.ptr<1>, !llvm.ptr\n"
            "  llvm.call @ext(%p, %q, %n) : (!llvm.ptr<1>, !llvm.ptr<1>, i64) -> ()")
    got = _refusal(body)
    assert got is None, f"{type(got).__name__}: {str(got)[:300]}"


CASES = [a_memory_intrinsic_on_the_scratchpad_is_refused,
         a_sub_byte_vector_through_a_scratchpad_pointer_is_refused,
         a_vector_of_scratchpad_pointers_is_refused,
         a_scalar_write_through_a_scratchpad_pointer_is_refused,
         what_a_lane_bank_takes_is_spared]

if __name__ == "__main__":
    bad = 0
    for c in CASES:
        try:
            c()
            print(f"ok   {c.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {c.__name__}: {type(e).__name__}: {str(e)[:300]}")
    sys.exit(1 if bad else 0)
