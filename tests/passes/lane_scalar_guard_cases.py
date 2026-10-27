"""TO_LLVM's postcondition (verify-lane-values-stay-vector), off types: no op takes a vector to a scalar
(a hoisted broadcast without the target's lane patterns leaves one); a scratchpad pointer is in no vector,
two-memory access, scalar write or sub-byte vector. Past llc, the object holds no vector-to-scalar move
(elf/lane_moves): build_elf refuses one.
    python3 tests/passes/lane_scalar_guard_cases.py        exit 0 iff every case holds"""
import os
import subprocess
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.driver import machine  # noqa: E402
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.passes import lower  # noqa: E402
from pytorchsim_triton_compiler.passes.analysis import verify_lane_values_stay_vector
from pytorchsim_triton_compiler.codegen import build  # noqa: E402
from pytorchsim_triton_compiler.passes.analysis import lane_moves  # noqa: E402

T = machine.subtarget()

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
        lower.to_llvm(src, out, subtarget=T)
        with open(out) as f:
            return f.read()


def _lane_scalars(text):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        scalar = verify_lane_values_stay_vector.ops_making_a_vector_scalar(ir.Module.parse(text))
        return [o.name for o in scalar]


def a_hoisted_broadcast_leaves_to_llvm_with_no_lane_value_made_scalar():
    got = _lane_scalars(_to_llvm(HOISTED))
    assert not got, got


def without_the_lane_patterns_the_guard_refuses_the_extract():
    from pytorchsim_triton_compiler.passes.conversion import pytorchsim_to_llvm as conv
    lower_lane_values = conv.lower_lane_values
    conv.lower_lane_values = lambda module, subtarget: None
    try:
        _to_llvm(HOISTED)
    except verify_lane_values_stay_vector.LaneValueMadeScalar as e:
        assert "llvm.extractelement" in str(e) and "vector<1xf32>" in str(e), str(e)
        return
    finally:
        conv.lower_lane_values = lower_lane_values
    raise AssertionError("TO_LLVM without the lane patterns was not refused")


def the_guard_reads_types_so_a_wide_extract_and_an_unknown_op_are_refused_too():
    """Breaks if the guard goes back to a list of op names: the wide extract %e, the bitcast %g and
    the op no list could know, %h, all leave a scalar; %f stays a vector and is spared."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(EVERY_FORM)
        got = [o.name for o in verify_lane_values_stay_vector.ops_making_a_vector_scalar(m)]
    assert got == ["llvm.extractelement", "vector.extract", "vector.reduction",
                   "llvm.intr.vector.reduce.add", "llvm.extractelement", "llvm.bitcast",
                   "some.unnamed_op"], got


def the_guard_ends_to_llvm():
    """It is TO_LLVM's invariant check, run once the list is done, and no pass of it."""
    assert lower.LLVM_CHECKS == [verify_lane_values_stay_vector], lower.LLVM_CHECKS
    assert verify_lane_values_stay_vector not in lower.TO_LLVM, lower.TO_LLVM


ROW_SUM = """func.func @f(%x: memref<4x64xf32, 1>, %o: memref<4x64xf32, 1>) {
  %z = arith.constant dense<-0.0> : vector<1xf32>
  affine.for %r = 0 to 4 {
    %v = affine.vector_load %x[%r, 0] : memref<4x64xf32, 1>, vector<64xf32>
    %w = vector.shape_cast %v : vector<64xf32> to vector<1x64xf32>
    %s = vector.multi_reduction <add>, %w, %z [1] : vector<1x64xf32> to vector<1xf32>
    %b = vector.broadcast %s : vector<1xf32> to vector<64xf32>
    %y = arith.subf %v, %b : vector<64xf32>
    affine.vector_store %y, %o[%r, 0] : memref<4x64xf32, 1>, vector<64xf32>
  }
  return
}
"""

MADE_SCALAR = """define float @ext3(<8 x float> %v) {
  %e = extractelement <8 x float> %v, i32 3
  ret float %e
}
define i64 @ext0(<8 x i64> %v) {
  %e = extractelement <8 x i64> %v, i32 0
  ret i64 %e
}
define i1 @any(<8 x i1> %v) {
  %e = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> %v)
  ret i1 %e
}
define float @fmaximum(<8 x float> %v) {
  %e = call float @llvm.vector.reduce.fmaximum.v8f32(<8 x float> %v)
  ret float %e
}
declare i1 @llvm.vector.reduce.or.v8i1(<8 x i1>)
declare float @llvm.vector.reduce.fmaximum.v8f32(<8 x float>)
"""

VFREDUSUM, VFREDOSUM = 0b000001, 0b000011


def _llc_obj(ll_path, obj):
    subprocess.run([toolchain.LLC, f"-march={T.llc_march()}", "-O3", f"-mattr={T.llc_mattr()}",
                    f"--riscv-v-vector-bits-max={T.vlen_bits}", "-filetype=obj", ll_path, "-o", obj],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _object_of_mlir(text, d):
    """`text` through TO_LLVM, mlir-translate and llc, as the build runs them: the object's path."""
    src, out, ll, obj = (os.path.join(d, n) for n in ("k.mlir", "l.mlir", "k.ll", "k.o"))
    with open(src, "w") as f:
        f.write(text)
    lower.to_llvm(src, out, subtarget=T)
    subprocess.run([toolchain.MLIR_TRANSLATE, "-mlir-to-llvmir", out, "-o", ll], check=True)
    _llc_obj(ll, obj)
    return obj


def _opfvv_funct6(obj):
    return [w >> 26 for _, _, w in lane_moves.instruction_words(obj)
            if w is not None and w & 0x7F == lane_moves.OP_V and (w >> 12) & 7 == lane_moves.OPFVV]


def a_hoisted_broadcast_is_built_with_no_lane_value_in_a_scalar_register():
    """Breaks if llc lowers the standard splat of element 0 through vfmv.f.s again."""
    with tempfile.TemporaryDirectory() as d:
        got = lane_moves.vector_to_scalar_moves(_object_of_mlir(HOISTED, d))
    assert not got, got


def a_lane_row_reduction_is_one_unordered_vfred_kept_in_a_register():
    """Breaks if a lane's row sum stops being one vfredusum -- an ordered vfredosum, or a walk one
    element a step -- or its answer moves to a scalar register."""
    with tempfile.TemporaryDirectory() as d:
        obj = _object_of_mlir(ROW_SUM, d)
        moves, funct6 = lane_moves.vector_to_scalar_moves(obj), _opfvv_funct6(obj)
    assert not moves, moves
    assert funct6.count(VFREDUSUM) == 1 and VFREDOSUM not in funct6, funct6


def every_vector_to_scalar_move_is_found_in_the_object():
    """Breaks if the decoder misses one of the four forms: an extract to f and to x, an i1 any
    (vcpop) and a NaN-propagating fmaximum reduction, which llc takes through both."""
    with tempfile.TemporaryDirectory() as d:
        ll, obj = os.path.join(d, "k.ll"), os.path.join(d, "k.o")
        with open(ll, "w") as f:
            f.write(MADE_SCALAR)
        _llc_obj(ll, obj)
        got = sorted({(fn, m) for fn, _, m in lane_moves.vector_to_scalar_moves(obj)})
    assert got == [("any", "vcpop.m"), ("ext0", "vmv.x.s"), ("ext3", "vfmv.f.s"),
                   ("fmaximum", "vcpop.m"), ("fmaximum", "vfmv.f.s")], got


class _Linked(Exception):
    pass


def _build_with_object_of(made, d):
    """build_elf with llc's object made from the LLVM IR file `made`, the other tools recorded, the ones it
    starts beside another (start_tool) among them; the link raises _Linked."""
    def run(ctx, stage_name, cmd, **kw):
        for a in cmd:
            if a.startswith("-pass-remarks-output="):
                open(a.split("=", 1)[1], "w").close()
        if cmd[0] == toolchain.LLC and "-filetype=obj" in cmd:
            _llc_obj(made, cmd[cmd.index("-o") + 1])
        if cmd[0] == toolchain.RISCV_GCC:
            raise _Linked()
    saved = build.run_tool, build.start_tool, build.finish_tool
    build.run_tool = run
    build.start_tool = lambda ctx, cmd, **kw: types.SimpleNamespace(wait=lambda: run(ctx, "binary", cmd))
    build.finish_tool = lambda ctx, stage, running: running.wait()
    ctx = types.SimpleNamespace(kernel_path="k.py", spec=None, log=lambda *_: None)
    f = {k: os.path.join(d, k) for k in ("k.mlir", "k.ll", "k.s", "k.o", "w.c", "k.elf")}
    for k in ("k.ll", "k.s"):
        open(f[k], "w").close()
    try:
        build.build_elf(ctx, "binary", "t", f["k.mlir"], f["k.ll"], f["k.o"], f["k.s"], f["w.c"],
                        f["k.elf"])
    finally:
        build.run_tool, build.start_tool, build.finish_tool = saved


def build_elf_refuses_an_object_that_moves_a_lane_value_to_a_scalar_register():
    """Breaks if build_elf stops reading llc's object: refused before the link, as a named failure."""
    from pytorchsim_triton_compiler.driver.stage import StageError
    with tempfile.TemporaryDirectory() as d:
        made = os.path.join(d, "m.ll")
        with open(made, "w") as m:
            m.write(MADE_SCALAR)
        try:
            _build_with_object_of(made, d)
        except lane_moves.LaneValueMadeScalar as e:
            assert isinstance(e, StageError) and len(e.moves) == 5, (str(e), e.moves)
            return
        except _Linked:
            raise AssertionError("an object with vector-to-scalar moves was linked")
    raise AssertionError("build_elf did not refuse the object")


ARGS = ("%p: !llvm.ptr<1>, %q: !llvm.ptr<1>, %d: !llvm.ptr, %g: vector<2x!llvm.ptr<1>>, "
        "%k: vector<8xi1>, %k2: vector<2xi1>, %n: i64")


def _refusal(body):
    """The exception verify_lane_values_stay_vector.run raises on `body` as an llvm.func's, or None."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse("llvm.func @ext(!llvm.ptr<1>, !llvm.ptr<1>, i64)\n"
                            f"llvm.func @f({ARGS}) {{\n{body}\n  llvm.return\n}}\n")
        try:
            verify_lane_values_stay_vector.run(m)
        except Exception as e:  # noqa: BLE001
            return e
    return None


def _refused_as(body, why):
    got = _refusal(body)
    assert got is not None, f"not refused: {body.strip()}"
    assert getattr(got, "reason", None) == getattr(verify_lane_values_stay_vector, why), (
        why, type(got).__name__, str(got)[:200])


def a_memory_intrinsic_on_the_scratchpad_is_refused():
    """Breaks if convert-pytorchsim-to-llvm stops reading the effects stated on a second pointer: memcpy and
    memmove either way, and the inline form, move bytes no value carries."""
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
    """Breaks if convert-pytorchsim-to-llvm stops reading element widths: <8 x i1> stored, loaded, or
    masked-stored under an i1 mask is bit-packed where the buffer holds a byte per element."""
    for body in ["  %v = llvm.load %d : !llvm.ptr -> vector<8xi1>\n"
                 "  llvm.store %v, %p : vector<8xi1>, !llvm.ptr<1>",
                 "  %v = llvm.load %p : !llvm.ptr<1> -> vector<16xi1>",
                 "  %v = llvm.load %d : !llvm.ptr -> vector<8xi1>\n"
                 "  llvm.intr.masked.store %v, %p, %k {alignment = 1 : i32} : "
                 "vector<8xi1>, vector<8xi1> into !llvm.ptr<1>",
                 '  %v = "some.read"(%p) : (!llvm.ptr<1>) -> vector<8xi4>']:
        _refused_as(body, "SUB_BYTE")


def a_vector_of_scratchpad_pointers_is_refused():
    """Breaks if convert-pytorchsim-to-llvm stops reading pointer vectors: a gather, a scatter, or the address
    arithmetic that makes one gives each lane an address of its own."""
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
    """Breaks if convert-pytorchsim-to-llvm stops reading what a stated write carries: a scalar, a pointer, or
    a memset's byte stored through ptr<1> reaches lane 0's bank alone."""
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
         without_the_lane_patterns_the_guard_refuses_the_extract,
         the_guard_reads_types_so_a_wide_extract_and_an_unknown_op_are_refused_too,
         the_guard_ends_to_llvm,
         a_hoisted_broadcast_is_built_with_no_lane_value_in_a_scalar_register,
         a_lane_row_reduction_is_one_unordered_vfred_kept_in_a_register,
         every_vector_to_scalar_move_is_found_in_the_object,
         build_elf_refuses_an_object_that_moves_a_lane_value_to_a_scalar_register,
         a_memory_intrinsic_on_the_scratchpad_is_refused,
         a_sub_byte_vector_through_a_scratchpad_pointer_is_refused,
         a_vector_of_scratchpad_pointers_is_refused,
         a_scalar_write_through_a_scratchpad_pointer_is_refused]

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
