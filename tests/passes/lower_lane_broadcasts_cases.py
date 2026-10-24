"""pytorchsim-lower-lane-broadcasts: a vector<1>'s broadcast hoisted out of the loop that uses it
reaches llc as a vrgather, never as vfmv.f.s / vmv.x.s; without the pass, TO_LLVM gives exactly that.
    python3 tests/passes/lower_lane_broadcasts_cases.py        exit 0 iff every case holds"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lower  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

T = machine.target()
LANE_EXTRACT = ("vfmv.f.s", "vmv.x.s")
SUB = {"f": "arith.subf", "b": "arith.subf", "i": "arith.subi"}


def _hoisted(elem="f32", n=64, use=None):
    row, tile = f"memref<4x{elem}, 1>", f"memref<4x{n}x{elem}, 1>"
    use = use or f"%y = {SUB[elem[0]]} %v, %b : vector<{n}x{elem}>"
    return (f"func.func @f(%x: {tile}, %m: {row}, %o: {tile}) {{\n"
            "  affine.for %r = 0 to 4 {\n"
            f"    %s = affine.vector_load %m[%r] : {row}, vector<1x{elem}>\n"
            f"    %b = vector.broadcast %s : vector<1x{elem}> to vector<{n}x{elem}>\n"
            "    affine.for %c = 0 to 4 {\n"
            f"      %v = affine.vector_load %x[%r, 0] : {tile}, vector<{n}x{elem}>\n"
            f"      {use}\n"
            f"      affine.vector_store %y, %o[%r, 0] : {tile}, vector<{n}x{elem}>\n"
            "    }\n  }\n  return\n}\n")


def _asm(text):
    with tempfile.TemporaryDirectory() as d:
        src, out, ll, s = (os.path.join(d, f) for f in ("k.mlir", "l.mlir", "k.ll", "k.s"))
        with open(src, "w") as f:
            f.write(text)
        lower.to_llvm(src, out, target=T)
        subprocess.run([toolchain.MLIR_TRANSLATE, "-mlir-to-llvmir", out, "-o", ll], check=True)
        subprocess.run([toolchain.LLC, "-relocation-model=pic", "-march=" + T.llc_march(), "-O3",
                        "-mattr=" + T.llc_mattr(), ll, "-o", s], check=True)
        with open(s) as f:
            return [ln.split()[0] for ln in f
                    if ln.strip() and not ln.lstrip().startswith((".", "#")) and ":" not in ln]


def _gathers_and_never_extracts(text, what):
    ops = _asm(text)
    assert not set(LANE_EXTRACT) & set(ops), (what, sorted(set(LANE_EXTRACT) & set(ops)))
    assert "vrgather.vi" in ops, (what, ops)


def _parsed(text):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        return ir.Module.parse(text)


def _refused(text, *said):
    from pytorchsim_triton_compiler.lowering.passes import p12_lower_lane_broadcasts as p12
    m = _parsed(text)
    with m.context, ir.Location.unknown():
        try:
            p12.run(m, target=T)
        except p12.LaneBroadcastRefused as e:
            assert all(w in str(e) for w in said), (said, str(e))
            return
    raise AssertionError(f"not refused: {m}")


def a_hoisted_lane_broadcast_is_a_vrgather_for_every_element_type():
    for e in ("f32", "f16", "bf16", "i32", "i8", "i64"):
        _gathers_and_never_extracts(_hoisted(e, n=32), e)


def a_width_that_is_no_power_of_two_is_gathered_at_the_next_and_narrowed():
    _gathers_and_never_extracts(_hoisted("f32", n=48), "48xf32")


def an_i1_lane_value_is_gathered_as_i8():
    sel = ("%y = arith.select %k, %v, %x0 : vector<64xi1>, vector<64xf32>")
    text = _hoisted("f32", use=sel).replace(
        "%s = affine.vector_load %m[%r] : memref<4xf32, 1>, vector<1xf32>",
        "%s0 = affine.vector_load %m[%r] : memref<4xf32, 1>, vector<1xf32>\n"
        "    %z = arith.constant dense<0.0> : vector<1xf32>\n"
        "    %s = arith.cmpf ogt, %s0, %z : vector<1xf32>\n"
        "    %x0 = arith.constant dense<0.0> : vector<64xf32>").replace(
        "%b = vector.broadcast %s : vector<1xf32> to vector<64xf32>",
        "%k = vector.broadcast %s : vector<1xi1> to vector<64xi1>")
    _gathers_and_never_extracts(text, "i1")


def an_index_lane_value_is_gathered_as_i64():
    use = ("%w = arith.index_cast %b : vector<64xindex> to vector<64xi64>\n"
           "      %y = arith.addi %v, %w : vector<64xi64>")
    text = _hoisted("i64", n=32, use=use.replace("64x", "32x")).replace(
        "%b = vector.broadcast %s : vector<1xi64> to vector<32xi64>",
        "%i = arith.index_cast %s : vector<1xi64> to vector<1xindex>\n"
        "    %b = vector.broadcast %i : vector<1xindex> to vector<32xindex>")
    _gathers_and_never_extracts(text, "index")


def an_unused_lane_broadcast_is_erased_whatever_its_width():
    from pytorchsim_triton_compiler.lowering.passes import p12_lower_lane_broadcasts as p12
    m = _parsed("func.func @f(%s: vector<1xf32>) {\n"
                "  %b = vector.broadcast %s : vector<1xf32> to vector<4096xf32>\n"
                "  return\n}\n")
    with m.context, ir.Location.unknown():
        p12.run(m, target=T)
    assert "vector.broadcast" not in {o.name for o in walk(m.operation)}, m


def a_used_lane_broadcast_above_one_register_group_is_refused():
    n = 2 * 8 * T.vlen_bits // 32
    _refused("func.func @f(%%s: vector<1xf32>) -> vector<%dxf32> {\n"
             "  %%b = vector.broadcast %%s : vector<1xf32> to vector<%dxf32>\n"
             "  return %%b : vector<%dxf32>\n}\n" % (n, n, n), "LMUL-8", str(n))


def a_lane_broadcast_to_rank_two_is_refused():
    _refused("func.func @f(%s: vector<1xf32>) -> vector<4x8xf32> {\n"
             "  %b = vector.broadcast %s : vector<1xf32> to vector<4x8xf32>\n"
             "  return %b : vector<4x8xf32>\n}\n", "rank 2")


def a_lane_value_with_no_integer_view_is_refused():
    _refused("func.func @f(%s: vector<1xi4>) -> vector<8xi4> {\n"
             "  %b = vector.broadcast %s : vector<1xi4> to vector<8xi4>\n"
             "  return %b : vector<8xi4>\n}\n", "i4")


def a_scalar_broadcast_is_uniform_and_left_alone():
    from pytorchsim_triton_compiler.lowering.passes import p12_lower_lane_broadcasts as p12
    m = _parsed("func.func @f(%s: f32) -> vector<64xf32> {\n"
                "  %b = vector.broadcast %s : f32 to vector<64xf32>\n"
                "  return %b : vector<64xf32>\n}\n")
    with m.context, ir.Location.unknown():
        p12.run(m, target=T)
    assert [o.name for o in walk(m.operation)].count("vector.broadcast") == 1, m


def the_pass_opens_to_llvm():
    from pytorchsim_triton_compiler.lowering.passes import p11b_lower_runtime_attributes as p11b
    from pytorchsim_triton_compiler.lowering.passes import p12_lower_lane_broadcasts as p12
    assert lower.TO_LLVM[:2] == [p11b, p12], lower.TO_LLVM


CASES = [a_hoisted_lane_broadcast_is_a_vrgather_for_every_element_type,
         a_width_that_is_no_power_of_two_is_gathered_at_the_next_and_narrowed,
         an_i1_lane_value_is_gathered_as_i8,
         an_index_lane_value_is_gathered_as_i64,
         an_unused_lane_broadcast_is_erased_whatever_its_width,
         a_used_lane_broadcast_above_one_register_group_is_refused,
         a_lane_broadcast_to_rank_two_is_refused,
         a_lane_value_with_no_integer_view_is_refused,
         a_scalar_broadcast_is_uniform_and_left_alone,
         the_pass_opens_to_llvm]

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
