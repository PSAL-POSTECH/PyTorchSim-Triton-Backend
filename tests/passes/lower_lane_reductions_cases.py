"""A lane's own row folded by torchsim-legalize reaches llc as one RVV vfred*/vred* whose answer stays in a
register: no vslidedown ladder, no vfmv.f.s / vmv.x.s; a kind the V extension has no reduction for is refused.
    python3 tests/passes/lower_lane_reductions_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from case_runner import run  # noqa: E402
from ir_helpers import asm_mnemonics  # noqa: E402
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lower, p06_legalize  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

T = machine.target()
LANE_EXTRACT = ("vfmv.f.s", "vmv.x.s")
COMBINE = {"add": ("arith.addf", "arith.addi"), "maxnumf": ("arith.maxnumf", None),
           "minnumf": ("arith.minnumf", None), "maximumf": ("arith.maximumf", None),
           "maxsi": (None, "arith.maxsi"), "minui": (None, "arith.minui"),
           "xor": (None, "arith.xori"), "mul": ("arith.mulf", "arith.muli")}


def _row(kind, elem="f32", n=128):
    comb = COMBINE[kind][0 if elem[0] in "fb" else 1]
    t, r = f"memref<128x{n}x{elem}, 1>", f"memref<128x{elem}, 1>"
    return ("module {\n"
            f"  memref.global @row_spad_128lane : {r}\n"
            f"  memref.global @tile_spad_128lane : {t}\n"
            "  func.func @f() {\n"
            f"    %r = memref.get_global @row_spad_128lane : {r}\n"
            f"    %t = memref.get_global @tile_spad_128lane : {t}\n"
            "    affine.for %l = 0 to 128 {\n"
            "      affine.for %i = 0 to 1 {\n"
            f"        %s = affine.vector_load %r[%i] {{torchsim.lane_splat}} : {r}, vector<1x{elem}>\n"
            f"        %b = vector.broadcast %s : vector<1x{elem}> to vector<{n}x{elem}>\n"
            f"        %m = affine.for %j = 0 to {n} step {n} iter_args(%a = %b) -> (vector<{n}x{elem}>) {{\n"
            f"          %x = affine.vector_load %t[%i, %j] : {t}, vector<{n}x{elem}>\n"
            f"          %y = {comb} %x, %a : vector<{n}x{elem}>\n"
            f"          affine.yield %y : vector<{n}x{elem}>\n"
            "        } {torchsim.reduce_loop, torchsim.vector_loop}\n"
            f"        %v = vector.reduction <{kind}>, %m : vector<{n}x{elem}> into {elem}\n"
            f"        %w = vector.broadcast %v : {elem} to vector<1x{elem}>\n"
            f"        affine.vector_store %w, %r[%i] : {r}, vector<1x{elem}>\n"
            "      }\n"
            "    } {torchsim.vlane_loop = true}\n"
            "    return\n  }\n}\n")


def _legalized(text):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(text)
        p06_legalize.run(m, lanes=T.vector_lanes, vlen=T.vlen_bits, lmul=8)
        assert m.operation.verify()
        return str(m), [o.name for o in walk(m.operation)]


def _one_reduction(kind, elem, insn, n=128):
    text, names = _legalized(_row(kind, elem, n))
    assert names.count("vector.multi_reduction") == 1, (kind, elem, names)
    ops = asm_mnemonics(text, T)
    what = (kind, elem, n)
    assert ops.count(insn) == 1, (what, insn, sorted(set(o for o in ops if "red" in o)))
    assert not [o for o in ops if o.startswith("vslidedown")], (what, ops)
    assert not set(LANE_EXTRACT) & set(ops), (what, sorted(set(LANE_EXTRACT) & set(ops)))


def a_float_sum_is_one_vfredusum():
    _one_reduction("add", "f32", "vfredusum.vs")


def a_float_max_and_min_are_one_vfredmax_and_one_vfredmin():
    _one_reduction("maxnumf", "f32", "vfredmax.vs")
    _one_reduction("minnumf", "f32", "vfredmin.vs")


def an_integer_reduction_is_its_vred():
    _one_reduction("add", "i32", "vredsum.vs")
    _one_reduction("maxsi", "i32", "vredmax.vs")
    _one_reduction("minui", "i32", "vredminu.vs")
    _one_reduction("xor", "i32", "vredxor.vs")


def a_row_of_one_register_to_an_lmul_8_group_is_one_reduction():
    for n in (T.vlen_bits // 32, 2 * T.vlen_bits // 32, 8 * T.vlen_bits // 32):
        _one_reduction("add", "f32", "vfredusum.vs", n=n)


def a_row_that_is_no_power_of_two_is_one_reduction_of_its_own_length():
    _one_reduction("maxnumf", "f32", "vfredmax.vs", n=3 * T.vlen_bits // 32)


def a_kind_with_no_rvv_reduction_keeps_the_halving_tree():
    """The tree halves down to vector<1> and its last combiner is what is stored: no
    multi_reduction, and no shuffle widening the answer to a pair."""
    for kind, elem in (("mul", "f32"), ("mul", "i32"), ("maximumf", "f32"), ("add", "f16")):
        comb = COMBINE[kind][0 if elem[0] in "fb" else 1]
        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        with ctx, ir.Location.unknown():
            m = ir.Module.parse(_row(kind, elem))
            p06_legalize.run(m, lanes=T.vector_lanes, vlen=T.vlen_bits, lmul=8)
            assert m.operation.verify()
            names = [o.name for o in walk(m.operation)]
            assert "vector.multi_reduction" not in names, (kind, elem, names)
            assert "vector.shuffle" not in names, (kind, elem, names)
            st = [o for o in walk(m.operation) if o.name == "affine.vector_store"]
            v = st[-1].operands[0]
            assert ir.VectorType(v.type).shape == [1] and v.owner.name == comb, (kind, elem, v)


def a_lane_reduction_with_no_rvv_instruction_is_refused():
    from pytorchsim_triton_compiler.lowering.passes import p12a_lower_lane_reductions as p12a
    for kind, elem in (("mul", "f32"), ("maximumf", "f32"), ("add", "f16")):
        ctx = ir.Context()
        with ctx, ir.Location.unknown():
            m = ir.Module.parse(
                f"func.func @f(%x: vector<1x8x{elem}>, %a: vector<1x{elem}>) -> vector<1x{elem}> {{\n"
                f"  %r = vector.multi_reduction <{kind}>, %x, %a [1] : vector<1x8x{elem}> "
                f"to vector<1x{elem}>\n  return %r : vector<1x{elem}>\n}}\n")
            try:
                p12a.run(m, target=T)
            except p12a.LaneReductionRefused as e:
                assert kind in str(e), (kind, str(e))
                continue
            raise AssertionError(f"not refused: {kind} {elem}")


def the_pass_runs_after_lane_broadcasts_before_llvm():
    from pytorchsim_triton_compiler.lowering.passes import p12a_lower_lane_reductions as p12a
    from pytorchsim_triton_compiler.lowering.passes import p12_lower_lane_broadcasts as p12
    assert lower.TO_LLVM.index(p12a) == lower.TO_LLVM.index(p12) + 1, lower.TO_LLVM


CASES = [a_float_sum_is_one_vfredusum,
         a_float_max_and_min_are_one_vfredmax_and_one_vfredmin,
         an_integer_reduction_is_its_vred,
         a_row_of_one_register_to_an_lmul_8_group_is_one_reduction,
         a_row_that_is_no_power_of_two_is_one_reduction_of_its_own_length,
         a_kind_with_no_rvv_reduction_keeps_the_halving_tree,
         a_lane_reduction_with_no_rvv_instruction_is_refused,
         the_pass_runs_after_lane_broadcasts_before_llvm]

if __name__ == "__main__":
    sys.exit(run(CASES))
