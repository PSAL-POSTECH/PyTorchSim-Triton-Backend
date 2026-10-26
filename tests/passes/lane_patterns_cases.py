"""convert-pytorchsim-to-llvm's lane patterns, applied before the standard lowering: a vector<1>'s broadcast
hoisted out of the loop that uses it reaches llc as a vrgather, and a lane's own row folded by
pytorchsim-legalize-vectors as one vfred*/vred* whose answer stays in a register -- never vfmv.f.s /
vmv.x.s, no vslidedown ladder; what has no register group or no RVV reduction is refused. Read off llc's
object, decoded (elf/lane_moves).
    python3 tests/passes/lane_patterns_cases.py        exit 0 iff every case holds"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from mlir.dialects import affine, arith, vector  # noqa: E402
from pytorchsim_triton_compiler.lowering.elf import lane_moves  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import (  # noqa: E402
    b08_legalize_vectors, c24_convert_pytorchsim_to_llvm as c24, lower)
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_register_groups import (  # noqa: E402
    LaneBroadcastRefused, LaneReductionRefused)

T = machine.target()
OPIVI, OPFVV, OPMVV, OPIVX = 3, 1, 2, 4
VRGATHER_VI = (OPIVI, 0b001100)
VSLIDEDOWN = {(OPIVI, 0b001111), (OPIVX, 0b001111)}
REDUCTION = {"vfredusum": (OPFVV, 0b000001), "vfredmax": (OPFVV, 0b000111),
             "vfredmin": (OPFVV, 0b000101), "vredsum": (OPMVV, 0b000000),
             "vredmax": (OPMVV, 0b000111), "vredminu": (OPMVV, 0b000100), "vredxor": (OPMVV, 0b000011)}
SUB = {"f": "arith.subf", "b": "arith.subf", "i": "arith.subi"}
COMBINE = {"add": ("arith.addf", "arith.addi"), "maxnumf": ("arith.maxnumf", None),
           "minnumf": ("arith.minnumf", None), "maximumf": ("arith.maximumf", None),
           "maxsi": (None, "arith.maxsi"), "minui": (None, "arith.minui"),
           "xor": (None, "arith.xori"), "mul": ("arith.mulf", "arith.muli")}
COMBINER = {"arith.mulf": arith.MulFOp, "arith.muli": arith.MulIOp, "arith.maximumf": arith.MaximumFOp,
            "arith.addf": arith.AddFOp}


def _object(text, d):
    """`text` through TO_LLVM, mlir-translate and llc: the object's path."""
    src, out, ll, obj = (os.path.join(d, f) for f in ("k.mlir", "l.mlir", "k.ll", "k.o"))
    with open(src, "w") as f:
        f.write(text)
    lower.to_llvm(src, out, target=T)
    subprocess.run([toolchain.MLIR_TRANSLATE, "-mlir-to-llvmir", out, "-o", ll], check=True)
    subprocess.run([toolchain.LLC, "-relocation-model=pic", "-march=" + T.llc_march(), "-O3",
                    "-mattr=" + T.llc_mattr(), "-filetype=obj", ll, "-o", obj], check=True)
    return obj


def _vector_ops(text):
    """((funct3, funct6) of every OP-V instruction, the vector-to-scalar moves) of `text`'s object."""
    with tempfile.TemporaryDirectory() as d:
        obj = _object(text, d)
        ops = [((w >> 12) & 7, w >> 26) for _, _, w in lane_moves.instruction_words(obj)
               if w is not None and w & 0x7F == lane_moves.OP_V]
        return ops, lane_moves.vector_to_scalar_moves(obj)


def _parsed(text):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        return ir.Module.parse(text)


def _count(m, cls):
    return sum(isinstance(o.opview, cls) for o in walk(m.operation))


# A lane value's broadcast.

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


def _gathers_and_never_extracts(text, what):
    ops, moves = _vector_ops(text)
    assert not moves, (what, moves)
    assert VRGATHER_VI in ops, (what, ops)


def _refused(text, refusal, *said):
    m = _parsed(text)
    with m.context, ir.Location.unknown():
        try:
            c24.lower_lane_values(m, T)
        except refusal as e:
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
    m = _parsed("func.func @f(%s: vector<1xf32>) {\n"
                "  %b = vector.broadcast %s : vector<1xf32> to vector<4096xf32>\n"
                "  return\n}\n")
    with m.context, ir.Location.unknown():
        c24.lower_lane_values(m, T)
    assert _count(m, vector.BroadcastOp) == 0, m


def a_used_lane_broadcast_above_one_register_group_is_refused():
    n = 2 * 8 * T.vlen_bits // 32
    _refused("func.func @f(%%s: vector<1xf32>) -> vector<%dxf32> {\n"
             "  %%b = vector.broadcast %%s : vector<1xf32> to vector<%dxf32>\n"
             "  return %%b : vector<%dxf32>\n}\n" % (n, n, n), LaneBroadcastRefused, "LMUL-8", str(n))


def a_lane_broadcast_to_rank_two_is_refused():
    _refused("func.func @f(%s: vector<1xf32>) -> vector<4x8xf32> {\n"
             "  %b = vector.broadcast %s : vector<1xf32> to vector<4x8xf32>\n"
             "  return %b : vector<4x8xf32>\n}\n", LaneBroadcastRefused, "rank 2")


def a_lane_value_with_no_integer_view_is_refused():
    _refused("func.func @f(%s: vector<1xi4>) -> vector<8xi4> {\n"
             "  %b = vector.broadcast %s : vector<1xi4> to vector<8xi4>\n"
             "  return %b : vector<8xi4>\n}\n", LaneBroadcastRefused, "i4")


def a_scalar_broadcast_is_uniform_and_left_alone():
    m = _parsed("func.func @f(%s: f32) -> vector<64xf32> {\n"
                "  %b = vector.broadcast %s : f32 to vector<64xf32>\n"
                "  return %b : vector<64xf32>\n}\n")
    with m.context, ir.Location.unknown():
        c24.lower_lane_values(m, T)
    assert _count(m, vector.BroadcastOp) == 1, m


# A lane's row reduction.

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
            f"        %s = affine.vector_load %r[%i] {{pytorchsim.lane_splat}} : {r}, vector<1x{elem}>\n"
            f"        %b = vector.broadcast %s : vector<1x{elem}> to vector<{n}x{elem}>\n"
            f"        %m = affine.for %j = 0 to {n} step {n} iter_args(%a = %b) -> (vector<{n}x{elem}>) {{\n"
            f"          %x = affine.vector_load %t[%i, %j] : {t}, vector<{n}x{elem}>\n"
            f"          %y = {comb} %x, %a : vector<{n}x{elem}>\n"
            f"          affine.yield %y : vector<{n}x{elem}>\n"
            "        } {pytorchsim.reduce_loop, pytorchsim.vector_loop}\n"
            f"        %v = vector.reduction <{kind}>, %m : vector<{n}x{elem}> into {elem}\n"
            f"        %w = vector.broadcast %v : {elem} to vector<1x{elem}>\n"
            f"        affine.vector_store %w, %r[%i] : {r}, vector<1x{elem}>\n"
            "      }\n"
            "    } {pytorchsim.vlane_loop = true}\n"
            "    return\n  }\n}\n")


def _legalized(text):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(text)
        b08_legalize_vectors.run(m, lanes=T.vector_lanes, vlen=T.vlen_bits, lmul=8)
        assert m.operation.verify()
        return str(m), _count(m, vector.MultiDimReductionOp)


def _one_reduction(kind, elem, insn, n=128):
    text, reductions = _legalized(_row(kind, elem, n))
    assert reductions == 1, (kind, elem, reductions)
    ops, moves = _vector_ops(text)
    what = (kind, elem, n)
    assert ops.count(REDUCTION[insn]) == 1, (what, insn, ops)
    assert not VSLIDEDOWN & set(ops), (what, ops)
    assert not moves, (what, moves)


def a_float_sum_is_one_vfredusum():
    _one_reduction("add", "f32", "vfredusum")


def a_float_max_and_min_are_one_vfredmax_and_one_vfredmin():
    _one_reduction("maxnumf", "f32", "vfredmax")
    _one_reduction("minnumf", "f32", "vfredmin")


def an_integer_reduction_is_its_vred():
    _one_reduction("add", "i32", "vredsum")
    _one_reduction("maxsi", "i32", "vredmax")
    _one_reduction("minui", "i32", "vredminu")
    _one_reduction("xor", "i32", "vredxor")


def a_row_of_one_register_to_an_lmul_8_group_is_one_reduction():
    for n in (T.vlen_bits // 32, 2 * T.vlen_bits // 32, 8 * T.vlen_bits // 32):
        _one_reduction("add", "f32", "vfredusum", n=n)


def a_row_that_is_no_power_of_two_is_one_reduction_of_its_own_length():
    _one_reduction("maxnumf", "f32", "vfredmax", n=3 * T.vlen_bits // 32)


def a_kind_with_no_rvv_reduction_keeps_the_halving_tree():
    """The tree halves down to vector<1> and its last combiner is what is stored: no
    multi_reduction, and no shuffle widening the answer to a pair."""
    for kind, elem in (("mul", "f32"), ("mul", "i32"), ("maximumf", "f32"), ("add", "f16")):
        comb = COMBINER[COMBINE[kind][0 if elem[0] in "fb" else 1]]
        m = _parsed(_row(kind, elem))
        with m.context, ir.Location.unknown():
            b08_legalize_vectors.run(m, lanes=T.vector_lanes, vlen=T.vlen_bits, lmul=8)
            assert m.operation.verify()
            assert _count(m, vector.MultiDimReductionOp) == 0, (kind, elem, m)
            assert _count(m, vector.ShuffleOp) == 0, (kind, elem, m)
            st = [o for o in walk(m.operation) if isinstance(o.opview, affine.AffineVectorStoreOp)]
            v = st[-1].operands[0]
            assert ir.VectorType(v.type).shape == [1] and isinstance(v.owner.opview, comb), (kind, elem, v)


def a_lane_reduction_with_no_rvv_instruction_is_refused():
    for kind, elem in (("mul", "f32"), ("maximumf", "f32"), ("add", "f16")):
        _refused(f"func.func @f(%x: vector<1x8x{elem}>, %a: vector<1x{elem}>) -> vector<1x{elem}> {{\n"
                 f"  %r = vector.multi_reduction <{kind}>, %x, %a [1] : vector<1x8x{elem}> "
                 f"to vector<1x{elem}>\n  return %r : vector<1x{elem}>\n}}\n",
                 LaneReductionRefused, kind)


# Where they live.

def legalize_vectors_reads_the_patterns_table():
    """Breaks if b08's choice of a lane reduction over a halving tree and the pattern that lowers it
    read two tables: b08 asks c24.lane_reduction_instruction, the table the pattern lowers by."""
    m = _parsed("func.func @f() { return }")
    with m.context:
        names = b08_legalize_vectors._build_lane_reduction.__code__.co_varnames
        assert "lane_reduction_instruction" in names, "b08 no longer reads c24's table"
        for e in (ir.F32Type.get(), ir.F16Type.get(), ir.IntegerType.get_signless(1)):
            assert c24.lane_reduction_instruction("mul", e) is None, e
        assert c24.lane_reduction_instruction("add", ir.F32Type.get()) == "vfredusum"


def the_patterns_outrank_upstream_and_no_pass_of_to_llvm_is_theirs():
    assert c24.LANE_BENEFIT > 1, c24.LANE_BENEFIT
    names = [p.__name__.rsplit(".", 1)[-1] for p in lower.TO_LLVM]
    assert not [n for n in names if "lane" in n], names


CASES = [a_hoisted_lane_broadcast_is_a_vrgather_for_every_element_type,
         a_width_that_is_no_power_of_two_is_gathered_at_the_next_and_narrowed,
         an_i1_lane_value_is_gathered_as_i8,
         an_index_lane_value_is_gathered_as_i64,
         an_unused_lane_broadcast_is_erased_whatever_its_width,
         a_used_lane_broadcast_above_one_register_group_is_refused,
         a_lane_broadcast_to_rank_two_is_refused,
         a_lane_value_with_no_integer_view_is_refused,
         a_scalar_broadcast_is_uniform_and_left_alone,
         a_float_sum_is_one_vfredusum,
         a_float_max_and_min_are_one_vfredmax_and_one_vfredmin,
         an_integer_reduction_is_its_vred,
         a_row_of_one_register_to_an_lmul_8_group_is_one_reduction,
         a_row_that_is_no_power_of_two_is_one_reduction_of_its_own_length,
         a_kind_with_no_rvv_reduction_keeps_the_halving_tree,
         a_lane_reduction_with_no_rvv_instruction_is_refused,
         legalize_vectors_reads_the_patterns_table,
         the_patterns_outrank_upstream_and_no_pass_of_to_llvm_is_theirs]

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
