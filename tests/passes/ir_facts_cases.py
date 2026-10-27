"""A fact about the IR is read from the bindings -- an op's name, a value's type, an attribute --
never from a substring of the printed text, which a string attribute, a location, a longer name
or another spelling of the same thing answers wrongly (#30).
    python3 tests/passes/ir_facts_cases.py        exit 0 iff every case holds"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.driver import compiler  # noqa: E402
from pytorchsim_triton_compiler import toolchain  # noqa: E402
from pytorchsim_triton_compiler.passes.utils import indexing_maps, ir_utils, linear_layout
from pytorchsim_triton_compiler.passes import lower, transform
from pytorchsim_triton_compiler.passes.conversion import pytorchsim_to_llvm
from pytorchsim_triton_compiler.driver.stage import StageError  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402


def _ctx():
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    return ctx


def _module(body):
    return ir.Module.parse("module {\n" + body + "\n}\n")


def _file(d, name, text):
    path = os.path.join(d, name)
    with open(path, "w") as f:
        f.write(text)
    return path


# transform._check_no_zero_stride ---------------------------------------------------------------
def a_string_spelling_a_zero_stride_is_not_a_memref_type():
    with _ctx(), ir.Location.unknown():
        m = _module('''func.func @f(%a: memref<4xf32, strided<[1]>>) {
  "test.op"(%a) {note = "strided<[0, 1]>"} : (memref<4xf32, strided<[1]>>) -> ()
  return
}''')
        transform._check_no_zero_stride("probe", m, "probe.mlir")


def a_memref_type_with_a_zero_stride_is_refused():
    with _ctx(), ir.Location.unknown():
        m = _module('''func.func @f(%a: memref<4x2xf32, strided<[0, 1]>>) {
  return
}''')
        try:
            transform._check_no_zero_stride("probe", m, "probe.mlir")
        except StageError as e:
            assert "zero stride" in str(e)
        else:
            raise AssertionError("a zero stride passed")


# pipeline expect_ir: an op name counts ops ------------------------------------------------------
def _ops(d, body):
    path = _file(d, "09-probe.mlir", "module {\n" + body + "\n}\n")
    with open(path) as f:
        text = f.read()
    return text, compiler._op_counts_of(path, "probe")


def an_op_name_in_a_string_or_a_location_is_not_an_op():
    with tempfile.TemporaryDirectory() as d:
        text, ops = _ops(d, '''"test.op"() {note = "math.exp"} : () -> () loc("math.exp")''')
    assert compiler._count(text, ops, "math.exp") == 0, compiler._count(text, ops, "math.exp")


def a_longer_op_name_is_not_the_op():
    with tempfile.TemporaryDirectory() as d:
        text, ops = _ops(d, '''func.func @f(%x: f32) -> f32 {
  %y = math.expm1 %x : f32
  return %y : f32
}''')
    assert compiler._count(text, ops, "math.exp") == 0
    assert compiler._count(text, ops, "math.expm1") == 1


def an_op_printed_short_is_counted_by_its_name():
    with tempfile.TemporaryDirectory() as d:
        text, ops = _ops(d, "func.func @f() {\n  return\n}")
    assert compiler._count(text, ops, "func.return") == 1
    assert compiler._count(text, ops, ("func.return", "cf.br")) == 1


def an_attribute_of_that_name_is_counted_with_the_ops():
    with tempfile.TemporaryDirectory() as d:
        text, ops = _ops(d, """"test.a"() {test.pin} : () -> ()
"test.b"() {test.pin = 3 : i32, note = "test.pin"} : () -> ()
"test.pin"() : () -> ()""")
    assert compiler._count(text, ops, "test.pin") == 3, compiler._count(text, ops, "test.pin")


def a_type_spelling_stays_a_token_of_the_text():
    with tempfile.TemporaryDirectory() as d:
        text, ops = _ops(d, '''func.func @f(%a: memref<16xi32>) -> memref<16xi32> {
  return %a : memref<16xi32>
}''')
    assert compiler._count(text, ops, "memref<16xi32>") == 3


def a_triton_artifact_is_counted_by_the_triton_frontend():
    ttir = '''module {
  tt.func public @k(%p: !tt.ptr<f32>) {
    %v = tt.load %p {note = "tt.store"} : !tt.ptr<f32>
    tt.return
  }
}
'''
    with tempfile.TemporaryDirectory() as d:
        path = _file(d, "01-ttir.mlir", ttir)
        ops, attrs = compiler._op_counts_of(path, "probe")
    assert ops.get("tt.load") == 1 and ops.get("tt.store", 0) == 0 and attrs == {}, (ops, attrs)


# lower: the dialects left after lowering ------------------------------------------------------
def a_dialect_named_in_a_string_is_not_residue():
    with tempfile.TemporaryDirectory() as d:
        path = _file(d, "10-lowered.mlir", '''module {
  llvm.func @f() {
    %0 = llvm.mlir.constant(1 : i32) {note = "pytorchsim.transfer arith.addi"} : i32
    llvm.return
  }
}
''')
        held = lower.ops_in_file(path)
    assert {n.split(".")[0] for n in held} == {"llvm"}, held


def an_op_of_another_dialect_is_residue():
    with tempfile.TemporaryDirectory() as d:
        path = _file(d, "10-lowered.mlir", '''module {
  llvm.func @f() {
    %0 = arith.constant 1 : i32
    llvm.return
  }
}
''')
        held = lower.ops_in_file(path)
    assert "arith.constant" in held


# ir_utils.is_type / holds_type: a pointer type -----------------------------------------------------
def a_type_named_like_a_pointer_is_not_one():
    with _ctx(), ir.Location.unknown():
        assert not ir_utils.is_type(ir.Type.parse("!test.ptrish<f32>"), "tt", "ptr")
        assert ir_utils.is_type(ir.Type.parse("!tt.ptr<f32>"), "tt", "ptr")
        assert not ir_utils.is_type(ir.Type.parse("!tt.ptr<f32>"), "ptr", "ptr")
        assert ir_utils.is_type(ir.Type.parse("!ptr.ptr<#llvm.address_space<1>>"), "ptr", "ptr")
        assert not ir_utils.is_type(ir.Type.parse("f32"), "ptr", "ptr")
        assert ir_utils.holds_type(ir.Type.parse("tensor<4x!tt.ptr<f32>>"), "tt", "ptr")
        assert not ir_utils.holds_type(ir.Type.parse("tensor<4xf32>"), "tt", "ptr")


# convert-pytorchsim-to-llvm._flatten: an llvm.array by its dims
# ----------------------------------------------------------
def an_llvm_array_is_read_by_its_dims():
    from mlir.dialects import llvm
    with _ctx(), ir.Location.unknown():
        numel, elem, lanes = pytorchsim_to_llvm._flatten(
            ir.Type.parse("!llvm.array<4 x array<8 x f32>>"))
        assert (numel, lanes) == (32, 4) and isinstance(elem, ir.F32Type), (numel, elem, lanes)
        numel, elem, lanes = pytorchsim_to_llvm._flatten(
            ir.Type.parse('!llvm.array<2 x struct<"array<3 x f32>", (f32)>>'))
        assert (numel, lanes) == (2, 2) and isinstance(elem, llvm.StructType), (numel, elem, lanes)
        numel, elem, lanes = pytorchsim_to_llvm._flatten(ir.Type.parse("i32"))
        assert (numel, lanes) == (1, 1) and isinstance(elem, ir.IntegerType)


def a_spad_global_reserves_exactly_its_need():
    """reserve_spad_storage sizes each lane's share at need_per_lane: one and three elements stay
    one and three (they were rounded up to MIN_VEC = 2 while a vector<2> could be stored there)."""
    with _ctx(), ir.Location.unknown():
        m = ir.Module.parse(
            "llvm.mlir.global internal @a_spad_128lane() {addr_space = 1 : i32} : "
            "!llvm.array<128 x array<1 x f32>>\n"
            "llvm.mlir.global internal @b_spad_128lane() {addr_space = 1 : i32} : "
            "!llvm.array<128 x array<3 x i32>>\n")
        got = pytorchsim_to_llvm.reserve_spad_storage(m, 128, 256)
        assert [(n, c) for n, c, _ in got] == [("a_spad_128lane", 1), ("b_spad_128lane", 3)], got


# ir_utils.enum_of / attr_is: enum attributes -------------------------------------------------------
def a_reduction_kind_is_the_attribute_not_its_spelling():
    kinds = ["add", "maxnumf", "mul"]
    with _ctx(), ir.Location.unknown():
        maxnumf = ir.Attribute.parse("#vector.kind<maxnumf>")
        assert ir_utils.enum_of(maxnumf, "#vector.kind<%s>", kinds) == "maxnumf"
        assert ir_utils.enum_of(ir.Attribute.parse("#vector.kind<add>"), "#vector.kind<%s>", kinds) == "add"
        assert ir_utils.enum_of(ir.Attribute.parse("#vector.kind<minsi>"), "#vector.kind<%s>", kinds) is None
        assert ir_utils.enum_of(ir.StringAttr.get("<add>"), "#vector.kind<%s>", kinds) is None


def an_iterator_type_is_the_attribute_not_its_spelling():
    with _ctx(), ir.Location.unknown():
        assert ir_utils.attr_is(ir.Attribute.parse("#linalg.iterator_type<parallel>"),
                              "#linalg.iterator_type<parallel>")
        assert not ir_utils.attr_is(ir.Attribute.parse("#linalg.iterator_type<reduction>"),
                                  "#linalg.iterator_type<parallel>")
        assert not ir_utils.attr_is(ir.StringAttr.get("parallel"), "#linalg.iterator_type<parallel>")


# linear_layout.get_bases: the encoding's body as an attribute --------------------------------
def a_linear_encoding_is_read_however_it_is_spaced():
    with _ctx(), ir.Location.unknown():
        tight = ir.Attribute.parse("#ttg.linear<{register=[[1]],lane=[[0],[2]],warp=[],block=[]}>")
        assert linear_layout.get_bases(tight) == ([[1]], [[0], [2]])
        spaced = ir.Attribute.parse(
            "#ttg.linear<{register = [[1, 0], [0, 4]], lane = [], warp = [], block = []}>")
        assert linear_layout.get_bases(spaced) == ([[1, 0], [0, 4]], [])
        warped = ir.Attribute.parse("#ttg.linear<{register = [], lane = [], warp = [[1]], block = []}>")
        assert linear_layout.get_bases(warped) is None
        assert linear_layout.get_bases(ir.Attribute.parse("#ttg.blocked<{sizePerThread = [1]}>")) is None
        spelled = ir.StringAttr.get("#ttg.linear<{register = [], lane = [], warp = [], block = []}>")
        assert linear_layout.get_bases(spelled) is None


# linear_layout.compose_reshape: Triton's reshapeLayout ----------------------------------------
def a_reshape_renames_each_basis_row_major_and_refuses_other_shapes():
    """Breaks if compose_reshape renames a basis other than row-major, as Triton's reshapeLayout
    does, or reshapes instead of refusing a split or merged dim not a power of two, or shapes of
    different sizes."""
    got = linear_layout.compose_reshape(([[1, 0], [0, 2]], [[0, 1], [2, 0]]), [4, 4], [2, 8])
    assert got == ([[0, 4], [0, 2]], [[0, 1], [1, 0]]), got
    assert linear_layout.compose_reshape(([[0, 1]], [[1, 0]]), [2, 2], [4]) == ([[1]], [[2]])
    for src, dst in (([3, 4], [12]), ([4, 4], [2, 4])):
        try:
            linear_layout.compose_reshape(([], [[1, 0]]), src, dst)
        except ValueError:
            continue
        raise AssertionError(f"{src} -> {dst} was reshaped")


# indexing_maps._stated_maps: a map result is an iteration dim by its AffineDimExpr -----------
def a_map_result_is_an_iteration_dim_by_its_affine_expr():
    with _ctx(), ir.Location.unknown():
        m = _module('''func.func @f(%a: memref<8x4xf32>, %b: memref<2x8xf32>, %c: memref<2x4xf32>) {
  "pytorchsim.dot"(%a, %b, %c) {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>,
                                               affine_map<(d0, d1, d2) -> (d1, d2)>,
                                               affine_map<(d0, d1, d2) -> (d1, d0)>]}
      : (memref<8x4xf32>, memref<2x8xf32>, memref<2x4xf32>) -> ()
  "test.op"(%a) {indexing_maps = [affine_map<(d0, d1)[s0] -> (d0 + d1, s0, 0, d1)>]}
      : (memref<8x4xf32>) -> ()
  return
}''')
        dot, other = list(m.body.operations[0].regions[0].blocks[0].operations)[:2]
        assert indexing_maps._stated_maps(dot) == [[2, 0], [1, 2], [1, 0]]
        assert indexing_maps.matmul_axis(dot, 2, indexing_maps.MATMUL_N) == 0
        assert indexing_maps.matmul_axis(dot, 0, indexing_maps.MATMUL_K) == 0
        assert indexing_maps._stated_maps(other) == [[None, None, None, 1]]


# convert-transfers-to-dma's _erase_globals_nothing_reads, dep_analysis.buffer_of: a symbol is its attribute
_SYMBOLS = '''memref.global "private" @"spad@" : memref<8xf32, 1>
memref.global "private" @dead : memref<8xf32, 1>
func.func @f() -> f32 {
  %c0 = arith.constant 0 : index
  %s = memref.get_global @"spad@" : memref<8xf32, 1>
  %x = memref.load %s[%c0] : memref<8xf32, 1>
  return %x : f32
}'''


def a_read_global_is_kept_whatever_its_name_spells():
    """Breaks if convert-transfers-to-dma tells a global's name from its printed text, which `@"spad@"`
    loses."""
    from pytorchsim_triton_compiler.passes.conversion import transfers_to_dma
    with _ctx(), ir.Location.unknown():
        m = _module(_SYMBOLS)
        transfers_to_dma._erase_globals_nothing_reads(m.operation.regions[0].blocks[0])
        kept = [ir.StringAttr(o.attributes["sym_name"]).value for o in ir_utils.walk(m.operation)
                if o.name == "memref.global"]
        assert kept == ["spad@"], kept
        assert m.operation.verify()


def the_tog_names_a_buffer_by_its_symbol():
    """Breaks if the TOG's dep_analysis names a buffer by the printed text of its symbol."""
    from pytorchsim_triton_compiler.togsim import dep_analysis
    with _ctx(), ir.Location.unknown():
        m = _module(_SYMBOLS)
        g = [o for o in ir_utils.walk(m.operation) if o.name == "memref.get_global"][0]
        assert dep_analysis.buffer_of(g.results[0]) == "spad@"


# triton_compile.drop_noops: erased as ops, in the Triton frontend's own process -----------------
def a_noop_named_elsewhere_is_not_erased_and_the_op_is():
    ttir = '''module {
  tt.func public @k(%p: !tt.ptr<f32>, %c: i1) {
    llvm.intr.assume %c : i1
    %v = tt.load %p {note = "llvm.intr.assume"} : !tt.ptr<f32>
    tt.return loc("llvm.intr.assume")
  }
}
'''
    probe = '''
import sys
from pytorchsim_triton_compiler import frontend as triton_compile
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import make_backend
ctx = ir.context(); ir.load_dialects(ctx); make_backend(GPUTarget("cpu", 0, 0)).load_dialects(ctx)
module = ir.parse_mlir_module(sys.argv[1], ctx)
n = triton_compile.drop_noops(module, ctx)
print(n, sorted(triton_compile.op_counts.__wrapped__(module) if False else []))
left = []
module.walk(lambda op: left.append(op.get_name()))
print(n, " ".join(sorted(left)))
'''
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with tempfile.TemporaryDirectory() as d:
        path = _file(d, "01-ttir.mlir", ttir)
        env = toolchain.triton_env()
        env["PYTHONPATH"] = root + os.pathsep + env["PYTHONPATH"]
        got = subprocess.run([sys.executable, "-c", probe, path], env=env, capture_output=True,
                             text=True)
    assert got.returncode == 0, got.stderr[-800:]
    last = got.stdout.strip().splitlines()[-1]
    n, names = last.split(" ", 1)
    assert n == "1" and "llvm.intr.assume" not in names.split() and "tt.return" in names, last


CASES = [a_string_spelling_a_zero_stride_is_not_a_memref_type, a_memref_type_with_a_zero_stride_is_refused,
         an_op_name_in_a_string_or_a_location_is_not_an_op, a_longer_op_name_is_not_the_op,
         an_op_printed_short_is_counted_by_its_name, an_attribute_of_that_name_is_counted_with_the_ops,
         a_type_spelling_stays_a_token_of_the_text,
         a_triton_artifact_is_counted_by_the_triton_frontend,
         a_dialect_named_in_a_string_is_not_residue, an_op_of_another_dialect_is_residue,
         a_type_named_like_a_pointer_is_not_one, an_llvm_array_is_read_by_its_dims,
         a_spad_global_reserves_exactly_its_need,
         a_reduction_kind_is_the_attribute_not_its_spelling,
         an_iterator_type_is_the_attribute_not_its_spelling,
         a_linear_encoding_is_read_however_it_is_spaced,
         a_reshape_renames_each_basis_row_major_and_refuses_other_shapes,
         a_map_result_is_an_iteration_dim_by_its_affine_expr,
         a_noop_named_elsewhere_is_not_erased_and_the_op_is,
         a_read_global_is_kept_whatever_its_name_spells, the_tog_names_a_buffer_by_its_symbol]


def main():
    bad = 0
    for case in CASES:
        try:
            case()
            print(f"ok    {case.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
