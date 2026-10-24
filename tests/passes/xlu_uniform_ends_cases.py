"""A copy from a UNIFORM buffer is each lane's copy from its own bank; a copy into one lists every
lane (lower_convert); a vector broadcast along the lanes is gathered into one UNIFORM row (p02).
    python3 tests/passes/xlu_uniform_ends_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects import pytorchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family import lower_convert  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family.lib_xlu import (  # noqa: E402
    XLU_CROSSING, XLU_OPS, emitter_of, plan_of_xlu_op)
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family.lib_xlu_move import (  # noqa: E402
    KIND_ATTR, ROUNDS_ATTR, XLU_MOVE_OP)
from pytorchsim_triton_compiler.lowering.passes.lib_bank_layout import EVERY_LANE  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import indexing_maps, walk  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p02_materialize_layouts as p02  # noqa: E402

pytorchsim.register()
LANES = 128
U = "memref<1x16xf32, 1>"
B = "memref<128x1x16xf32, 1>"
BROADCAST = f"""
  "pytorchsim.convert_layout"(%u, %b) <{{added_dims = array<i64: 0>, dst_perm = array<i64: 0, 1, 2>,
      src_perm = array<i64: 0, 1>}}> : ({U}, {B}) -> ()"""


def _lowered(args, body):
    """`(move kind, whether it says EVERY_LANE, rounds)` of every move and the name of every op,
    after lower_convert ran on a function of `args` and `body`."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(f"func.func @f({args}) {{\n{body}\n  return\n}}\n")
        lower_convert.run(m, lanes=LANES)
        assert m.operation.verify()
        ops = [o for o in walk(m.operation)]
        none = ir.ArrayAttr.get([])
        moves = [(ir.StringAttr(o.attributes[KIND_ATTR]).value,
                  EVERY_LANE in o.attributes,
                  [tuple(ir.DenseI64ArrayAttr(r))
                   for r in ir.ArrayAttr(o.attributes.get(ROUNDS_ATTR, none))])
                 for o in ops if o.name == XLU_MOVE_OP]
        return moves, [o.name for o in ops]


def a_column_into_a_uniform_row_is_no_crossing():
    pairs = [((f, 0), (lane, f)) for f in range(16) for lane in range(LANES)]
    assert all(plan_of_xlu_op(c, pairs, [16], [1, 16], LANES, False, 16) is None
               for c in XLU_OPS if emitter_of(c) == XLU_CROSSING)
    moves, _ = _lowered(f"%c: memref<16xf32, 1>, %u: {U}", f"""
  "pytorchsim.convert_layout"(%c, %u) <{{dst_perm = array<i64: 0, 1>, src_perm = array<i64: 0>}}>
      : (memref<16xf32, 1>, {U}) -> ()""")
    assert [(k, e, len(r)) for k, e, r in moves] == [("gather", True, 1)], moves


def a_row_picked_into_a_uniform_buffer_is_one_lane_sent_to_every_lane():
    moves, _ = _lowered(f"%d: memref<8x4xf32, 1>, %u: memref<1x4xf32, 1>", """
  "pytorchsim.convert_layout"(%d, %u) <{src_perm = array<i64: 0, 1>, dst_perm = array<i64: 0, 1>,
      static_offsets = array<i64: 3, 0>, static_sizes = array<i64: 1, 4>,
      static_strides = array<i64: 1, 1>}> : (memref<8x4xf32, 1>, memref<1x4xf32, 1>) -> ()""")
    assert len(moves) == 1 and moves[0][1], moves
    assert {(r[1], r[2], r[4], r[5]) for r in moves[0][2]} == {(0, 3, 0, LANES)}, moves


def a_broadcast_of_a_uniform_row_is_copied_in_lane():
    moves, names = _lowered(f"%u: {U}, %b: {B}", BROADCAST)
    assert moves == [] and "vector.transfer_write" in names, (moves, names)


def a_fold_result_also_stored_is_copied_in_lane():
    moves, names = _lowered(f"%d: memref<2x16xf32, 1>, %u: {U}, %b: {B}, %o: memref<16xf32>", f"""
  %c0 = arith.constant 0 : index
  "pytorchsim.fold"(%d, %u) <{{by = "arith.maximumf"}}>
      : (memref<2x16xf32, 1>, {U}) -> ()
  "pytorchsim.transfer"(%o, %c0, %u, %c0, %c0) <{{direction = "store", dram_arg = 3 : i64,
      dram_stride = [16, 1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}>
      : (memref<16xf32>, index, {U}, index, index) -> (){BROADCAST}""")
    assert moves == [] and "pytorchsim.fold" in names, (moves, names)
    assert "vector.transfer_write" in names, names


def a_uniform_row_into_one_row_of_a_spread_buffer_stays_a_move():
    """Each lane's copy from its own bank would write that row in all 128 lanes, over the rows the
    other lanes hold; the copy writes lane 5's alone."""
    moves, _ = _lowered(f"%u: {U}, %d: memref<128x16xf32, 1>", f"""
  "pytorchsim.convert_layout"(%u, %d) <{{src_perm = array<i64: 0, 1>, dst_perm = array<i64: 0, 1>,
      static_offsets = array<i64: 5, 0>, static_sizes = array<i64: 1, 16>,
      static_strides = array<i64: 1, 1>, into_window}}> : ({U}, memref<128x16xf32, 1>) -> ()""")
    assert len(moves) == 1 and not moves[0][1], moves
    assert {(r[4], r[5]) for r in moves[0][2]} == {(5, 6)}, moves


GATHER = """
  %r = linalg.broadcast ins(%s : tensor<64xf32>) outs(%i : tensor<512x64xf32>) dimensions = [0]
  %o = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%r, %w : tensor<512x64xf32>, tensor<512x64xf32>) outs(%i : tensor<512x64xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %m = arith.mulf %a, %b : f32
      linalg.yield %m : f32
  } -> tensor<512x64xf32>"""


def _broadcast_rewritten(tail, check):
    """`check(copies, generics, names)` -- the copies' and the generics' ops and every op name -- after
    p02 rewrote the lane broadcast of a [64] vector over [512, 64] that GATHER and `tail` read."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(
            "func.func @f(%s: tensor<64xf32>, %i: tensor<512x64xf32>, %w: tensor<512x64xf32>,"
            " %v: tensor<64xf32>) {"
            f"{GATHER}{tail}\n  return\n}}\n")
        p02._rewrite_broadcast_across_lanes(
            next(o for o in walk(m.operation) if o.name == "linalg.broadcast"))
        assert m.operation.verify()
        ops = list(walk(m.operation))
        check([o for o in ops if o.name == pytorchsim.ConvertLayoutOp.OPERATION_NAME],
              [o for o in ops if o.name == "linalg.generic"], [o.name for o in ops])


def _row_ty():
    return ir.RankedTensorType.get([1, 64], ir.F32Type.get())


def _reads_the_row(gen, k=0):
    """Whether operand `k` of `gen` is the [1, 64] row, read through a map of axis 0 at 0, then d1."""
    exprs = list(indexing_maps(gen)[k].results)
    return (gen.operands[k].type == _row_ty() and isinstance(exprs[0], ir.AffineConstantExpr)
            and ir.AffineConstantExpr(exprs[0]).value == 0 and isinstance(exprs[1], ir.AffineDimExpr)
            and ir.AffineDimExpr(exprs[1]).position == 1)


def a_vector_broadcast_along_the_lanes_is_one_uniform_row_read_through_maps():
    def check(copies, gens, _names):
        assert [c.results[0].type for c in copies] == [_row_ty()], copies
        assert len(gens) == 1 and _reads_the_row(gens[0]), gens
    _broadcast_rewritten("", check)


def a_reader_that_states_no_map_takes_the_row_copied_in_lane():
    """Breaks if a reader of no LinalgOp maps is handed the [1, 64] row it cannot read."""
    def check(copies, gens, names):
        assert [c.results[0].type for c in copies] == [
            _row_ty(), ir.RankedTensorType.get([512, 64], ir.F32Type.get())], copies
        assert _reads_the_row(gens[0]) and "tensor.collapse_shape" in names, names
    _broadcast_rewritten("\n  %x = tensor.extract_slice %r[0, 0] [8, 64] [1, 1]"
                         " : tensor<512x64xf32> to tensor<8x64xf32>", check)


def a_named_reader_reads_the_row_as_the_generic_of_its_maps():
    """Breaks if a reader is chosen by its op name: linalg.reduce and linalg.map state their maps
    through LinalgOp, so each becomes a generic reading the row, and nothing is copied in lane."""
    def check(copies, gens, names):
        assert [c.results[0].type for c in copies] == [_row_ty()], copies
        assert len(gens) == 3 and all(_reads_the_row(g) for g in gens), gens
        assert "linalg.reduce" not in names and "linalg.map" not in names, names
        reds = [g for g in gens if "reduction" in str(g.attributes["iterator_types"])]
        assert len(reds) == 1, gens
    _broadcast_rewritten("""
  %e = tensor.empty() : tensor<512xf32>
  %x = linalg.reduce ins(%r : tensor<512x64xf32>) outs(%e : tensor<512xf32>) dimensions = [1]
    (%a: f32, %b: f32) { %c = arith.addf %a, %b : f32
      linalg.yield %c : f32 }
  %y = linalg.map ins(%r, %w : tensor<512x64xf32>, tensor<512x64xf32>) outs(%i : tensor<512x64xf32>)
    (%a: f32, %b: f32, %c: f32) { %d = arith.subf %a, %b : f32
      linalg.yield %d : f32 }""", check)


def a_fold_over_the_lanes_of_the_broadcast_takes_the_row_copied_in_lane():
    """Breaks if a reader's map drops axis 0 when nothing else bounds the loop over it: a sum over the
    lanes of the broadcast would lose its trip count, so it reads the row copied in every lane."""
    def check(copies, gens, names):
        assert [c.results[0].type for c in copies] == [
            _row_ty(), ir.RankedTensorType.get([512, 64], ir.F32Type.get())], copies
        assert "linalg.reduce" in names, names
    _broadcast_rewritten("""
  %x = linalg.reduce ins(%r : tensor<512x64xf32>) outs(%v : tensor<64xf32>) dimensions = [0]
    (%a: f32, %b: f32) { %c = arith.addf %a, %b : f32
      linalg.yield %c : f32 }""", check)


def a_gather_into_a_uniform_row_is_one_lap():
    moves, _ = _lowered(f"%c: memref<64xf32, 1>, %u: memref<1x64xf32, 1>", """
  "pytorchsim.convert_layout"(%c, %u) <{added_dims = array<i64: 0>, dst_perm = array<i64: 0, 1>,
      src_perm = array<i64: 0>}> : (memref<64xf32, 1>, memref<1x64xf32, 1>) -> ()""")
    assert [(k, e, len(r)) for k, e, r in moves] == [("gather", True, 1)], moves


CASES = [a_column_into_a_uniform_row_is_no_crossing,
         a_row_picked_into_a_uniform_buffer_is_one_lane_sent_to_every_lane,
         a_broadcast_of_a_uniform_row_is_copied_in_lane,
         a_fold_result_also_stored_is_copied_in_lane,
         a_uniform_row_into_one_row_of_a_spread_buffer_stays_a_move,
         a_vector_broadcast_along_the_lanes_is_one_uniform_row_read_through_maps,
         a_reader_that_states_no_map_takes_the_row_copied_in_lane,
         a_named_reader_reads_the_row_as_the_generic_of_its_maps,
         a_fold_over_the_lanes_of_the_broadcast_takes_the_row_copied_in_lane,
         a_gather_into_a_uniform_row_is_one_lap]

if __name__ == "__main__":
    bad = 0
    for c in CASES:
        try:
            c()
            print(f"ok   {c.__name__}")
        except Exception as e:
            bad += 1
            print(f"FAIL {c.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)
