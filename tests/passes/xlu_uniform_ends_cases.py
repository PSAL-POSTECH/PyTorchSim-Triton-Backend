"""A copy from a UNIFORM buffer is each lane's copy from its own bank; a copy into one lists every
lane, so the move that carries it reaches every lane and says so (lower_copies).
    python3 tests/passes/xlu_uniform_ends_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects import torchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import lower_copies  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e.lib_xlu import (  # noqa: E402
    XLU_CROSSING, XLU_OPS, emitter_of, plan_of_xlu_op)
from pytorchsim_triton_compiler.lowering.passes.lib_cross_lane import (  # noqa: E402
    CROSS_LANE_KIND, CROSS_LANE_OP, CROSS_LANE_ROUNDS, EVERY_LANE)
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

torchsim.register()
LANES = 128
U = "memref<1x16xf32, 1>"
B = "memref<128x1x16xf32, 1>"
BROADCAST = f"""
  "torchsim.copy"(%u, %b) <{{added_dims = array<i64: 0>, dst_perm = array<i64: 0, 1, 2>,
      src_perm = array<i64: 0, 1>}}> : ({U}, {B}) -> ()"""


def _lowered(args, body):
    """`(move kind, whether it says EVERY_LANE, rounds)` of every move and the name of every op,
    after lower_copies ran on a function of `args` and `body`."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(f"func.func @f({args}) {{\n{body}\n  return\n}}\n")
        lower_copies.run(m, lanes=LANES)
        assert m.operation.verify()
        ops = [o for o in walk(m.operation)]
        moves = [(ir.StringAttr(o.attributes[CROSS_LANE_KIND]).value,
                  EVERY_LANE in o.attributes,
                  [tuple(ir.DenseI64ArrayAttr(r))
                   for r in ir.ArrayAttr(o.attributes.get(CROSS_LANE_ROUNDS, ir.ArrayAttr.get([])))])
                 for o in ops if o.name == CROSS_LANE_OP]
        return moves, [o.name for o in ops]


def a_column_into_a_uniform_row_is_no_crossing():
    pairs = [((f, 0), (lane, f)) for f in range(16) for lane in range(LANES)]
    assert all(plan_of_xlu_op(c, pairs, [16], [1, 16], LANES, False) is None
               for c in XLU_OPS if emitter_of(c) == XLU_CROSSING)
    moves, _ = _lowered(f"%c: memref<16xf32, 1>, %u: {U}", f"""
  "torchsim.copy"(%c, %u) <{{dst_perm = array<i64: 0, 1>, src_perm = array<i64: 0>}}>
      : (memref<16xf32, 1>, {U}) -> ()""")
    assert len(moves) == 1 and moves[0][0] != XLU_CROSSING and moves[0][1], moves
    assert {(r[4], r[5]) for r in moves[0][2]} == {(0, LANES)}, moves


def a_row_picked_into_a_uniform_buffer_is_one_lane_sent_to_every_lane():
    moves, _ = _lowered(f"%d: memref<8x4xf32, 1>, %u: memref<1x4xf32, 1>", """
  "torchsim.copy"(%d, %u) <{src_perm = array<i64: 0, 1>, dst_perm = array<i64: 0, 1>,
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
  "torchsim.crosslane_move"(%d, %u)
      {{torchsim.fold_by = "arith.maximumf", torchsim.move_kind = "fold"}}
      : (memref<2x16xf32, 1>, {U}) -> ()
  "torchsim.transfer"(%o, %c0, %u, %c0, %c0) <{{dma_kind = "MVOUT", dram_arg = 3 : i64,
      dram_stride = [16, 1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}>
      : (memref<16xf32>, index, {U}, index, index) -> (){BROADCAST}""")
    assert [k for k, _a, _r in moves] == ["fold"], moves
    assert "vector.transfer_write" in names, names


def a_uniform_row_into_one_row_of_a_spread_buffer_stays_a_move():
    """Each lane's copy from its own bank would write that row in all 128 lanes, over the rows the
    other lanes hold; the copy writes lane 5's alone."""
    moves, _ = _lowered(f"%u: {U}, %d: memref<128x16xf32, 1>", f"""
  "torchsim.copy"(%u, %d) <{{src_perm = array<i64: 0, 1>, dst_perm = array<i64: 0, 1>,
      static_offsets = array<i64: 5, 0>, static_sizes = array<i64: 1, 16>,
      static_strides = array<i64: 1, 1>, into_window}}> : ({U}, memref<128x16xf32, 1>) -> ()""")
    assert len(moves) == 1 and not moves[0][1], moves
    assert {(r[4], r[5]) for r in moves[0][2]} == {(5, 6)}, moves


CASES = [a_column_into_a_uniform_row_is_no_crossing,
         a_row_picked_into_a_uniform_buffer_is_one_lane_sent_to_every_lane,
         a_broadcast_of_a_uniform_row_is_copied_in_lane,
         a_fold_result_also_stored_is_copied_in_lane,
         a_uniform_row_into_one_row_of_a_spread_buffer_stays_a_move]

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
