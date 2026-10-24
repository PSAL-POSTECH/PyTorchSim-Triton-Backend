"""A UNIFORM buffer takes only values proven the same in every lane, or a FOLD across the lanes; a
DISTRIBUTED one never takes a scalar store (p07a_check_lane_uniformity).
    python3 tests/passes/lane_uniformity_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects import pytorchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family import lower_copies  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import transform  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import (  # noqa: E402
    p07a_check_lane_uniformity as p07a)

pytorchsim.register()
LANES = 128
D = "memref<128x16xf32, 1>"
U = "memref<1x16xf32, 1>"


def _func(args, body):
    return f"func.func @f({args}) {{\n{body}\n  return\n}}\n"


def _whys(text, before=()):
    """What p07a refuses in `text`, after the passes `before` ran on it."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(text)
        for p in before:
            p.run(m, lanes=LANES)
        assert m.operation.verify()
        return [why for _op, why in p07a.violations(m)]


def a_row_of_a_distributed_buffer_stored_into_a_uniform_one_is_refused():
    got = _whys(_func(f"%d: {D}, %u: {U}", """
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %v = affine.vector_load %d[%c5, %c0] : memref<128x16xf32, 1>, vector<16xf32>
  affine.vector_store %v, %u[%c0, %c0] : memref<1x16xf32, 1>, vector<16xf32>"""))
    assert len(got) == 1 and "lane-variant vector<16xf32>" in got[0], got


def what_a_uniform_buffer_holds_computes_a_uniform_value_through_a_loop():
    """Breaks if a loop-carried value is VARIANT for being carried: the greatest fixpoint proves the
    accumulator from its init and its yield."""
    got = _whys(_func(f"%u: {U}, %w: {U}, %s: f32", """
  %c0 = arith.constant 0 : index
  %z = arith.constant dense<0.0> : vector<16xf32>
  %b = vector.broadcast %s : f32 to vector<16xf32>
  %r = affine.for %i = 0 to 4 iter_args(%acc = %z) -> (vector<16xf32>) {
    %v = affine.vector_load %u[0, 0] : memref<1x16xf32, 1>, vector<16xf32>
    %x = arith.addf %acc, %v : vector<16xf32>
    %y = arith.mulf %x, %b : vector<16xf32>
    affine.yield %y : vector<16xf32>
  }
  affine.vector_store %r, %w[%c0, %c0] : memref<1x16xf32, 1>, vector<16xf32>"""))
    assert got == [], got


def a_loop_carrying_a_distributed_row_hands_back_a_variant_value():
    got = _whys(_func(f"%d: {D}, %w: {U}", """
  %c0 = arith.constant 0 : index
  %z = arith.constant dense<0.0> : vector<16xf32>
  %r = affine.for %i = 0 to 4 iter_args(%acc = %z) -> (vector<16xf32>) {
    %v = affine.vector_load %d[%i, 0] : memref<128x16xf32, 1>, vector<16xf32>
    %x = arith.addf %acc, %v : vector<16xf32>
    affine.yield %x : vector<16xf32>
  }
  affine.vector_store %r, %w[%c0, %c0] : memref<1x16xf32, 1>, vector<16xf32>"""))
    assert len(got) == 1 and "lane-variant" in got[0], got


def a_block_argument_carrying_a_distributed_row_is_variant_inside_its_loop():
    got = _whys(_func(f"%d: {D}, %w: {U}", """
  %c0 = arith.constant 0 : index
  %v = affine.vector_load %d[%c0, %c0] : memref<128x16xf32, 1>, vector<16xf32>
  %r = affine.for %i = 0 to 4 iter_args(%acc = %v) -> (vector<16xf32>) {
    affine.vector_store %acc, %w[%c0, %c0] : memref<1x16xf32, 1>, vector<16xf32>
    affine.yield %acc : vector<16xf32>
  }"""))
    assert len(got) == 1 and "lane-variant" in got[0], got


def the_lane_index_is_variant_and_a_register_position_is_not():
    """Breaks if an op of no operands and no effects is proven UNIFORM whatever it is: vlane_idx is
    one, and its value is the lane that runs it."""
    body = """
  %c0 = arith.constant 0 : index
  %l = "pytorchsim.vlane_idx"() : () -> vector<16xindex>
  %s = vector.step : vector<16xindex>
  %li = arith.index_cast %l : vector<16xindex> to vector<16xi32>
  %si = arith.index_cast %s : vector<16xindex> to vector<16xi32>
  affine.vector_store %si, %u[%c0, %c0] : memref<1x16xi32, 1>, vector<16xi32>
  affine.vector_store %li, %w[%c0, %c0] : memref<1x16xi32, 1>, vector<16xi32>"""
    got = _whys(_func("%u: memref<1x16xi32, 1>, %w: memref<1x16xi32, 1>", body))
    assert len(got) == 1 and "lane-variant vector<16xi32>" in got[0], got


def a_value_out_of_a_unit_state_is_variant():
    """Breaks if an op whose only effects are on a resource no value names is proven UNIFORM: a pop
    of a unit's state yields what that lane's unit holds."""
    got = _whys(_func(f"%u: {U}", """
  %t = "pytorchsim.vcix"() <{form = "v.i", imm = 0 : i64, opcode = 0 : i64, layouts = [],
      out_layout = unit, operandSegmentSizes = array<i32: 0, 0>}> : () -> tensor<1x16xf32>
  bufferization.materialize_in_destination %t in writable %u
      : (tensor<1x16xf32>, memref<1x16xf32, 1>) -> ()"""))
    assert len(got) == 1 and "lane-variant tensor<1x16xf32>" in got[0], got


def a_buffer_copy_from_a_distributed_row_into_a_uniform_buffer_is_refused():
    got = _whys(_func(f"%d: {D}, %u: {U}", """
  %r = memref.subview %d[5, 0] [1, 16] [1, 1]
      : memref<128x16xf32, 1> to memref<1x16xf32, strided<[16, 1], offset: 80>, 1>
  memref.copy %r, %u : memref<1x16xf32, strided<[16, 1], offset: 80>, 1> to memref<1x16xf32, 1>"""))
    assert len(got) == 1 and "written from DISTRIBUTED buffer" in got[0], got


def a_fold_across_the_lanes_may_write_a_uniform_buffer():
    got = _whys(_func(f"%d: {D}, %u: {U}", """
  "pytorchsim.crosslane_move"(%d, %u) {pytorchsim.fold_by = "arith.addf", pytorchsim.move_kind = "fold"}
      : (memref<128x16xf32, 1>, memref<1x16xf32, 1>) -> ()"""))
    assert got == [], got


def a_copy_of_one_lane_row_into_a_uniform_buffer_reaches_every_lane():
    """lower-copies carries the row into every lane, the move says so with EVERY_LANE, and p07a
    takes it; before, the row reached the one lane the destination's bank is."""
    got = _whys(_func(f"%d: {D}, %u: {U}", """
  "pytorchsim.copy"(%d, %u) <{src_perm = array<i64: 0, 1>, dst_perm = array<i64: 0, 1>,
      static_offsets = array<i64: 5, 0>, static_sizes = array<i64: 1, 16>,
      static_strides = array<i64: 1, 1>}> : (memref<128x16xf32, 1>, memref<1x16xf32, 1>) -> ()"""),
                  before=(lower_copies,))
    assert got == [], got


def every_lane_is_read_off_the_pairs_a_move_carries():
    """Breaks if EVERY_LANE is stamped from the destination's placement: pairs that reach one lane,
    or take a row from two sources, state nothing about every lane."""
    from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family.lib_xlu import reaches_every_lane
    full = [((5, k), (lane, k)) for k in range(4) for lane in range(LANES)]
    assert reaches_every_lane(full, LANES)
    assert not reaches_every_lane([((5, k), (0, k)) for k in range(4)], LANES)
    assert not reaches_every_lane([((lane, 0), (lane, 0)) for lane in range(LANES)], LANES)
    assert not reaches_every_lane([], LANES)


def a_move_into_a_uniform_buffer_that_is_not_every_lane_is_refused():
    got = _whys(_func(f"%d: {D}, %u: {U}", """
  "pytorchsim.crosslane_move"(%d, %u) {pytorchsim.move_kind = "permute"}
      : (memref<128x16xf32, 1>, memref<1x16xf32, 1>) -> ()"""))
    assert len(got) == 1 and "move between lanes writes UNIFORM" in got[0], got


def a_scalar_store_into_a_distributed_buffer_is_refused_and_into_a_uniform_one_is_not():
    got = _whys(_func(f"%d: {D}, %u: {U}, %s: f32", """
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  memref.store %s, %u[%c0, %c3] : memref<1x16xf32, 1>
  memref.store %s, %d[%c3, %c0] : memref<128x16xf32, 1>"""))
    assert len(got) == 1 and "scalar f32 is stored into DISTRIBUTED" in got[0], got


def a_scalar_read_of_a_distributed_buffer_is_refused_and_of_a_uniform_one_is_not():
    """Breaks if only writes are checked: a scalar load sees one lane's bank, whichever op spells it."""
    got = _whys(_func(f"%d: {D}, %u: {U}, %w: {U}", """
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %a = memref.load %d[%c3, %c0] : memref<128x16xf32, 1>
  %b = affine.load %d[3, 0] : memref<128x16xf32, 1>
  %c = memref.load %u[%c0, %c3] : memref<1x16xf32, 1>
  memref.store %c, %w[%c0, %c0] : memref<1x16xf32, 1>"""))
    assert len(got) == 2 and all("read from DISTRIBUTED" in w for w in got), got


def an_index_or_a_rank_0_vector_stored_into_a_distributed_buffer_is_a_scalar_store():
    """Breaks if only integer and float types count as scalar."""
    got = _whys(_func("%d: memref<128x16xindex, 1>, %e: memref<128x16xf32, 1>, %s: f32", """
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  memref.store %c3, %d[%c3, %c0] : memref<128x16xindex, 1>
  %v = vector.broadcast %s : f32 to vector<f32>
  vector.transfer_write %v, %e[%c3, %c0] : vector<f32>, memref<128x16xf32, 1>"""))
    assert len(got) == 2 and all("stored into DISTRIBUTED" in w for w in got), got


def a_scratchpad_writer_with_a_body_of_its_own_is_refused_unread():
    """Breaks if an op with regions that writes the scratchpad is passed unchecked."""
    got = _whys(_func(f"%d: {D}, %s: f32", """
  linalg.fill ins(%s : f32) outs(%d : memref<128x16xf32, 1>)"""))
    assert len(got) == 1 and "body of its own" in got[0], got


def the_check_ends_the_transform_phases():
    assert transform.PHASES[-1][1][-1] is p07a, transform.PHASES[-1]


CASES = [a_row_of_a_distributed_buffer_stored_into_a_uniform_one_is_refused,
         what_a_uniform_buffer_holds_computes_a_uniform_value_through_a_loop,
         a_loop_carrying_a_distributed_row_hands_back_a_variant_value,
         a_block_argument_carrying_a_distributed_row_is_variant_inside_its_loop,
         the_lane_index_is_variant_and_a_register_position_is_not,
         a_value_out_of_a_unit_state_is_variant,
         a_buffer_copy_from_a_distributed_row_into_a_uniform_buffer_is_refused,
         a_fold_across_the_lanes_may_write_a_uniform_buffer,
         a_copy_of_one_lane_row_into_a_uniform_buffer_reaches_every_lane,
         every_lane_is_read_off_the_pairs_a_move_carries,
         a_move_into_a_uniform_buffer_that_is_not_every_lane_is_refused,
         a_scalar_store_into_a_distributed_buffer_is_refused_and_into_a_uniform_one_is_not,
         a_scalar_read_of_a_distributed_buffer_is_refused_and_of_a_uniform_one_is_not,
         an_index_or_a_rank_0_vector_stored_into_a_distributed_buffer_is_a_scalar_store,
         a_scratchpad_writer_with_a_body_of_its_own_is_refused_unread,
         the_check_ends_the_transform_phases]

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
