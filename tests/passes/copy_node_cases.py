"""The resident mm's rule (contract/copy_node.plan, budget), the pass that finds its copy node (a
whole-argument load at the kernel's top feeding a dot) and the TOG's refusal of a work item that cannot fit
beside what it holds. python3 tests/passes/copy_node_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.contract import copy_node as cn  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p07b_plan_copy_node as p07b  # noqa: E402
from pytorchsim_triton_compiler.lowering.tog.passes import build_skeleton as bs  # noqa: E402

V6E = dict(in_bytes=2, acc_bytes=4, out_bytes=2, lanes=128, spad_lane_bytes=1 << 20, max_numel=1 << 26)


def _plan(M, K, N):
    from pytorchsim_triton_compiler.contract import target
    return cn.plan(M, N, K, **V6E, array_width=target.load("tpuv6e").matmul_width)


def the_larger_operand_is_held():
    got = _plan(4096, 4096, 4096 * 2)
    assert got.resident == 1 and got.tile >= 128, got


def a_tie_holds_b():
    assert _plan(4096, 4096, 4096).resident == 1


def the_smaller_is_held_when_the_larger_fills_the_scratchpad():
    """As v6e XLA does where one operand is the whole 128 MiB of VMEM. A buffer counts as the compiler cuts
    it: 8192x8192x5120's B rides a 8192x8192 block, cut to 8192x5120, so its 80 MiB is held as XLA holds it;
    192 columns stay a 256 block."""
    assert _plan(8192, 8192, 4096).resident == 1
    assert _plan(4096, 8192, 8192).resident == 0
    assert _plan(8192, 8192, 5120).resident == 1
    assert cn.fitted(5120, 128) == 5120 and cn.fitted(192, 128) == 256


def both_filling_the_scratchpad_hold_none_and_say_why():
    got = _plan(8192, 8192, 8192)
    assert got.resident is None and "A does not fit" in got.why and "B does not fit" in got.why, got


def a_vector_product_holds_none():
    assert _plan(1, 4096, 4096).resident is None


def the_tile_is_the_largest_that_fits_beside_the_held_operand():
    got = _plan(16384, 4096, 4096)
    assert got.resident == 1
    assert cn.fits(1, got.tile, 16384, 4096, 4096, **V6E) and not cn.fits(1, got.tile * 2, 16384, 4096, 4096, **V6E)


def the_tile_walks_no_padded_tail_where_a_smaller_one_divides_the_extent():
    """Breaks if a tail tile padded past the data is taken (it is computed whole): N = 3072 streamed by 2048
    walks 4096, by 1024 exactly 3072; N = 768 by 512 walks 1024, by 256 exactly 768."""
    assert _plan(4096, 2048, 3072).tile == 1024, _plan(4096, 2048, 3072)
    assert _plan(8192, 3072, 768).tile == 256, _plan(8192, 3072, 768)


def no_tile_is_narrower_than_the_matrix_unit():
    """Breaks if a tile narrower than Target.matmul_width (256 on tpuv6e) is taken for a tail it avoids: N =
    1152 by 128 is 9 pushes of a 256-wide unit, by 256 five (the last half full)."""
    assert _plan(4096, 384, 1152).tile == 256, _plan(4096, 384, 1152)


def an_elf_that_holds_takes_the_whole_bank():
    assert cn.budget(1 << 20, 2, True) == 1 << 20 and cn.budget(1 << 20, 2, False) == 1 << 19


_MOD = """
#map = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d0)>
module {{
  func.func @k(%a: memref<{na}xbf16>, %b: memref<1024xbf16>, %c: memref<1024xf32>) {{
    %c0 = arith.constant 0 : index
    %ta = memref.alloc() : memref<32x32xbf16, 1>
    %tb = memref.alloc() : memref<32x32xbf16, 1>
    %acc = memref.alloc() : memref<32x32xf32, 1>
    "torchsim.transfer"(%b, %c0, %tb, %c0, %c0) <{{dma_kind = "MVIN", dram_arg = 1 : i64, dram_stride = [1, 32],
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}>
        : (memref<1024xbf16>, index, memref<32x32xbf16, 1>, index, index) -> ()
    {pre}
    affine.for %i = 0 to 2 {{
      "torchsim.transfer"(%a, %c0, %ta, %c0, %c0) <{{dma_kind = "MVIN", dram_arg = 0 : i64, dram_stride = [1, 32],
          operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}>
          : (memref<{na}xbf16>, index, memref<32x32xbf16, 1>, index, index) -> ()
      "torchsim.dot"(%ta, %tb, %acc) <{{indexing_maps = [#map, #map1, #map2]}}>
          : (memref<32x32xbf16, 1>, memref<32x32xbf16, 1>, memref<32x32xf32, 1>) -> ()
      "torchsim.transfer"(%c, %c0, %acc, %c0, %c0) <{{dma_kind = "MVOUT", dram_arg = 2 : i64, dram_stride = [1, 32],
          operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}>
          : (memref<1024xf32>, index, memref<32x32xf32, 1>, index, index) -> ()
    }}
    return
  }}
}}
"""


def _stamped(na=2048, pre=""):
    from mlir.ir import ArrayAttr, DictAttr, IntegerAttr, Module, StringAttr
    module = Module.parse(_MOD.format(na=na, pre=pre))
    p07b.run(module)
    got = DictAttr(module.operation.attributes[p07b.COPY_NODE])
    return [IntegerAttr(a).value for a in ArrayAttr(got["args"])], StringAttr(got["why"]).value


def the_whole_load_at_the_top_is_the_copy_node():
    args, why = _stamped()
    assert args == [1] and "arg 0: loaded inside a loop" in why, (args, why)


def a_load_of_part_of_its_argument_is_none():
    from mlir.ir import Module
    module = Module.parse(_MOD.format(na=2048, pre="").replace("memref<1024xbf16>", "memref<4096xbf16>"))
    p07b.run(module)
    assert p07b.plan(module)[0] == [], p07b.plan(module)


def two_whole_loads_make_one_copy_node():
    """Breaks if a kernel whose streamed operand fits one tile (loaded whole at the top too) gets a copy node
    for each: XLA prefetches one operand, so the second load would be timed as a second exposed copy. A tie
    holds B."""
    from mlir.ir import ArrayAttr, DictAttr, IntegerAttr, Module
    a_in_loop = ('      "torchsim.transfer"(%a, %c0, %ta, %c0, %c0) <{{dma_kind = "MVIN", dram_arg = 0 : i64, '
                 'dram_stride = [1, 32],\n          operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}>\n'
                 '          : (memref<{na}xbf16>, index, memref<32x32xbf16, 1>, index, index) -> ()\n')
    assert a_in_loop in _MOD
    text = _MOD.replace(a_in_loop, "").format(na=1024, pre=(
        '"torchsim.transfer"(%a, %c0, %ta, %c0, %c0) <{dma_kind = "MVIN", dram_arg = 0 : i64, dram_stride = [1, 32], '
        'operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}> : '
        '(memref<1024xbf16>, index, memref<32x32xbf16, 1>, index, index) -> ()'))
    module = Module.parse(text)
    p07b.run(module)
    got = DictAttr(module.operation.attributes[p07b.COPY_NODE])
    args = [IntegerAttr(a).value for a in ArrayAttr(got["args"])]
    assert args == [1], args
    assert cn.order({0: 4, 1: 4}) == [1, 0] and cn.order({0: 8, 1: 4}) == [0, 1]


def an_argument_moved_twice_is_none():
    again = ('"torchsim.transfer"(%b, %c0, %tb, %c0, %c0) <{dma_kind = "MVIN", dram_arg = 1 : i64, '
             'dram_stride = [1, 32], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}> : '
             '(memref<1024xbf16>, index, memref<32x32xbf16, 1>, index, index) -> ()')
    args, why = _stamped(pre=again)
    assert args == [] and "moved by more than one transfer" in why, (args, why)


def a_work_item_that_fits_beside_the_held_buffer_passes():
    bs.refuse_a_spad_overrun({0, 2}, [256, 512, 256], [1], spad_bytes=1024)


def streamed_buffers_that_cannot_fit_beside_the_held_one_are_refused():
    """Breaks if a TOG whose loads would wait for spad the held bytes never free (a wedge) reaches TOGSim."""
    try:
        bs.refuse_a_spad_overrun({0, 2, 3}, [256, 512, 256, 64], [1], spad_bytes=1024)
    except RuntimeError as e:
        assert "576 B of buffers beside the 512 B held" in str(e), e
        return
    raise AssertionError("no refusal")


CASES = [the_larger_operand_is_held, a_tie_holds_b, the_smaller_is_held_when_the_larger_fills_the_scratchpad,
         both_filling_the_scratchpad_hold_none_and_say_why, a_vector_product_holds_none,
         the_tile_is_the_largest_that_fits_beside_the_held_operand,
         the_tile_walks_no_padded_tail_where_a_smaller_one_divides_the_extent, no_tile_is_narrower_than_the_matrix_unit,
         an_elf_that_holds_takes_the_whole_bank,
         the_whole_load_at_the_top_is_the_copy_node, a_load_of_part_of_its_argument_is_none,
         an_argument_moved_twice_is_none, two_whole_loads_make_one_copy_node,
         a_work_item_that_fits_beside_the_held_buffer_passes,
         streamed_buffers_that_cannot_fit_beside_the_held_one_are_refused]


def main():
    from mlir.ir import Context, Location
    failed = 0
    for case in CASES:
        try:
            with Context() as ctx, Location.unknown():
                ctx.allow_unregistered_dialects = True
                case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
