"""The TOG sizes a masked load's clamped axis at run time (build_skeleton._live_extents) whenever
no barrier count has to be rebuilt from it: a synchronous load signals no semaphore, so it is sized
even when its semaphore operand is a constant every other op shares; an async one on such a
semaphore keeps the static tile. Without the first, a K of 768 padded to a 1024-row tile moves
1024 rows of B before the first compute.
    python3 tests/passes/tog_extent_cases.py        exit 0 iff every case holds"""
import os
import sys
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.lowering.tog.passes import build_skeleton  # noqa: E402
from pytorchsim_triton_compiler.lowering.tog.passes.build_tog import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.tog.passes._mlir_util import walk_ops  # noqa: E402

_MOD = """
module {
  memref.global "private" @spad : memref<16x8xbf16, 1>
  func.func @kernel(%x: memref<96xbf16>) -> vector<8xbf16> {
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %s = memref.get_global @spad : memref<16x8xbf16, 1>
    "torchsim.transfer"(%x, %c0, %s, %c0, %c0, %c0, %c6) <{dma_kind = "MVIN", dram_arg = 0 : i64,
        dram_stride = [1, 16], masked_axes = [0], masked_fill = 0 : i64,
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}>
        : (memref<96xbf16>, index, memref<16x8xbf16, 1>, index, index, index, index) -> ()
    %v = vector.load %s[%c0, %c0] : memref<16x8xbf16, 1>, vector<8xbf16>
    return %v : vector<8xbf16>
  }
}
"""

_Node = namedtuple("_Node", "tile_size is_async")


def _extents(is_async):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(_MOD, ctx)
        block = build_skeleton._kernel_block(module)
        tx = next(o for o in walk_ops(block) if o.operation.name == build_skeleton.TRANSFER)
        got = build_skeleton._live_extents(block, {tx.operation: _Node([16, 8], is_async)})
        return [axis for axis, _ in got.get(tx.operation, ())]


def a_synchronous_load_on_a_shared_constant_is_sized_at_run_time():
    got = _extents(False)
    assert got == [0], got


def an_async_load_on_a_shared_constant_keeps_its_static_tile():
    got = _extents(True)
    assert got == [], got


CASES = [a_synchronous_load_on_a_shared_constant_is_sized_at_run_time,
         an_async_load_on_a_shared_constant_keeps_its_static_tile]


def main():
    failed = 0
    for case in CASES:
        try:
            case()
            ok, why = True, ""
        except Exception as e:  # noqa: BLE001
            ok, why = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{case.__name__:64} {'PASS' if ok else 'FAIL'}  {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
