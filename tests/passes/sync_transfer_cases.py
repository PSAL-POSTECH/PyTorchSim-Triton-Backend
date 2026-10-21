"""A transfer is synchronous unless it is cut into subtiles: no async, no wait, its readers ordered
after it by the buffer. An async one whose wait sat at a reader inside a loop cut that loop into one
block a trip for the TOG -- 4097 blocks for a 1x2048x2048 matrix-vector product.
    python3 tests/passes/sync_transfer_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.lowering.passes import lib_ir, p07_make_transfers_async  # noqa: E402
from pytorchsim_triton_compiler.dialects.torchsim import TransferOp, WaitOp  # noqa: E402

_MOD = """
module {{
  memref.global "private" @spad : memref<8x16xf32, 1>
  func.func @f(%x: memref<65536xf32>) -> f32 {{
    %c0 = arith.constant 0 : index
    %s = memref.get_global @spad : memref<8x16xf32, 1>
    "torchsim.transfer"(%x, %c0, %s, %c0, %c0) <{{dma_kind = "MVIN", dram_arg = 0 : i64,
        dram_stride = [16, 1]{subtile}, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}>
        : (memref<65536xf32>, index, memref<8x16xf32, 1>, index, index) -> ()
    %z = arith.constant 0.0 : f32
    %r = affine.for %i = 0 to 8 iter_args(%acc = %z) -> (f32) {{
      %v = affine.load %s[%i, 0] : memref<8x16xf32, 1>
      %n = arith.addf %acc, %v : f32
      affine.yield %n : f32
    }}
    return %r : f32
  }}
}}
"""


def _run(subtile=""):
    from mlir.ir import Module
    module = Module.parse(_MOD.format(subtile=subtile))
    p07_make_transfers_async.run(module)
    assert module.operation.verify()
    ops = list(lib_ir.walk(module.operation))
    return ([o for o in ops if o.name == TransferOp.OPERATION_NAME][0],
            [o for o in ops if o.name == WaitOp.OPERATION_NAME])


def a_whole_transfer_stays_synchronous_and_has_no_wait():
    tx, waits = _run()
    assert "async" not in tx.attributes, tx
    assert not waits, waits


def a_subtiled_transfer_is_async_with_one_wait():
    from mlir.ir import BoolAttr
    tx, waits = _run(", subtile_size = [4, 16]")
    assert BoolAttr(tx.attributes["async"]).value, tx
    assert len(waits) == 1, waits


_VIEWED = """
module {
  memref.global "private" @spad : memref<8x16xf32, 1>
  func.func @f(%x: memref<65536xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %s = memref.get_global @spad : memref<8x16xf32, 1>
    %v = memref.subview %s[0, 0] [8, 16] [1, 1]
        : memref<8x16xf32, 1> to memref<8x16xf32, strided<[16, 1]>, 1>
    "torchsim.transfer"(%x, %c0, %v, %c0, %c0) <{dma_kind = "MVIN", dram_arg = 0 : i64,
        dram_stride = [16, 1], subtile_size = [4, 16],
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}>
        : (memref<65536xf32>, index, memref<8x16xf32, strided<[16, 1]>, 1>, index, index) -> ()
    %r = affine.load %v[0, 0] : memref<8x16xf32, strided<[16, 1]>, 1>
    return %r : f32
  }
}
"""


def a_subtiled_transfers_wait_lands_in_the_buffer_its_tile_views():
    """Breaks if p07 builds a wait that names no buffer, or names the view and not its buffer."""
    from mlir.ir import Module
    module = Module.parse(_VIEWED)
    p07_make_transfers_async.run(module)
    assert module.operation.verify()
    ops = list(lib_ir.walk(module.operation))
    spad = [o for o in ops if o.name == "memref.get_global"][0].results[0]
    waits = [o for o in ops if o.name == WaitOp.OPERATION_NAME]
    assert len(waits) == 1 and list(waits[0].opview.landing) == [spad], waits


CASES = [a_whole_transfer_stays_synchronous_and_has_no_wait,
         a_subtiled_transfer_is_async_with_one_wait,
         a_subtiled_transfers_wait_lands_in_the_buffer_its_tile_views]


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
