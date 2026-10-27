"""Every transfer is async and takes one wait before its tile is first read, at its own loop depth: a
wait at a reader inside a loop the transfer is not in would cut that loop into one block a trip for the
TOG -- 4097 blocks for a 1x2048x2048 matrix-vector product.
    python3 tests/passes/transfer_wait_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.passes.transforms import place_transfer_waits
from pytorchsim_triton_compiler.passes.utils import ir_utils
from pytorchsim_triton_compiler.dialect import pytorchsim  # noqa: E402
from pytorchsim_triton_compiler.dialect.pytorchsim import TransferOp, WaitOp  # noqa: E402

_MOD = """
module {{
  memref.global "private" @spad : memref<8x16xf32, 1>
  func.func @f(%x: memref<65536xf32>) -> f32 {{
    %c0 = arith.constant 0 : index
    %s = memref.get_global @spad : memref<8x16xf32, 1>
    "pytorchsim.transfer"(%x, %c0, %s, %c0, %c0) <{{async = true, direction = #pytorchsim<direction load>,
        dram_arg = 0 : i64,
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


_KEPT = []    # the modules _run parsed, alive while a case walks the ops it returned


def _run(subtile=""):
    from mlir.ir import Module
    module = Module.parse(_MOD.format(subtile=subtile))
    _KEPT.append(module)
    place_transfer_waits.run(module)
    assert module.operation.verify()
    ops = list(ir_utils.walk(module.operation))
    return ([o for o in ops if o.name == TransferOp.OPERATION_NAME][0],
            [o for o in ops if o.name == WaitOp.OPERATION_NAME])


def a_whole_transfer_takes_one_wait_before_the_loop_that_reads_it():
    """Breaks if the wait of a tile moved once sits inside the loop reading it (one TOG block a trip)."""
    tx, waits = _run()
    assert len(waits) == 1, waits
    after = list(waits[0].block.operations)
    nxt = after[[o.operation for o in after].index(waits[0]) + 1]
    assert waits[0].parent.name == "func.func" and nxt.name == "affine.for", (waits[0].parent.name, nxt.name)


def a_subtiled_transfer_takes_its_semaphore_and_one_wait_for_what_it_signals():
    """Breaks if pytorchsim-place-transfer-waits leaves the transfer on the semaphore it came with, waits on
    another one, or waits for other than what the transfer signals (moved_shape: its 4x16 subtile)."""
    tx, waits = _run(", subtile_size = [4, 16]")
    sem = tx.operands[pytorchsim.operand_number(tx, "sem")]
    assert len(waits) == 1, waits
    assert sem != tx.operands[pytorchsim.operand_number(tx, "dram_idx")], "semaphore not assigned"
    assert waits[0].opview.sem == sem, (waits[0], sem)
    assert _amount(waits[0]) == 4 * 16, _amount(waits[0])


_VIEWED = """
module {
  memref.global "private" @spad : memref<8x16xf32, 1>
  func.func @f(%x: memref<65536xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %s = memref.get_global @spad : memref<8x16xf32, 1>
    %v = memref.subview %s[0, 0] [8, 16] [1, 1]
        : memref<8x16xf32, 1> to memref<8x16xf32, strided<[16, 1]>, 1>
    "pytorchsim.transfer"(%x, %c0, %v, %c0, %c0) <{async = true, direction = #pytorchsim<direction load>,
        dram_arg = 0 : i64,
        dram_stride = [16, 1], subtile_size = [4, 16],
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}>
        : (memref<65536xf32>, index, memref<8x16xf32, strided<[16, 1]>, 1>, index, index) -> ()
    %r = affine.load %v[0, 0] : memref<8x16xf32, strided<[16, 1]>, 1>
    return %r : f32
  }
}
"""


def a_subtiled_transfers_wait_lands_in_the_buffer_its_tile_views():
    """Breaks if pytorchsim-place-transfer-waits builds a wait that names no buffer, or names the view and not
    its buffer."""
    from mlir.ir import Module
    module = Module.parse(_VIEWED)
    place_transfer_waits.run(module)
    assert module.operation.verify()
    ops = list(ir_utils.walk(module.operation))
    spad = [o for o in ops if o.name == "memref.get_global"][0].results[0]
    waits = [o for o in ops if o.name == WaitOp.OPERATION_NAME]
    assert len(waits) == 1 and list(waits[0].opview.landing) == [spad], waits


_LOOPED = """
module {{
  func.func @f(%y: memref<65536xf32>) {{
    %c0 = arith.constant 0 : index
    %z = arith.constant 0.0 : f32
    %s = memref.alloc() : memref<8x16xf32, 1>
    {loop_open}
      linalg.fill ins(%z : f32) outs(%s : memref<8x16xf32, 1>)
      "pytorchsim.transfer"(%y, %c0, %s, %c0, %c0) <{{async = true,
          direction = #pytorchsim<direction store>, dram_arg = 0 : i64,
          dram_stride = [16, 1], subtile_size = [4, 16],
          operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}>
          : (memref<65536xf32>, index, memref<8x16xf32, 1>, index, index) -> ()
    {loop_close}
    memref.dealloc %s : memref<8x16xf32, 1>
    return
  }}
}}
"""

_COUNTED = ("affine.for %i = 0 to 3 {", "}")


def _looped(loop):
    from mlir.ir import Module
    module = Module.parse(_LOOPED.format(loop_open=loop[0], loop_close=loop[1]))
    place_transfer_waits.run(module)
    assert module.operation.verify()
    ops = list(ir_utils.walk(module.operation))
    return module, ops, [o for o in ops if o.name == WaitOp.OPERATION_NAME]


def _amount(wait):
    from mlir.ir import IntegerAttr
    return IntegerAttr(wait.opview.amount.owner.attributes["value"]).value


def a_looped_store_waits_before_the_next_trips_writer_and_once_more_before_its_free():
    """Breaks if the store's last trip is still in flight when its buffer is freed (no tail wait),
    or if the wait before the next trip's writer is lost."""
    _m, ops, waits = _looped(_COUNTED)
    loop = [o for o in ops if o.name == "affine.for"][0]
    inner = [w for w in waits if w.parent.name == "affine.for"]
    tail = [w for w in waits if w.parent.name == "func.func"]
    assert len(inner) == 1 and inner[0].parent == loop, waits
    assert list(inner[0].block.operations)[0] == inner[0], "not before the trip's writer"
    assert len(tail) == 1, waits
    after = list(tail[0].block.operations)
    i = after.index(tail[0])
    assert after[i + 1].name == "memref.dealloc", after[i + 1]
    assert _amount(tail[0]) == 4 * 16 == _amount(inner[0]), (_amount(tail[0]), _amount(inner[0]))
    assert list(tail[0].opview.landing) == [ops[[o.name for o in ops].index("memref.alloc")].results[0]]


def a_store_no_writer_follows_waits_before_its_buffers_free():
    """Breaks if an async store nothing writes after keeps no wait: its free would end the buffer
    with the transfer still reading it."""
    _m, ops, waits = _looped(("", ""))
    assert len(waits) == 1, waits
    after = list(waits[0].block.operations)
    assert after[after.index(waits[0]) + 1].name == "memref.dealloc"
    assert _amount(waits[0]) == 64


def a_store_in_an_uncounted_loop_waits_at_the_end_of_each_trip():
    """Breaks if pytorchsim-place-transfer-waits states a tail wait whose batch is not a constant, or leaves a
    store in flight at its buffer's free: past a loop of no constant trip count each trip's store is waited at
    the body's end."""
    _m, ops, waits = _looped(("%c1 = arith.constant 1 : index\n    %u = \"test.n\"() : () -> index\n"
                              "    scf.for %i = %c0 to %u step %c1 {", "}"))
    assert len(waits) == 1, waits
    body = list(waits[0].block.operations)
    assert body[body.index(waits[0]) + 1].name == "scf.yield", body[body.index(waits[0]) + 1]
    assert _amount(waits[0]) == 64, _amount(waits[0])


_STEP = """
module {
  memref.global "private" @a : memref<8x16xf32, 1>
  memref.global "private" @b : memref<8x16xf32, 1>
  func.func @f(%x: memref<65536xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %a = memref.get_global @a : memref<8x16xf32, 1>
    %b = memref.get_global @b : memref<8x16xf32, 1>
    "pytorchsim.transfer"(%x, %c0, %a, %c0, %c0) <{async = true, direction = #pytorchsim<direction load>, dram_arg = 0 : i64,
        dram_stride = [16, 1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}>
        : (memref<65536xf32>, index, memref<8x16xf32, 1>, index, index) -> ()
    "pytorchsim.transfer"(%x, %c0, %b, %c0, %c0) <{async = true, direction = #pytorchsim<direction load>, dram_arg = 0 : i64,
        dram_stride = [16, 1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}>
        : (memref<65536xf32>, index, memref<8x16xf32, 1>, index, index) -> ()
    %u = affine.load %a[0, 0] : memref<8x16xf32, 1>
    %v = arith.addf %u, %u : f32
    %w = affine.load %b[0, 0] : memref<8x16xf32, 1>
    %r = arith.addf %v, %w : f32
    return %r : f32
  }
}
"""


def whole_loads_started_together_are_waited_before_the_first_of_them_is_read():
    """Breaks if the compute reading one whole-tile load overlaps another started with it: the v6e waits a
    step's operands before its body (attention decode's V under QK made the simulator faster than the v6e)."""
    from mlir.ir import Module
    module = Module.parse(_STEP)
    _KEPT.append(module)
    place_transfer_waits.run(module)
    ops = [o for o in ir_utils.walk(module.operation) if o.name in (WaitOp.OPERATION_NAME, "affine.load")]
    assert [o.name for o in ops] == [WaitOp.OPERATION_NAME] * 2 + ["affine.load"] * 2, [o.name for o in ops]


_GATHER = """
module {
  memref.global "private" @i : memref<8xi64, 1>
  memref.global "private" @g : memref<8x16xf32, 1>
  func.func @f(%x: memref<65536xf32>, %y: memref<65536xi64>) -> f32 {
    %c0 = arith.constant 0 : index
    %i = memref.get_global @i : memref<8xi64, 1>
    %g = memref.get_global @g : memref<8x16xf32, 1>
    "pytorchsim.transfer"(%y, %c0, %i, %c0, %c0) <{async = true, direction = #pytorchsim<direction load>, dram_arg = 1 : i64,
        dram_stride = [1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}>
        : (memref<65536xi64>, index, memref<8xi64, 1>, index, index) -> ()
    "pytorchsim.transfer"(%x, %c0, %g, %c0, %c0, %i) <{async = true,
        direction = #pytorchsim<direction load>, dram_arg = 0 : i64, dram_stride = [0, 1], indirect_dim = 0 : i64,
        offset_stride = 16 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}>
        : (memref<65536xf32>, index, memref<8x16xf32, 1>, index, index, memref<8xi64, 1>) -> ()
    %u = affine.load %g[0, 0] : memref<8x16xf32, 1>
    return %u : f32
  }
}
"""


def a_gather_waits_after_it_starts_not_where_its_indices_are_read():
    """Breaks if a gather joins the whole loads before it: it is the first read of its indices' load, so
    a wait of its own placed there would come before the gather starts (TOGSim: a wait no load signaled)."""
    from mlir.ir import Module
    module = Module.parse(_GATHER)
    _KEPT.append(module)
    place_transfer_waits.run(module)
    ops = [o.name for o in ir_utils.walk(module.operation)
           if o.name in (WaitOp.OPERATION_NAME, TransferOp.OPERATION_NAME, "affine.load")]
    assert ops == [TransferOp.OPERATION_NAME, WaitOp.OPERATION_NAME, TransferOp.OPERATION_NAME,
                   WaitOp.OPERATION_NAME, "affine.load"], ops


_BRANCHED = """
module {
  memref.global "private" @spad : memref<8x16xf32, 1>
  func.func @f(%x: memref<65536xf32>, %p: i1) -> f32 {
    %c0 = arith.constant 0 : index
    %s = memref.get_global @spad : memref<8x16xf32, 1>
    "pytorchsim.transfer"(%x, %c0, %s, %c0, %c0) <{async = true, direction = #pytorchsim<direction load>,
        dram_arg = 0 : i64,
        dram_stride = [16, 1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}>
        : (memref<65536xf32>, index, memref<8x16xf32, 1>, index, index) -> ()
    %r = scf.if %p -> (f32) {
      %a = affine.load %s[0, 0] : memref<8x16xf32, 1>
      scf.yield %a : f32
    } else {
      %b = affine.load %s[1, 0] : memref<8x16xf32, 1>
      scf.yield %b : f32
    }
    return %r : f32
  }
}
"""


def a_tile_both_arms_of_a_branch_read_is_waited_before_the_branch():
    """Breaks if the wait sits in the arm holding the first read: the other arm reads the tile unwaited,
    and its work-item ends with the load in flight (togsim.hpp refuses the TOG)."""
    from mlir.ir import Module
    module = Module.parse(_BRANCHED)
    _KEPT.append(module)
    place_transfer_waits.run(module)
    assert module.operation.verify()
    waits = [o for o in ir_utils.walk(module.operation) if o.name == WaitOp.OPERATION_NAME]
    assert len(waits) == 1, waits
    after = list(waits[0].block.operations)
    nxt = after[[o.operation for o in after].index(waits[0]) + 1]
    assert waits[0].parent.name == "func.func" and nxt.name == "scf.if", (waits[0].parent.name, nxt.name)


CASES = [a_whole_transfer_takes_one_wait_before_the_loop_that_reads_it,
         a_tile_both_arms_of_a_branch_read_is_waited_before_the_branch,
         a_gather_waits_after_it_starts_not_where_its_indices_are_read,
         whole_loads_started_together_are_waited_before_the_first_of_them_is_read,
         a_subtiled_transfer_takes_its_semaphore_and_one_wait_for_what_it_signals,
         a_subtiled_transfers_wait_lands_in_the_buffer_its_tile_views,
         a_looped_store_waits_before_the_next_trips_writer_and_once_more_before_its_free,
         a_store_no_writer_follows_waits_before_its_buffers_free,
         a_store_in_an_uncounted_loop_waits_at_the_end_of_each_trip]


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
