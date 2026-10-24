"""lower_copies prices a copy of a fresh DMA load both ways: as its XLU move,
and as that load's extra pieces into the copy's destination; the cheaper one is what it emits.
    python3 tests/passes/lower_copies_cost_cases.py        exit 0 iff every case holds"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import suite_target  # noqa: E402
os.environ.setdefault("TORCHSIM_COMPILE_TARGET", suite_target.NAME)
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family import lower_copies, lower_dma  # noqa: E402,F401
from pytorchsim_triton_compiler.lowering.passes.lib_descriptor import read_cut  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

_MOD = """
module attributes {{"ttg.threads-per-warp" = 128 : i32}} {{
  func.func @k(%x: memref<{n}xf32>) {{
    %c0 = arith.constant 0 : index
    %b = memref.alloc() {{alignment = 64 : i64}} : memref<{kv}x{g}x{d}xf32, 1>
    %c = memref.alloc() {{alignment = 64 : i64}} : memref<128x{d}xf32, 1>
    "pytorchsim.transfer"(%x, %c0, %b, %c0, %c0) <{{direction = "load", dram_arg = 0 : i64,
        dram_stride = [{d}, 0, 1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}>
        : (memref<{n}xf32>, index, memref<{kv}x{g}x{d}xf32, 1>, index, index) -> ()
    "pytorchsim.copy"(%b, %c) <{{dst_perm = array<i64: 0, 1>, src_perm = array<i64: 0, 1, 2>}}>
        : (memref<{kv}x{g}x{d}xf32, 1>, memref<128x{d}xf32, 1>) -> ()
    memref.dealloc %b : memref<{kv}x{g}x{d}xf32, 1>
    memref.dealloc %c : memref<128x{d}xf32, 1>
    return
  }}
}}"""


def _lowered(g, d):
    """gqa's repeat of each of 128 / g kv rows g times, d wide, through lower-copies: (the moves,
    the cut transfers' pieces, the transfers' destination shapes)."""
    from mlir.ir import MemRefType, Module
    kv = 128 // g
    m = Module.parse(_MOD.format(n=kv * d, kv=kv, g=g, d=d))
    lower_copies.run(m, 128)
    ops = list(walk(m.operation))
    moves = sum(1 for o in ops if o.name == "pytorchsim.crosslane_move")
    tx = [o for o in ops if o.name == "pytorchsim.transfer"]
    cut = [read_cut(o) for o in tx]
    return moves, cut, [list(MemRefType(o.operands[2].type).shape) for o in tx]


def a_copy_its_load_does_in_fewer_cycles_becomes_that_load():
    """g = 16, d = 64: the load into the [128, 64] destination is 8 pieces, 7 * 27 cycles more, and
    the move 394 cycles. Breaks if lower-copies always moves a copy."""
    moves, cut, shapes = _lowered(16, 64)
    assert moves == 0 and shapes == [[128, 64]] and cut[0] is not None, (moves, cut, shapes)


def a_copy_the_unit_moves_in_fewer_cycles_stays_a_move():
    """g = 4, d = 64: 32 pieces, 31 * 27 cycles more, against a move of 202. Breaks if a copy whose
    source one plain load fills is folded into it whatever it costs."""
    moves, cut, shapes = _lowered(4, 64)
    assert moves == 1 and shapes == [[32, 4, 64]] and cut == [None], (moves, cut, shapes)


CASES = [a_copy_its_load_does_in_fewer_cycles_becomes_that_load,
         a_copy_the_unit_moves_in_fewer_cycles_stays_a_move]


def main():
    from mlir.ir import Context, Location
    bad = 0
    for case in CASES:
        ctx = Context()
        ctx.allow_unregistered_dialects = True
        with ctx, Location.unknown():
            try:
                case()
                print(f"ok    {case.__name__}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
