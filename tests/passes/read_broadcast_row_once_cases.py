"""tritonpytorchsim-read-broadcast-row-once on GEMV's x tile as legalize leaves it: which reads move
to index 0, which buffers are left alone because some writer is not a stride-0 transfer, and where it runs.
    python3 tests/passes/read_broadcast_row_once_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import (  # noqa: E402
    p06a_hoist_loop_invariants as p06a, p06b_read_broadcast_row_once as p06b,
    p07_make_transfers_async, transform)
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

TILE = "memref<8x512xbf16, 1>"
DRAM = "memref<4096xbf16>"
SEG5 = "operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>"


def _transfer(buf, stride):
    return (f'    "torchsim.transfer"(%d, %c0, {buf}, %c0, %c0) <{{dma_kind = "MVIN", dram_arg = 0 : i64, '
            f"dram_stride = {stride}, {SEG5}}}> : ({DRAM}, index, {TILE}, index, index) -> ()\n")


def _kernel(writers, extra=""):
    """x's tile read row by row in a strip loop, written by `writers`."""
    return ("memref.global @x : memref<8x512xbf16, 1>\n"
            f"func.func @f(%d: {DRAM}, %v: vector<64xbf16>) {{\n  %c0 = arith.constant 0 : index\n"
            f"  %x = memref.get_global @x : {TILE}\n  %x2 = memref.get_global @x : {TILE}\n"
            + writers + extra +
            "  affine.for %r = 0 to 8 {\n    affine.for %c = 0 to 512 step 64 {\n"
            f"      %a = affine.vector_load %x[%r, %c] : {TILE}, vector<64xbf16>\n"
            f"      %b = affine.vector_load %x2[%r, %c] : {TILE}, vector<64xbf16>\n"
            "    }\n  }\n  return\n}\n")


def _rows(text):
    """The row index expression of each affine.vector_load after the pass, as a constant or None."""
    m = ir.Module.parse(text)
    p06b.run(m)
    assert m.operation.verify()
    out = []
    for o in walk(m.operation):
        if o.name == "affine.vector_load":
            e = ir.AffineMapAttr(o.attributes["map"]).value.results[0]
            out.append(e.value if isinstance(e, ir.AffineConstantExpr) else None)
    return out


def a_stride_zero_row_is_read_at_row_zero_through_every_alias():
    assert _rows(_kernel(_transfer("%x", "[0, 1]"))) == [0, 0]


def a_transfer_that_walks_the_row_axis_leaves_the_reads():
    assert _rows(_kernel(_transfer("%x", "[512, 1]"))) == [None, None]


def one_walking_writer_among_stride_zero_ones_leaves_the_reads():
    assert _rows(_kernel(_transfer("%x", "[0, 1]") + _transfer("%x2", "[512, 1]"))) == [None, None]


def a_store_into_the_tile_leaves_the_reads():
    store = f"  affine.vector_store %v, %x2[3, 0] : {TILE}, vector<64xbf16>\n"
    assert _rows(_kernel(_transfer("%x", "[0, 1]"), store)) == [None, None]


def a_view_of_the_tile_leaves_the_reads():
    view = (f"  %s = memref.subview %x[0, 0] [8, 64] [1, 1] : {TILE} to "
            "memref<8x64xbf16, strided<[512, 1]>, 1>\n")
    assert _rows(_kernel(_transfer("%x", "[0, 1]"), view)) == [None, None]


def the_pass_runs_after_hoisting_and_before_async():
    """After vectorize the read indices are bank-local, so row 0 is this bank's own copy."""
    ps = dict(transform.PHASES)[transform.MACHINE_KNOWN]
    assert ps.index(p06a) < ps.index(p06b) < ps.index(p07_make_transfers_async), ps


CASES = [a_stride_zero_row_is_read_at_row_zero_through_every_alias,
         a_transfer_that_walks_the_row_axis_leaves_the_reads,
         one_walking_writer_among_stride_zero_ones_leaves_the_reads,
         a_store_into_the_tile_leaves_the_reads,
         a_view_of_the_tile_leaves_the_reads,
         the_pass_runs_after_hoisting_and_before_async]


def main():
    failed = 0
    for case in CASES:
        try:
            with ir.Context(), ir.Location.unknown():
                case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
