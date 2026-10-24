"""pytorchsim-hoist-loop-invariants on the strip loop legalize leaves: what leaves it, what a
write in the loop to the same buffer -- through any view -- keeps in, and where the pass runs.
    python3 tests/passes/hoist_loop_invariants_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import (  # noqa: E402
    p06_legalize, p06a_hoist_loop_invariants as p06a, p07_make_transfers_async, transform)
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

ROW = "memref<4xf32, 1>"
TILE = "memref<4x256xf32, 1>"
DRAM = "memref<4096xf32>"
SEG5 = "operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>"


def _strip(body, upper="256", args=""):
    """softmax's normalize loop as legalize leaves it: a tile row minus its row value, broadcast."""
    return ("memref.global @row : memref<4xf32, 1>\nmemref.global @tile : memref<4x256xf32, 1>\n"
            "memref.global @other : memref<4x256xf32, 1>\nfunc.func private @opaque()\n"
            f"func.func @f(%d: {DRAM}{args}) {{\n  %c0 = arith.constant 0 : index\n"
            f"  %c64 = arith.constant 64 : index\n  %m = memref.get_global @row : {ROW}\n"
            f"  %x = memref.get_global @tile : {TILE}\n  %o = memref.get_global @other : {TILE}\n"
            "  affine.for %r = 0 to 4 {\n"
            f"    affine.for %c = 0 to {upper} step 64 {{\n"
            f"      %v = affine.vector_load %x[%r, %c] : {TILE}, vector<64xf32>\n"
            f"      %s = affine.vector_load %m[%r] : {ROW}, vector<1xf32>\n"
            "      %b = vector.broadcast %s : vector<1xf32> to vector<64xf32>\n"
            "      %y = arith.subf %v, %b : vector<64xf32>\n"
            f"      affine.vector_store %y, %x[%r, %c] : {TILE}, vector<64xf32>\n" + body +
            "    }\n  }\n  return\n}\n")


def _run(text):
    m = ir.Module.parse(text)
    p06a.run(m)
    assert m.operation.verify()
    return m


def _loops(m):
    return [o for o in walk(m.operation) if o.name == "affine.for"]


def _in(loop, name):
    return [o for o in walk(loop) if o.name == name]


def _row_left(m):
    """Whether the row value and its broadcast left the strip loop for the row loop."""
    rows, strip = _loops(m)
    assert len(_in(rows, "vector.broadcast")) == 1, m
    return not _in(strip, "vector.broadcast")


def a_row_value_and_its_broadcast_leave_the_strip_loop():
    """The point of the pass: one vector<1> load and broadcast per row, not per strip. The strip
    loop writes @tile, which @row is not; breaks if two globals are taken to alias."""
    m = _run(_strip(""))
    assert _row_left(m), m
    assert len(_in(_loops(m)[1], "affine.vector_load")) == 1, m


def a_load_of_a_buffer_the_loop_writes_stays():
    m = _run(_strip(f"      affine.vector_store %s, %m[%r] : {ROW}, vector<1xf32>\n"))
    assert not _row_left(m), m


def a_load_whose_buffer_is_written_through_a_bank_view_stays():
    """Breaks if a write through a view is taken for a write to another buffer."""
    body = (f'      %bv = "pytorchsim.bank_view"(%m) : ({ROW}) -> memref<1x4xf32, 1>\n'
            "      affine.store %f, %bv[0, 0] : memref<1x4xf32, 1>\n")
    assert not _row_left(_run(_strip(body, args=", %f: f32"))), body


def a_store_through_another_buffers_bank_view_lets_the_row_leave():
    """Breaks if bank_view is not a view (its result is then a root that may alias @row)."""
    body = (f'      %bv = "pytorchsim.bank_view"(%o) : ({TILE}) -> memref<1x1024xf32, 1>\n'
            "      affine.store %f, %bv[0, 0] : memref<1x1024xf32, 1>\n")
    assert _row_left(_run(_strip(body, args=", %f: f32"))), body


def _copy(src, dst):
    return (f'      "pytorchsim.convert_layout"({src}, {dst}) <{{dst_perm = array<i64: 0, 1>, '
            f"src_perm = array<i64: 0, 1>}}> : ({TILE}, {TILE}) -> ()\n")


def a_copy_into_another_buffer_lets_the_row_leave():
    """Breaks if UnitState, which a copy writes, may be a buffer again (not addressable)."""
    assert _row_left(_run(_strip(_copy("%x", "%o"))))


def a_copy_into_the_row_buffer_keeps_it():
    body = (f'      "pytorchsim.convert_layout"(%x, %m) <{{dst_perm = array<i64: 0>, src_perm = array<i64: 0, 1>}}>'
            f" : ({TILE}, {ROW}) -> ()\n")
    assert not _row_left(_run(_strip(body)))


def _mvin(sram, sram_t):
    return (f'      "pytorchsim.transfer"(%d, %c0, {sram}, %c0, %c0) <{{direction = "load", '
            f"dram_arg = 0 : i64, dram_stride = [256, 1], {SEG5}}}> : ({DRAM}, index, {sram_t}, "
            "index, index) -> ()\n")


def a_transfer_into_another_buffer_lets_the_row_leave():
    """Breaks if the semaphore a transfer signals may be a buffer again (not addressable)."""
    assert _row_left(_run(_strip(_mvin("%o", TILE))))


def a_transfer_into_the_row_buffer_keeps_it():
    assert not _row_left(_run(_strip(_mvin("%m", ROW))))


def a_load_after_a_wait_in_the_loop_stays():
    """A wait lands writes to any buffer; upstream affine LICM hoists the load past it."""
    assert not _row_left(_run(_strip('      "pytorchsim.wait"(%c0, %c64) : (index, index) -> ()\n')))


def an_op_of_unknown_effects_in_the_loop_keeps_every_read():
    assert not _row_left(_run(_strip("      func.call @opaque() : () -> ()\n")))


def a_loop_that_may_not_run_keeps_its_read():
    """Breaks if a read leaves a loop of unknown trip count: it may then read what nothing loads."""
    m = _run(_strip("", upper="%n", args=", %n: index"))
    assert not _row_left(m), m


def _alias(body, dst_t):
    return (f"func.func @alias(%dst: {dst_t}, %out: memref<4xf32, 1>, %c: vector<1xf32>) {{\n"
            "  %c0 = arith.constant 0 : index\n  affine.for %i = 0 to 4 {\n" + body +
            "  }\n  return\n}\n")


def _all_inside(m, names):
    loop = _loops(m)[0]
    return all(_in(loop, n) for n in names)


def a_store_through_an_expand_shape_keeps_the_load_of_its_root():
    """The @alias the reviews reproduced: upstream affine LICM, asking only users of the same SSA
    memref, moves the store through the view out above the load of the root."""
    body = ("    %x = memref.load %dst[] : memref<f32, 1>\n"
            "    affine.store %x, %out[%i] : memref<4xf32, 1>\n"
            "    %flat = memref.expand_shape %dst [] output_shape [1] : memref<f32, 1> into "
            "memref<1xf32, 1>\n"
            "    affine.vector_store %c, %flat[%c0] : memref<1xf32, 1>, vector<1xf32>\n")
    m = _run(_alias(body, "memref<f32, 1>"))
    assert _all_inside(m, ("memref.load", "affine.vector_store")), m


def a_load_through_a_collapse_shape_of_a_written_root_stays():
    """The same through the other view: the root is written, the view read."""
    body = ("    %one = memref.collapse_shape %dst [] : memref<1xf32, 1> into memref<f32, 1>\n"
            "    %x = memref.load %one[] : memref<f32, 1>\n"
            "    affine.store %x, %out[%i] : memref<4xf32, 1>\n"
            "    affine.vector_store %c, %dst[%c0] : memref<1xf32, 1>, vector<1xf32>\n")
    m = _run(_alias(body, "memref<1xf32, 1>"))
    assert _all_inside(m, ("memref.load", "affine.vector_store")), m


def the_pass_runs_after_legalize_and_before_async():
    """Before make-transfers-async no wait stands between a transfer and its readers; legalize
    leaves the strip loops it hoists out of. Between them may run what makes no loop, like p06b."""
    ps = dict(transform.PHASES)[transform.MACHINE_KNOWN]
    assert ps.index(p06_legalize) < ps.index(p06a) < ps.index(p07_make_transfers_async), ps


CASES = [a_row_value_and_its_broadcast_leave_the_strip_loop,
         a_load_of_a_buffer_the_loop_writes_stays,
         a_load_whose_buffer_is_written_through_a_bank_view_stays,
         a_store_through_another_buffers_bank_view_lets_the_row_leave,
         a_copy_into_another_buffer_lets_the_row_leave,
         a_copy_into_the_row_buffer_keeps_it,
         a_transfer_into_another_buffer_lets_the_row_leave,
         a_transfer_into_the_row_buffer_keeps_it,
         a_load_after_a_wait_in_the_loop_stays,
         an_op_of_unknown_effects_in_the_loop_keeps_every_read,
         a_loop_that_may_not_run_keeps_its_read,
         a_store_through_an_expand_shape_keeps_the_load_of_its_root,
         a_load_through_a_collapse_shape_of_a_written_root_stays,
         the_pass_runs_after_legalize_and_before_async]


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
