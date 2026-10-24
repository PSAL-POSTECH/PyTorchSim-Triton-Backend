"""pytorchsim-share-spad-buffers: scratchpad buffers of one type whose lives do not overlap share one
global; overlapping ones, or of another type, keep their own.
    python3 tests/passes/share_spad_buffers_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p12b_share_spad_buffers as share  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

_HEAD = """module {
  memref.global "private" @a : memref<8x16xf32, 1>
  memref.global "private" @b : memref<8x16xf32, 1>
  memref.global "private" @c : memref<8x8xf32, 1>
  func.func @f(%x: f32, %d: memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %n = arith.constant 128 : index
    %s = arith.constant 0 : index
    %a = memref.get_global @a : memref<8x16xf32, 1>
    %b = memref.get_global @b : memref<8x16xf32, 1>
    %c = memref.get_global @c : memref<8x8xf32, 1>
"""
_TAIL = """    return
  }
}"""


def _run(body):
    """The globals left, and the global each get_global names, after the pass over `body`."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(_HEAD + body + _TAIL)
        share.run(m)
        assert m.operation.verify()
        ops = list(walk(m.operation))
        left = sorted(ir.StringAttr(o.attributes["sym_name"]).value for o in ops if o.name == "memref.global")
        named = [ir.FlatSymbolRefAttr(o.attributes["name"]).value for o in ops if o.name == "memref.get_global"]
        return left, named


def two_buffers_one_after_the_other_share_one_global():
    left, named = _run("""
    memref.store %x, %a[%c0, %c0] : memref<8x16xf32, 1>
    %ra = memref.load %a[%c0, %c0] : memref<8x16xf32, 1>
    memref.store %ra, %b[%c0, %c0] : memref<8x16xf32, 1>
    %rb = memref.load %b[%c0, %c0] : memref<8x16xf32, 1>
""")
    assert left == ["a", "c"], left
    assert named == ["a", "a", "c"], named


def two_buffers_alive_at_once_keep_their_own():
    left, _ = _run("""
    memref.store %x, %a[%c0, %c0] : memref<8x16xf32, 1>
    memref.store %x, %b[%c0, %c0] : memref<8x16xf32, 1>
    %ra = memref.load %a[%c0, %c0] : memref<8x16xf32, 1>
    %rb = memref.load %b[%c0, %c0] : memref<8x16xf32, 1>
""")
    assert left == ["a", "b", "c"], left


def a_buffer_of_another_type_shares_with_none():
    left, named = _run("""
    memref.store %x, %a[%c0, %c0] : memref<8x16xf32, 1>
    %ra = memref.load %a[%c0, %c0] : memref<8x16xf32, 1>
    memref.store %ra, %c[%c0, %c0] : memref<8x8xf32, 1>
""")
    assert "c" in left and named[2] == "c", (left, named)


def a_loop_that_touches_a_buffer_is_its_life_whole():
    """a is written in the loop's first op and read back before b's use there; b is touched later in
    the same loop. The next trip reads a again, so a's life is the loop and b cannot take its place."""
    left, _ = _run("""
    affine.for %i = 0 to 4 {
      memref.store %x, %a[%c0, %c0] : memref<8x16xf32, 1>
      %ra = memref.load %a[%c0, %c0] : memref<8x16xf32, 1>
      memref.store %ra, %b[%c0, %c0] : memref<8x16xf32, 1>
    }
""")
    assert left == ["a", "b", "c"], left


_TX = ('"pytorchsim.transfer"(%d, %c0, %a, %c0, %s) <{async = true, direction = "store", dram_arg = 1 : i64, '
       'dram_stride = [16, 1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}> '
       ': (memref<128xf32>, index, memref<8x16xf32, 1>, index, index) -> ()')


def _store_a_then_use_b(wait_before_b, wait_after_b):
    w = '    "pytorchsim.wait"(%s, %n) : (index, index) -> ()\n'
    return _run(f"""
    memref.store %x, %a[%c0, %c0] : memref<8x16xf32, 1>
    {_TX}
{w if wait_before_b else ""}    memref.store %x, %b[%c0, %c0] : memref<8x16xf32, 1>
    %rb = memref.load %b[%c0, %c0] : memref<8x16xf32, 1>
{w if wait_after_b else ""}""")


def an_async_store_keeps_its_buffer_until_its_wait():
    assert _store_a_then_use_b(True, False)[0] == ["a", "c"]
    assert _store_a_then_use_b(False, True)[0] == ["a", "b", "c"]


def an_async_store_with_no_wait_keeps_its_buffer_to_the_end():
    assert _store_a_then_use_b(False, False)[0] == ["a", "b", "c"]


def a_buffer_no_op_touches_takes_anothers_global():
    """b is named but only viewed (a dot's C its pops never write): it shares a's global although a
    lives over the whole function."""
    left, named = _run("""
    memref.store %x, %a[%c0, %c0] : memref<8x16xf32, 1>
    %vb = memref.subview %b[0, 0] [8, 8] [1, 1] : memref<8x16xf32, 1> to memref<8x8xf32, strided<[16, 1]>, 1>
    %ra = memref.load %a[%c0, %c0] : memref<8x16xf32, 1>
""")
    assert left == ["a", "c"] and named == ["a", "a", "c"], (left, named)


def an_allocated_buffer_keeps_its_own_and_lets_the_globals_share():
    """Breaks if p12b tells a buffer only by its global (#230): an allocated one then stops every
    global of its function from sharing. Its dealloc touches nothing, and it is never folded."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(_HEAD + """
    %e = memref.alloc() : memref<8x16xf32, 1>
    memref.store %x, %a[%c0, %c0] : memref<8x16xf32, 1>
    %ra = memref.load %a[%c0, %c0] : memref<8x16xf32, 1>
    memref.store %ra, %e[%c0, %c0] : memref<8x16xf32, 1>
    memref.store %ra, %b[%c0, %c0] : memref<8x16xf32, 1>
    %rb = memref.load %b[%c0, %c0] : memref<8x16xf32, 1>
    %re = memref.load %e[%c0, %c0] : memref<8x16xf32, 1>
    memref.dealloc %e : memref<8x16xf32, 1>
""" + _TAIL)
        share.run(m)
        assert m.operation.verify()
        ops = list(walk(m.operation))
        fn = [o for o in ops if o.name == "func.func"][0]
        assert share.SHARED_NOT not in fn.attributes, fn.attributes[share.SHARED_NOT]
        left = sorted(ir.StringAttr(o.attributes["sym_name"]).value for o in ops if o.name == "memref.global")
        assert left == ["a", "c"], left
        assert sum(o.name == "memref.alloc" for o in ops) == 1


CASES = [an_allocated_buffer_keeps_its_own_and_lets_the_globals_share,
         two_buffers_one_after_the_other_share_one_global, two_buffers_alive_at_once_keep_their_own,
         a_buffer_of_another_type_shares_with_none, a_loop_that_touches_a_buffer_is_its_life_whole,
         an_async_store_keeps_its_buffer_until_its_wait, an_async_store_with_no_wait_keeps_its_buffer_to_the_end,
         a_buffer_no_op_touches_takes_anothers_global]


def main():
    bad = 0
    for case in CASES:
        try:
            case()
            print(f"ok    {case.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
