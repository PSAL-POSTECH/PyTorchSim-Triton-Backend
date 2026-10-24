"""The TOG keys a scratchpad buffer an op allocates (a space-1 memref.alloc, #230) as it keys its global
twin -- the same ids on every DMA and segment, the same bytes -- through views and a loop's carried
init; a scratchpad value whose buffer it cannot find, or a loop yielding back another buffer than it
took, is refused, never keyed -1; a scratchpad function argument still has no buffer.
    python3 tests/runtime/tog_buffer_roots_cases.py        exit 0 iff every case gets what it names"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pytorchsim_triton_compiler  # noqa: E402,F401
from pytorchsim_triton_compiler.lowering.tog.passes import build_skeleton as bs  # noqa: E402
from pytorchsim_triton_compiler.lowering.tog.passes import togsim_ops as ts  # noqa: E402
from pytorchsim_triton_compiler.lowering.tog.passes.build_tog import _find_kernel, ir  # noqa: E402
from alloc_twin import to_allocs  # noqa: E402

_T = "memref<8x8xf32, 1>"
_MVIN = ('"pytorchsim.transfer"(%d, %c0, {buf}, %c0, %c0) <{{dma_kind = "MVIN", dram_arg = 0 : i64, '
         'dram_stride = [8, 1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}> '
         ': (memref<64xf32>, index, ' + _T + ', index, index) -> ()')
_MVOUT = _MVIN.replace("MVIN", "MVOUT").replace("dram_arg = 0", "dram_arg = 1").replace("%d,", "%e,")


def _kernel(body, args=""):
    return f"""module {{
  memref.global @a : {_T}
  memref.global @b : {_T}
  func.func @kernel(%d: memref<64xf32>, %e: memref<64xf32>{args}) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %a = memref.get_global @a : {_T}
    %b = memref.get_global @b : {_T}
{body}
    return
  }}
}}"""


_CARRIED = _kernel(f"""    {_MVIN.format(buf="%a")}
    %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%v = %b) -> ({_T}) {{
      %x = memref.load %a[%i, %c0] : {_T}
      %w = memref.subview %v[0, 0] [8, 8] [1, 1] : {_T} to {_T}
      memref.store %x, %w[%i, %c0] : {_T}
      scf.yield %v : {_T}
    }}
    {_MVOUT.format(buf="%r")}""")


def _skeleton(text, as_allocs=False):
    """(each DMA's and segment's (read ids, write ids), the bytes table) the skeleton states, or
    the RuntimeError it refuses with."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(text)
        if as_allocs:
            to_allocs(m)
        try:
            bs.build_skeleton(m)
        except RuntimeError as e:
            return str(e)
        k = _find_kernel(m).operation

        def ids(o, name):
            a = ir.ArrayAttr(o.attributes[name])
            return [ir.IntegerAttr(a[i]).value for i in range(len(a))]
        got = [(ids(o, ts.ATTR_READ_BUFS), ids(o, ts.ATTR_WRITE_BUFS)) for o in _walk(k)
               if o.name in (ts.DMA, ts.COMPUTE)]
        return got, list(ir.DenseI64ArrayAttr(k.attributes[ts.ATTR_SPAD_BUFFER_BYTES]))


def _walk(op):
    for region in op.regions:
        for block in region.blocks:
            for o in block.operations:
                yield o.operation
                yield from _walk(o.operation)


def allocated_buffers_get_their_globals_ids_and_bytes():
    want = _skeleton(_CARRIED)
    assert not isinstance(want, str) and -1 not in sum((r + w for r, w in want[0]), []), want
    return _skeleton(_CARRIED, as_allocs=True) == want or _skeleton(_CARRIED, as_allocs=True)


def a_scratchpad_value_of_no_buffer_is_refused():
    made = f'    %m = "test.make"() : () -> {_T}\n    {_MVIN.format(buf="%m")}'
    return _skeleton(_kernel(made))


def a_loop_yielding_back_another_buffer_is_refused():
    return _skeleton(_CARRIED.replace(f"scf.yield %v : {_T}", f"scf.yield %a : {_T}"), True)


def a_scratchpad_argument_has_no_buffer():
    got = _skeleton(_kernel(f"    {_MVIN.format(buf='%s')}", args=f", %s: {_T}"))
    return got if isinstance(got, str) else got[0]


CASES = {
    "allocated_buffers_get_their_globals_ids_and_bytes":
        (allocated_buffers_get_their_globals_ids_and_bytes, True),
    "a_scratchpad_value_of_no_buffer_is_refused":
        (a_scratchpad_value_of_no_buffer_is_refused, "reaches no buffer"),
    "a_loop_yielding_back_another_buffer_is_refused":
        (a_loop_yielding_back_another_buffer_is_refused, "yields another back"),
    "a_scratchpad_argument_has_no_buffer":
        (a_scratchpad_argument_has_no_buffer, [([], [-1])]),
}


def main():
    failed = 0
    for name, (case, want) in CASES.items():
        try:
            got = case()
        except Exception as e:  # noqa: BLE001
            got = f"{type(e).__name__}: {e}"
        ok = (want in got) if isinstance(want, str) and isinstance(got, str) else got == want
        failed += not ok
        print(f"{name:50} {'PASS' if ok else 'FAIL'}  got {str(got)[:200]}  want {want}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
