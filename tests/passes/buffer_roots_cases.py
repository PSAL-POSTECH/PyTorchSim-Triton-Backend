"""What tells a scratchpad buffer from a view of one takes a buffer an op allocates (a space-1
memref.alloc, #230) as it takes a global: the UNIFORM count of 08-quality.json, and the buffer
lib_mxu's schedulers trust no other root to name.
    python3 tests/passes/buffer_roots_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from case_runner import run  # noqa: E402
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
import pytorchsim_triton_compiler.dialects.torchsim  # noqa: E402,F401
from pytorchsim_triton_compiler.metrics import quality  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime"))
from alloc_twin import to_allocs  # noqa: E402

_TWO = """module {
  memref.global @row_spad_1lane : memref<1x8xf32, 1>
  memref.global @tile_spad_128lane : memref<128x8xf32, 1>
  memref.global @image : memref<8xf32>
  func.func @f(%x: f32) {
    %c0 = arith.constant 0 : index
    %r = memref.get_global @row_spad_1lane : memref<1x8xf32, 1>
    %t = memref.get_global @tile_spad_128lane : memref<128x8xf32, 1>
    memref.store %x, %r[%c0, %c0] : memref<1x8xf32, 1>
    memref.store %x, %t[%c0, %c0] : memref<128x8xf32, 1>
    return
  }
}"""


def _on(as_allocs, check):
    """`check(module, func)` on _TWO, or on its alloc twin."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(_TWO)
        if as_allocs:
            to_allocs(m)
        assert m.operation.verify()
        fn = [o for o in m.body.operations if o.operation.name == "func.func"][0]
        return check(m, fn)


def an_allocated_one_lane_buffer_is_counted_as_its_global_is():
    """Breaks if lanes_of_a_buffer reads only a memref.global: the alloc twin then counts no
    one-lane buffer where the global form counts one, with an axis to spread."""
    want = _on(False, lambda m, _: quality.count_uniform(m))
    assert want == (1, 1), want
    got = _on(True, lambda m, _: quality.count_uniform(m))
    assert got == want, got


def _fresh_roots(m, fn):
    from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import lib_mxu
    from pytorchsim_triton_compiler.lowering.passes.lib_bank_layout import is_a_scratchpad_buffer
    entry = fn.regions[0].blocks[0]
    made = [o.results[0] for o in entry.operations
            if o.operation.name in ("memref.alloc", "memref.get_global")]
    return ([is_a_scratchpad_buffer(v) for v in made],
            [lib_mxu.is_fresh(lib_mxu.buffer_root(v)) for v in made],
            lib_mxu.is_fresh(entry.arguments[0]))


def an_allocated_buffer_is_a_root_no_other_names():
    """Breaks if the schedulers tell a fresh buffer by an op name list rather than the effect its
    op states, or the scratchpad's buffers miss an allocated one: alloc and global alike."""
    assert _on(False, _fresh_roots) == ([True, True], [True, True], False)
    assert _on(True, _fresh_roots) == ([True, True], [True, True], False)


CASES = [an_allocated_one_lane_buffer_is_counted_as_its_global_is,
         an_allocated_buffer_is_a_root_no_other_names]


def main():
    return run(CASES)


if __name__ == "__main__":
    sys.exit(main())
