"""torchsim-pipeline-reductions chunks a reduce whose input is a scratchpad buffer an op allocates (a
space-1 memref.alloc, #230) as it chunks its global twin: the dealloc is no other user, and the chunk
and partials buffers it makes are allocated and freed where the input was.
    python3 tests/passes/pipeline_reductions_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
import pytorchsim_triton_compiler.dialects.torchsim  # noqa: E402,F401
from pytorchsim_triton_compiler.lowering.passes import p04c_pipeline_reductions as pipe  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime"))
from alloc_twin import to_allocs  # noqa: E402

_ROW_SUM = """module {
  memref.global @buf1_spad_8lane : memref<8x128xf32, 1>
  memref.global @buf0_spad_8lane : memref<8xf32, 1>
  func.func @k(%arg0: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @buf1_spad_8lane : memref<8x128xf32, 1>
    %1 = memref.get_global @buf0_spad_8lane : memref<8xf32, 1>
    "torchsim.transfer"(%arg0, %c0, %0, %c0, %c0) <{dma_kind = "MVIN", dram_arg = 0 : i64, dram_stride = [128, 1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}> : (memref<1024xf32>, index, memref<8x128xf32, 1>, index, index) -> ()
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%0 : memref<8x128xf32, 1>) outs(%1 : memref<8xf32, 1>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.addf %in, %out : f32
      linalg.yield %2 : f32
    }
    return
  }
}"""

_LIFE = ("memref.get_global", "memref.alloc", "memref.dealloc", "memref.global")


def _pipelined(as_allocs):
    """(each buffer's shape after the pass, in walk order; the ops left but the buffers' own; the
    shapes each dealloc frees)."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(_ROW_SUM)
        if as_allocs:
            to_allocs(m)
        stages, pipe._STAGES = pipe._STAGES, 2
        try:
            pipe.run(m)
        finally:
            pipe._STAGES = stages
        assert m.operation.verify()
        ops = list(walk(m.operation))
        root = "memref.alloc" if as_allocs else "memref.get_global"
        shapes = sorted(ir.MemRefType(o.results[0].type).shape for o in ops if o.name == root)
        rest = [o.name for o in ops if o.name not in _LIFE]
        freed = sorted(ir.MemRefType(o.operands[0].type).shape for o in ops
                       if o.name == "memref.dealloc")
        return shapes, rest, freed


def an_allocated_input_is_chunked_as_its_global_is():
    """Breaks if p04c takes only a get_global for a reduce's input, or counts its dealloc as a
    second user: the alloc twin is then left whole while the global one runs as a loop of chunks."""
    want, rest, _ = _pipelined(False)
    assert want == [[8], [8, 64], [8, 64]], want
    assert "scf.for" in rest, rest
    got, got_rest, freed = _pipelined(True)
    assert (got, got_rest) == (want, rest), (got, got_rest)
    assert freed == got, freed


CASES = [an_allocated_input_is_chunked_as_its_global_is]


def main():
    bad = 0
    for case in CASES:
        try:
            case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
