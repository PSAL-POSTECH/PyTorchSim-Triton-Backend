"""pytorchsim-pipeline-reductions chunks a reduce over an op-allocated scratchpad buffer as its global
twin, and a cut load whose last axis a chunk translates; one it does not stays whole.
    python3 tests/passes/pipeline_reductions_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
import pytorchsim_triton_compiler.dialects.pytorchsim  # noqa: E402,F401
from pytorchsim_triton_compiler.lowering.passes import p04c_pipeline_reductions as pipe  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_descriptor import read_cut  # noqa: E402
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
    "pytorchsim.transfer"(%arg0, %c0, %0, %c0, %c0) <{dma_kind = "MVIN", dram_arg = 0 : i64, dram_stride = [128, 1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}> : (memref<1024xf32>, index, memref<8x128xf32, 1>, index, index) -> ()
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


_CUT_SUM = """module {
  memref.global @buf1_spad_8lane : memref<8x128xf32, 1>
  memref.global @buf0_spad_8lane : memref<8xf32, 1>
  func.func @k(%arg0: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %cI = arith.constant IDX : index
    %0 = memref.get_global @buf1_spad_8lane : memref<8x128xf32, 1>
    %1 = memref.get_global @buf0_spad_8lane : memref<8xf32, 1>
    "pytorchsim.transfer"(%arg0, %cI, %0, %c0, %c0) <{cut = CUT, dma_kind = "MVIN", dram_arg = 0 : i64, dram_stride = [128, 1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}> : (memref<1024xf32>, index, memref<8x128xf32, 1>, index, index) -> ()
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%0 : memref<8x128xf32, 1>) outs(%1 : memref<8xf32, 1>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.addf %in, %out : f32
      linalg.yield %2 : f32
    }
    return
  }
}"""


def _chunked_cut(cut, idx):
    """(whether the trip loop's load keeps the input's cut digits, what a trip adds to its
    dram_idx), or None when the load is left whole: [8, 128] loaded with `cut` at `idx`, 2 trips."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(_CUT_SUM.replace("CUT", cut).replace("IDX", str(idx)))
        before = read_cut(next(o for o in walk(m.operation) if o.name == "pytorchsim.transfer"))
        stages, pipe._STAGES = pipe._STAGES, 2
        try:
            pipe.run(m)
        finally:
            pipe._STAGES = stages
        assert m.operation.verify()
        loops = [o for o in walk(m.operation) if o.name == "scf.for"]
        if not loops:
            return None
        tx = next(o for o in walk(loops[0]) if o.name == "pytorchsim.transfer")
        add = tx.operands[1].owner
        assert add.operands[0].owner.attributes["value"] == ir.IntegerAttr.get(
            ir.IndexType.get(), idx), add
        mul = add.operands[1].owner
        step = ir.IntegerAttr(mul.operands[1].owner.attributes["value"]).value
        return read_cut(tx) == before, step


def a_cut_off_the_chunked_axis_is_chunked_with_its_cut():
    """Breaks if p04c skips a cut load again: rows read at (r + 2) % 8 are a cut of axis 0, which a
    chunk of the columns does not move, so each trip steps 64 columns and keeps the cut."""
    cut = "[array<i64: 0, 128, 1024, 256>, array<i64>]"
    assert _chunked_cut(cut, 256) == (True, 64), _chunked_cut(cut, 256)


def a_chunk_that_holds_whole_laps_steps_no_dram():
    """Breaks if p04c skips a cut load, or steps a trip by chunk * stride: columns read at c % 32
    repeat every 32, so the second chunk of 64 reads the DRAM the first does."""
    cut = "[array<i64>, array<i64: 0, 1, 32, 0>]"
    assert _chunked_cut(cut, 0) == (True, 0), _chunked_cut(cut, 0)


def a_wrap_a_chunk_does_not_translate_stays_whole():
    """Breaks if p04c chunks a last axis whose cut a chunk does not translate: columns at
    (c + 40) % 128 wrap inside the first chunk and not the second, so no one step reads both."""
    assert _chunked_cut("[array<i64>, array<i64: 0, 1, 128, 40>]", 40) is None


CASES = [an_allocated_input_is_chunked_as_its_global_is,
         a_cut_off_the_chunked_axis_is_chunked_with_its_cut,
         a_chunk_that_holds_whole_laps_steps_no_dram,
         a_wrap_a_chunk_does_not_translate_stays_whole]


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
