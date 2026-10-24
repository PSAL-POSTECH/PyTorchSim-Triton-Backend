"""pytorchsim-fit-to-numel cuts a scratchpad buffer an op allocates (a space-1 memref.alloc, #230) as
it cuts its global twin, and a dealloc of it is no reader that needs it whole.
    python3 tests/passes/fit_to_numel_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
import pytorchsim_triton_compiler.dialects.torchsim  # noqa: E402,F401
from pytorchsim_triton_compiler.lowering.passes import p04b_fit_to_numel as fit  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime"))
from alloc_twin import to_allocs  # noqa: E402

# reduce_bf16_tile_cast_in_its_readers at p04b's input: a 384-row tile on 512-row buffers
_REDUCE = """#map = affine_map<(d0, d1) -> (d1, d0)>
module attributes {"ttg.global-access-any-layout", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "torchsim:0", "ttg.threads-per-warp" = 128 : i32} {
  memref.global @buf6_spad_512lane : memref<512x8xbf16, 1>
  memref.global @buf5_spad_512lane : memref<512x8xf32, 1>
  memref.global @buf4_spad_512lane : memref<512x1x8xf32, 1>
  memref.global @buf3_spad_128lane : memref<128x1x8xf32, 1>
  memref.global @buf2_spad_1lane : memref<1x8xf32, 1>
  memref.global @buf1_spad_512lane : memref<512x8xf32, 1>
  memref.global @buf0_spad_512lane : memref<512x8xbf16, 1>
  func.func @reduce_bf16_tile_cast_in_its_readers_kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c384 = arith.constant 384 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c384_i32 = arith.constant 384 : i32
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = memref.get_global @buf6_spad_512lane : memref<512x8xbf16, 1>
    %1 = memref.get_global @buf5_spad_512lane : memref<512x8xf32, 1>
    %2 = memref.get_global @buf4_spad_512lane : memref<512x1x8xf32, 1>
    %3 = memref.get_global @buf3_spad_128lane : memref<128x1x8xf32, 1>
    %4 = memref.get_global @buf2_spad_1lane : memref<1x8xf32, 1>
    %5 = memref.get_global @buf1_spad_512lane : memref<512x8xf32, 1>
    %6 = memref.get_global @buf0_spad_512lane : memref<512x8xbf16, 1>
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [3072], strides: [1] : memref<*xbf16> to memref<3072xbf16>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [3072], strides: [1] : memref<*xbf16> to memref<3072xbf16>
    "torchsim.transfer"(%reinterpret_cast_1, %c0, %6, %c0, %c0, %c0, %c384) <{dma_kind = "MVIN", dram_arg = 0 : i64, dram_stride = [1, 384], masked_axes = [0], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<3072xbf16>, index, memref<512x8xbf16, 1>, index, index, index, index) -> ()
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : memref<512x8xbf16, 1>) outs(%5 : memref<512x8xf32, 1>) {
    ^bb0(%in: bf16, %out: f32):
      %7 = linalg.index 1 : index
      %8 = arith.index_cast %7 : index to i32
      %9 = arith.cmpi slt, %8, %c384_i32 : i32
      %10 = arith.extf %in : bf16 to f32
      %11 = arith.select %9, %10, %cst_0 : f32
      linalg.yield %11 : f32
    }
    linalg.fill ins(%cst_0 : f32) outs(%4 : memref<1x8xf32, 1>)
    %expand_shape = memref.expand_shape %5 [[0, 1, 2], [3]] output_shape [128, 1, 4, 8] : memref<512x8xf32, 1> into memref<128x1x4x8xf32, 1>
    linalg.fill ins(%cst_0 : f32) outs(%3 : memref<128x1x8xf32, 1>)
    linalg.reduce ins(%expand_shape : memref<128x1x4x8xf32, 1>) outs(%3 : memref<128x1x8xf32, 1>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %7 = arith.maxnumf %in, %init : f32
        linalg.yield %7 : f32
      }
    linalg.reduce ins(%3 : memref<128x1x8xf32, 1>) outs(%4 : memref<1x8xf32, 1>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %7 = arith.maxnumf %in, %init : f32
        linalg.yield %7 : f32
      }
    "torchsim.copy"(%4, %2) <{added_dims = array<i64: 0>, dst_perm = array<i64: 0, 1, 2>, src_perm = array<i64: 0, 1>}> : (memref<1x8xf32, 1>, memref<512x1x8xf32, 1>) -> ()
    %collapse_shape = memref.collapse_shape %2 [[0, 1], [2]] : memref<512x1x8xf32, 1> into memref<512x8xf32, 1>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %collapse_shape : memref<512x8xbf16, 1>, memref<512x8xf32, 1>) outs(%1 : memref<512x8xf32, 1>) {
    ^bb0(%in: bf16, %in_2: f32, %out: f32):
      %7 = arith.extf %in : bf16 to f32
      %8 = arith.subf %7, %in_2 : f32
      %9 = math.exp %8 : f32
      linalg.yield %9 : f32
    }
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<512x8xf32, 1>) outs(%5 : memref<512x8xf32, 1>) {
    ^bb0(%in: f32, %out: f32):
      %7 = linalg.index 1 : index
      %8 = arith.index_cast %7 : index to i32
      %9 = arith.cmpi slt, %8, %c384_i32 : i32
      %10 = arith.select %9, %in, %cst : f32
      linalg.yield %10 : f32
    }
    linalg.fill ins(%cst : f32) outs(%4 : memref<1x8xf32, 1>)
    linalg.fill ins(%cst : f32) outs(%3 : memref<128x1x8xf32, 1>)
    linalg.reduce ins(%expand_shape : memref<128x1x4x8xf32, 1>) outs(%3 : memref<128x1x8xf32, 1>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %7 = arith.addf %in, %init : f32
        linalg.yield %7 : f32
      }
    linalg.reduce ins(%3 : memref<128x1x8xf32, 1>) outs(%4 : memref<1x8xf32, 1>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %7 = arith.addf %in, %init : f32
        linalg.yield %7 : f32
      }
    "torchsim.copy"(%4, %2) <{added_dims = array<i64: 0>, dst_perm = array<i64: 0, 1, 2>, src_perm = array<i64: 0, 1>}> : (memref<1x8xf32, 1>, memref<512x1x8xf32, 1>) -> ()
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %collapse_shape : memref<512x8xf32, 1>, memref<512x8xf32, 1>) outs(%0 : memref<512x8xbf16, 1>) {
    ^bb0(%in: f32, %in_2: f32, %out: bf16):
      %7 = arith.divf %in, %in_2 : f32
      %8 = arith.truncf %7 : f32 to bf16
      linalg.yield %8 : bf16
    }
    "torchsim.transfer"(%reinterpret_cast, %c0, %0, %c0, %c0, %c0, %c384) <{dma_kind = "MVOUT", dram_arg = 1 : i64, dram_stride = [1, 384], masked_axes = [0], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<3072xbf16>, index, memref<512x8xbf16, 1>, index, index, index, index) -> ()
    return
  }
}"""

_LIFE = ("memref.get_global", "memref.alloc", "memref.dealloc")


def _fit(as_allocs):
    """(each buffer's type after the pass, in walk order; the ops left but the buffers' own; how
    many deallocs are left)."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(_REDUCE)
        if as_allocs:
            to_allocs(m)
        fit.run(m, grid=())
        assert m.operation.verify()
        ops = list(walk(m.operation))
        root = "memref.alloc" if as_allocs else "memref.get_global"
        types = [ir.MemRefType(o.results[0].type).shape for o in ops if o.name == root]
        rest = [o.name for o in ops if o.name not in _LIFE and o.name != "memref.global"]
        return types, rest, sum(o.name == "memref.dealloc" for o in ops)


def an_allocated_buffer_is_cut_as_its_global_is():
    """Breaks if p04b takes only a get_global for a buffer, or counts a dealloc as a reader of all
    of it: the alloc twin then keeps its 512 rows where the global is cut to 384."""
    want, rest, _ = _fit(False)
    assert [t[0] for t in want].count(384) == 5, want
    got, got_rest, deallocs = _fit(True)
    assert got == want, (got, want)
    assert got_rest == rest
    assert deallocs == 7, deallocs


def _source_freed_at_the_copy(tight):
    """The alloc twin, its exp generic reading %5 for %6 so the cast copy is %6's last use, and with
    `tight` %6's dealloc right after the copy (as ownership-based dealloc puts it). Returns the
    ops reading %6 or a view of it after its dealloc, and the generics' COPY_KEPT reasons."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(_REDUCE)
        allocs = to_allocs(m)
        a5, a6 = allocs[5], allocs[6]
        ops = list(walk(m.operation))
        cp, exp = [o for o in ops if o.name == "linalg.generic" and o.operands[0] == a6]
        exp.operands[0] = a5
        arg = exp.regions[0].blocks[0].arguments[0]
        arg.set_type(ir.MemRefType(a5.type).element_type)
        ext = list(arg.uses)[0].owner
        ext.results[0].replace_all_uses_with(arg)
        ext.erase()
        if tight:
            [o for o in ops if o.name == "memref.dealloc" and o.operands[0] == a6][0].move_after(cp)
        assert m.operation.verify()
        fit.run(m, grid=())
        assert m.operation.verify()
        body = [o.operation for o in m.body.operations[0].regions[0].blocks[0].operations]
        end = [k for k, o in enumerate(body) if o.name == "memref.dealloc" and o.operands[0] == a6]

        def of_a6(v):
            while v != a6 and isinstance(v.owner, ir.Operation) and v.owner.name in fit._VIEWS:
                v = v.owner.operands[0]
            return v == a6
        late = [o.name for top in body[end[0] + 1:] for o in [top] + walk(top)
                if any(of_a6(v) for v in o.operands)]
        kept = [str(o.attributes[fit.COPY_KEPT]) for o in walk(m.operation)
                if o.name == "linalg.generic" and fit.COPY_KEPT in o.attributes]
        return late, kept


def a_copy_whose_source_is_freed_before_a_reader_is_kept():
    """Breaks if p04b forwards a copy's readers to its source without asking where the source's
    life ends: with %6 freed right after the copy, the readers the copy fed then read freed %6."""
    late, kept = _source_freed_at_the_copy(True)
    assert late == [], f"uses of the source after its dealloc: {late}"
    assert any("life ends" in k for k in kept), kept
    late, kept = _source_freed_at_the_copy(False)
    assert late == [] and not any("life ends" in k for k in kept), (late, kept)


# ops_dot_cut_to_numel at p04b's input: a 128x384 @ 384x384 dot on blocks of 512, both loads filling 0 past
# 384
_DOT = """#map = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
module attributes {"ttg.global-access-any-layout", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "torchsim:0", "ttg.threads-per-warp" = 128 : i32} {
  func.func @ops_dot_cut_to_numel_kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c384 = arith.constant 384 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<512x128xbf16, 1>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<512x128xf32, 1>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<512x512xbf16, 1>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<512x128xbf16, 1>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [49152], strides: [1] : memref<*xbf16> to memref<49152xbf16>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [147456], strides: [1] : memref<*xbf16> to memref<147456xbf16>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [49152], strides: [1] : memref<*xbf16> to memref<49152xbf16>
    "torchsim.transfer"(%reinterpret_cast_4, %c0, %alloc_2, %c0, %c0, %c0, %c384) <{dma_kind = "MVIN", dram_arg = 0 : i64, dram_stride = [1, 384], masked_axes = [0], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<49152xbf16>, index, memref<512x128xbf16, 1>, index, index, index, index) -> ()
    "torchsim.transfer"(%reinterpret_cast_3, %c0, %alloc_1, %c0, %c0, %c0, %c384, %c0, %c384) <{dma_kind = "MVIN", dram_arg = 1 : i64, dram_stride = [1, 384], masked_axes = [1, 0], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 4, 0, 0, 0>}> : (memref<147456xbf16>, index, memref<512x512xbf16, 1>, index, index, index, index, index, index) -> ()
    linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<512x128xf32, 1>)
    "torchsim.dot"(%alloc_2, %alloc_1, %alloc_0) <{indexing_maps = [#map, #map1, #map2]}> : (memref<512x128xbf16, 1>, memref<512x512xbf16, 1>, memref<512x128xf32, 1>) -> ()
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%alloc_0 : memref<512x128xf32, 1>) outs(%alloc : memref<512x128xbf16, 1>) {
    ^bb0(%in: f32, %out: bf16):
      %0 = arith.truncf %in : f32 to bf16
      linalg.yield %0 : bf16
    }
    "torchsim.transfer"(%reinterpret_cast, %c0, %alloc, %c0, %c0, %c0, %c384) <{dma_kind = "MVOUT", dram_arg = 2 : i64, dram_stride = [1, 384], masked_axes = [0], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<49152xbf16>, index, memref<512x128xbf16, 1>, index, index, index, index) -> ()
    memref.dealloc %alloc : memref<512x128xbf16, 1>
    memref.dealloc %alloc_0 : memref<512x128xf32, 1>
    memref.dealloc %alloc_1 : memref<512x512xbf16, 1>
    memref.dealloc %alloc_2 : memref<512x128xbf16, 1>
    return
  }
}"""


def _dot_fit(fill):
    """The dot's operand shapes after the pass, both loads' masked_fill set to `fill` (bf16 bits)."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(_DOT)
        for o in walk(m.operation):
            if o.name == "torchsim.transfer" and "masked_fill" in o.attributes and \
                    ir.StringAttr(o.attributes["dma_kind"]).value == "MVIN":
                o.attributes["masked_fill"] = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), fill)
        fit.run(m, grid=())
        assert m.operation.verify()
        (dot,) = [o for o in walk(m.operation) if o.name == "torchsim.dot"]
        return [list(ir.MemRefType(v.type).shape) for v in dot.operands]


def a_dot_is_cut_to_what_its_store_keeps_and_its_product_reaches():
    """Breaks if p04b takes a dot as a reader of all of its buffers: N stays at the 512 of the block where the
    store keeps 384, and K at 512 where both loads fill 0 past 384."""
    assert _dot_fit(0) == [[384, 128], [384, 384], [384, 128]], _dot_fit(0)


def a_dot_keeps_k_where_no_padding_is_0():
    """Breaks if the K cut ignores the fill: with both loads filling 1.0 past 384 the product there is not 0.
    """
    one = 0x3F80
    assert _dot_fit(one) == [[512, 128], [384, 512], [384, 128]], _dot_fit(one)



# ops_broadcast_copy_cut_to_numel at p04b's input: x - max(x, axis 0) on 128x16 holding 8 columns, the max
# broadcast back over the rows by a torchsim.copy
_BROADCAST = """#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {"ttg.global-access-any-layout", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "torchsim:0", "ttg.threads-per-warp" = 128 : i32} {
  func.func @ops_broadcast_copy_cut_to_numel_kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0xFF800000 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x1x16xf32, 1>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x16xf32, 1>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<128x16xf32, 1>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<128x16xbf16, 1>
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1024], strides: [1] : memref<*xbf16> to memref<1024xbf16>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1] : memref<*xbf16> to memref<1024xbf16>
    "torchsim.transfer"(%reinterpret_cast_3, %c0, %alloc_2, %c0, %c0, %c0, %c8) <{dma_kind = "MVIN", dram_arg = 0 : i64, dram_stride = [8, 1], masked_axes = [1], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<1024xbf16>, index, memref<128x16xbf16, 1>, index, index, index, index) -> ()
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_2 : memref<128x16xbf16, 1>) outs(%alloc_1 : memref<128x16xf32, 1>) {
    ^bb0(%in: bf16, %out: f32):
      %0 = arith.extf %in : bf16 to f32
      linalg.yield %0 : f32
    }
    linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<1x16xf32, 1>)
    %expand_shape = memref.expand_shape %alloc_1 [[0, 1], [2]] output_shape [128, 1, 16] : memref<128x16xf32, 1> into memref<128x1x16xf32, 1>
    linalg.reduce ins(%expand_shape : memref<128x1x16xf32, 1>) outs(%alloc_0 : memref<1x16xf32, 1>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %0 = arith.maxnumf %in, %init : f32
        linalg.yield %0 : f32
      }
    "torchsim.copy"(%alloc_0, %alloc) <{added_dims = array<i64: 0>, dst_perm = array<i64: 0, 1, 2>, src_perm = array<i64: 0, 1>}> : (memref<1x16xf32, 1>, memref<128x1x16xf32, 1>) -> ()
    %collapse_shape = memref.collapse_shape %alloc [[0, 1], [2]] : memref<128x1x16xf32, 1> into memref<128x16xf32, 1>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_2, %collapse_shape : memref<128x16xbf16, 1>, memref<128x16xf32, 1>) outs(%alloc_2 : memref<128x16xbf16, 1>) {
    ^bb0(%in: bf16, %in_4: f32, %out: bf16):
      %0 = arith.extf %in : bf16 to f32
      %1 = arith.subf %0, %in_4 : f32
      %2 = arith.truncf %1 : f32 to bf16
      linalg.yield %2 : bf16
    }
    "torchsim.transfer"(%reinterpret_cast, %c0, %alloc_2, %c0, %c0, %c0, %c8) <{dma_kind = "MVOUT", dram_arg = 1 : i64, dram_stride = [8, 1], masked_axes = [1], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<1024xbf16>, index, memref<128x16xbf16, 1>, index, index, index, index) -> ()
    memref.dealloc %alloc : memref<128x1x16xf32, 1>
    memref.dealloc %alloc_0 : memref<1x16xf32, 1>
    memref.dealloc %alloc_1 : memref<128x16xf32, 1>
    memref.dealloc %alloc_2 : memref<128x16xbf16, 1>
    return
  }
}"""


def a_broadcast_copy_is_cut_through():
    """Breaks if p04b takes a broadcast copy as a reader of all of its source: the 1x16 max and every buffer
    cut with it keep 16 columns where the store keeps 8."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(_BROADCAST)
        fit.run(m, grid=())
        assert m.operation.verify()
        shapes = {tuple(ir.MemRefType(o.results[0].type).shape) for o in walk(m.operation) if o.name == "memref.alloc"}
    assert all(s[-1] == 8 for s in shapes), shapes


CASES = [an_allocated_buffer_is_cut_as_its_global_is,
         a_copy_whose_source_is_freed_before_a_reader_is_kept,
         a_dot_is_cut_to_what_its_store_keeps_and_its_product_reaches, a_dot_keeps_k_where_no_padding_is_0,
         a_broadcast_copy_is_cut_through]


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
