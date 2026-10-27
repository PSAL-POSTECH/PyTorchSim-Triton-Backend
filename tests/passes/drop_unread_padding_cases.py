"""pytorchsim-drop-unread-padding cuts each tile on tensors, before bufferize, so bufferize allocates the cut
buffers; a copy its mask no longer cuts is forwarded to its readers unless its input is written again.
    python3 tests/passes/drop_unread_padding_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from mlir.dialects import linalg, memref  # noqa: E402
from pytorchsim_triton_compiler.dialects import pytorchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import a05_drop_unread_padding as fit  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import a06_bufferize  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

# reduce_bf16_tile_cast_in_its_readers at drop-unread-padding's input: a 384-row tile on 512-row tiles
_REDUCE = """#map = affine_map<(d0, d1) -> (d1, d0)>
module attributes {"ttg.global-access-any-layout", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "pytorchsim:0", "ttg.threads-per-warp" = 128 : i32} {
  func.func @reduce_bf16_tile_cast_in_its_readers_kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant 0xFF800000 : f32
    %c384_i32 = arith.constant 384 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c384 = arith.constant 384 : index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [3072], strides: [1] : memref<*xbf16> to memref<3072xbf16>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [3072], strides: [1] : memref<*xbf16> to memref<3072xbf16>
    %0 = bufferization.alloc_tensor() : tensor<512x8xbf16>
    %1 = "pytorchsim.transfer"(%reinterpret_cast_1, %c0, %0, %c0, %c0, %c0, %c384) <{async = true, direction = "load", dram_arg = 0 : i64, dram_stride = [1, 384], masked_axes = [0], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<3072xbf16>, index, tensor<512x8xbf16>, index, index, index, index) -> tensor<512x8xbf16>
    %2 = tensor.empty() : tensor<512x8xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<512x8xbf16>) outs(%2 : tensor<512x8xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %17 = linalg.index 1 : index
      %18 = arith.index_cast %17 : index to i32
      %19 = arith.cmpi slt, %18, %c384_i32 : i32
      %20 = arith.extf %in : bf16 to f32
      %21 = arith.select %19, %20, %cst : f32
      linalg.yield %21 : f32
    } -> tensor<512x8xf32>
    %4 = tensor.empty() : tensor<1x8xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x8xf32>) -> tensor<1x8xf32>
    %expanded = tensor.expand_shape %3 [[0, 1, 2], [3]] output_shape [128, 1, 4, 8] : tensor<512x8xf32> into tensor<128x1x4x8xf32>
    %6 = tensor.empty() : tensor<128x1x8xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<128x1x8xf32>) -> tensor<128x1x8xf32>
    %reduced = linalg.reduce ins(%expanded : tensor<128x1x4x8xf32>) outs(%7 : tensor<128x1x8xf32>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %17 = arith.maxnumf %in, %init : f32
        linalg.yield %17 : f32
      }
    %reduced_2 = linalg.reduce ins(%reduced : tensor<128x1x8xf32>) outs(%5 : tensor<1x8xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %17 = arith.maxnumf %in, %init : f32
        linalg.yield %17 : f32
      }
    %8 = tensor.empty() : tensor<512x1x8xf32>
    %9 = "pytorchsim.convert_layout"(%reduced_2, %8) <{added_dims = array<i64: 0>, dst_perm = array<i64: 0, 1, 2>, src_perm = array<i64: 0, 1>}> : (tensor<1x8xf32>, tensor<512x1x8xf32>) -> tensor<512x1x8xf32>
    %collapsed = tensor.collapse_shape %9 [[0, 1], [2]] : tensor<512x1x8xf32> into tensor<512x8xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %collapsed : tensor<512x8xbf16>, tensor<512x8xf32>) outs(%2 : tensor<512x8xf32>) {
    ^bb0(%in: bf16, %in_7: f32, %out: f32):
      %17 = arith.extf %in : bf16 to f32
      %18 = arith.subf %17, %in_7 : f32
      %19 = math.exp %18 : f32
      linalg.yield %19 : f32
    } -> tensor<512x8xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<512x8xf32>) outs(%2 : tensor<512x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = linalg.index 1 : index
      %18 = arith.index_cast %17 : index to i32
      %19 = arith.cmpi slt, %18, %c384_i32 : i32
      %20 = arith.select %19, %in, %cst_0 : f32
      linalg.yield %20 : f32
    } -> tensor<512x8xf32>
    %12 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<1x8xf32>) -> tensor<1x8xf32>
    %expanded_3 = tensor.expand_shape %11 [[0, 1, 2], [3]] output_shape [128, 1, 4, 8] : tensor<512x8xf32> into tensor<128x1x4x8xf32>
    %13 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<128x1x8xf32>) -> tensor<128x1x8xf32>
    %reduced_4 = linalg.reduce ins(%expanded_3 : tensor<128x1x4x8xf32>) outs(%13 : tensor<128x1x8xf32>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %17 = arith.addf %in, %init : f32
        linalg.yield %17 : f32
      }
    %reduced_5 = linalg.reduce ins(%reduced_4 : tensor<128x1x8xf32>) outs(%12 : tensor<1x8xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %17 = arith.addf %in, %init : f32
        linalg.yield %17 : f32
      }
    %14 = "pytorchsim.convert_layout"(%reduced_5, %8) <{added_dims = array<i64: 0>, dst_perm = array<i64: 0, 1, 2>, src_perm = array<i64: 0, 1>}> : (tensor<1x8xf32>, tensor<512x1x8xf32>) -> tensor<512x1x8xf32>
    %collapsed_6 = tensor.collapse_shape %14 [[0, 1], [2]] : tensor<512x1x8xf32> into tensor<512x8xf32>
    %15 = tensor.empty() : tensor<512x8xbf16>
    %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%10, %collapsed_6 : tensor<512x8xf32>, tensor<512x8xf32>) outs(%15 : tensor<512x8xbf16>) {
    ^bb0(%in: f32, %in_7: f32, %out: bf16):
      %17 = arith.divf %in, %in_7 : f32
      %18 = arith.truncf %17 : f32 to bf16
      linalg.yield %18 : bf16
    } -> tensor<512x8xbf16>
    "pytorchsim.transfer"(%reinterpret_cast, %c0, %16, %c0, %c0, %c0, %c384) <{async = true, direction = "store", dram_arg = 1 : i64, dram_stride = [1, 384], masked_axes = [0], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<3072xbf16>, index, tensor<512x8xbf16>, index, index, index, index) -> ()
    return
  }
}"""


def _parsed(text):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        return ctx, ir.Module.parse(text)


def _fresh_shapes(m):
    """The shape of each tile the module makes (fit._is_fresh), in walk order."""
    return [list(ir.RankedTensorType(o.results[0].type).shape) for o in walk(m.operation)
            if fit._is_fresh(o)]


def bufferize_allocates_the_cut_tiles():
    """Breaks if drop-unread-padding leaves a tile bufferize allocates at its Triton block: a 512-row buffer
    is left where the store keeps 384."""
    ctx, m = _parsed(_REDUCE)
    with ctx, ir.Location.unknown():
        fit.run(m, grid=())
        assert m.operation.verify()
        a06_bufferize.run(m)
        rows = [ir.MemRefType(o.results[0].type).shape[0] for o in walk(m.operation)
                if isinstance(o.opview, memref.AllocOp)]
    assert 512 not in rows and rows.count(384) == 4, rows


def _forwarded(write_again):
    """(generics left, their COPY_KEPT reasons) of _REDUCE after the pass; with `write_again` a fill writes
    the loaded tile in place right after the cast copy that reads it, ahead of the copy's readers, and the
    exp reads the fill."""
    ctx, m = _parsed(_REDUCE)
    with ctx, ir.Location.unknown():
        from mlir.dialects import arith
        if write_again:
            cp, exp = [o for o in walk(m.operation) if isinstance(o.opview, linalg.GenericOp)][:2]
            with ir.InsertionPoint(cp):
                zero = arith.ConstantOp(ir.BF16Type.get(), 0.0).result
                filled = linalg.fill(zero, outs=[cp.operands[0]])
            filled.owner.move_after(cp)
            exp.operands[0] = filled
            assert m.operation.verify()
        fit.run(m, grid=())
        assert m.operation.verify()
        gens = [o for o in walk(m.operation) if isinstance(o.opview, linalg.GenericOp)]
        return len(gens), [ir.StringAttr(o.attributes[fit.COPY_KEPT]).value for o in gens
                           if fit.COPY_KEPT in o.attributes]


def a_copy_whose_input_is_written_again_before_a_reader_is_kept():
    """Breaks if drop-unread-padding forwards a copy's readers to its input without asking whether the input
    is written in place before them: bufferize then copies the tile again to keep the value they read."""
    n, kept = _forwarded(False)
    assert n == 2 and not kept, (n, kept)
    n, kept = _forwarded(True)
    assert any("written again" in k for k in kept), kept


# ops_dot_cut_to_numel at drop-unread-padding's input: a 128x384 @ 384x384 dot on blocks of 512, both loads
# filling 0 past 384
_DOT = """#map = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
module attributes {"ttg.global-access-any-layout", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "pytorchsim:0", "ttg.threads-per-warp" = 128 : i32} {
  func.func @ops_dot_cut_to_numel_kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c384 = arith.constant 384 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [49152], strides: [1] : memref<*xbf16> to memref<49152xbf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [147456], strides: [1] : memref<*xbf16> to memref<147456xbf16>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [49152], strides: [1] : memref<*xbf16> to memref<49152xbf16>
    %0 = bufferization.alloc_tensor() : tensor<512x128xbf16>
    %1 = "pytorchsim.transfer"(%reinterpret_cast_1, %c0, %0, %c0, %c0, %c0, %c384) <{async = true, direction = "load", dram_arg = 0 : i64, dram_stride = [1, 384], masked_axes = [0], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<49152xbf16>, index, tensor<512x128xbf16>, index, index, index, index) -> tensor<512x128xbf16>
    %2 = bufferization.alloc_tensor() : tensor<512x512xbf16>
    %3 = "pytorchsim.transfer"(%reinterpret_cast_0, %c0, %2, %c0, %c0, %c0, %c384, %c0, %c384) <{async = true, direction = "load", dram_arg = 1 : i64, dram_stride = [1, 384], masked_axes = [1, 0], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 4, 0, 0, 0>}> : (memref<147456xbf16>, index, tensor<512x512xbf16>, index, index, index, index, index, index) -> tensor<512x512xbf16>
    %4 = tensor.empty() : tensor<512x128xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<512x128xf32>) -> tensor<512x128xf32>
    %6 = "pytorchsim.dot"(%1, %3, %5) <{indexing_maps = [#map, #map1, #map2]}> : (tensor<512x128xbf16>, tensor<512x512xbf16>, tensor<512x128xf32>) -> tensor<512x128xf32>
    %7 = tensor.empty() : tensor<512x128xbf16>
    %8 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<512x128xf32>) outs(%7 : tensor<512x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %9 = arith.truncf %in : f32 to bf16
      linalg.yield %9 : bf16
    } -> tensor<512x128xbf16>
    "pytorchsim.transfer"(%reinterpret_cast, %c0, %8, %c0, %c0, %c0, %c384) <{async = true, direction = "store", dram_arg = 2 : i64, dram_stride = [1, 384], masked_axes = [0], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<49152xbf16>, index, tensor<512x128xbf16>, index, index, index, index) -> ()
    return
  }
}"""


def _dot_fit(fill):
    """The dot's operand shapes after the pass, both loads' masked_fill set to `fill` (bf16 bits)."""
    ctx, m = _parsed(_DOT)
    with ctx, ir.Location.unknown():
        for o in walk(m.operation):
            if isinstance(o.opview, pytorchsim.TransferOp) and "masked_fill" in o.attributes and \
                    ir.StringAttr(o.attributes["direction"]).value == "load":
                o.attributes["masked_fill"] = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), fill)
        fit.run(m, grid=())
        assert m.operation.verify()
        (dot,) = [o for o in walk(m.operation) if isinstance(o.opview, pytorchsim.DotOp)]
        return [list(ir.RankedTensorType(v.type).shape) for v in dot.operands]


def a_dot_is_cut_to_what_its_store_keeps_and_its_product_reaches():
    """Breaks if drop-unread-padding takes a dot as a reader of all of its tiles: N stays at the 512 of the
    block where the store keeps 384, and K at 512 where both loads fill 0 past 384."""
    assert _dot_fit(0) == [[384, 128], [384, 384], [384, 128]], _dot_fit(0)


def a_dot_keeps_k_where_no_padding_is_0():
    """Breaks if the K cut ignores the fill: with both loads filling 1.0 past 384 the product there is not 0.
    """
    one = 0x3F80
    assert _dot_fit(one) == [[512, 128], [384, 512], [384, 128]], _dot_fit(one)


# ops_broadcast_copy_cut_to_numel at drop-unread-padding's input: x - max(x, axis 0) on 128x16 holding 8
# columns, the max broadcast back over the rows by a pytorchsim.convert_layout
_BROADCAST = """#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {"ttg.global-access-any-layout", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "pytorchsim:0", "ttg.threads-per-warp" = 128 : i32} {
  func.func @ops_broadcast_copy_cut_to_numel_kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1024], strides: [1] : memref<*xbf16> to memref<1024xbf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1] : memref<*xbf16> to memref<1024xbf16>
    %0 = bufferization.alloc_tensor() : tensor<128x16xbf16>
    %1 = "pytorchsim.transfer"(%reinterpret_cast_0, %c0, %0, %c0, %c0, %c0, %c8) <{async = true, direction = "load", dram_arg = 0 : i64, dram_stride = [8, 1], masked_axes = [1], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<1024xbf16>, index, tensor<128x16xbf16>, index, index, index, index) -> tensor<128x16xbf16>
    %2 = tensor.empty() : tensor<128x16xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<128x16xbf16>) outs(%2 : tensor<128x16xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %10 = arith.extf %in : bf16 to f32
      linalg.yield %10 : f32
    } -> tensor<128x16xf32>
    %4 = tensor.empty() : tensor<1x16xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %expanded = tensor.expand_shape %3 [[0, 1], [2]] output_shape [128, 1, 16] : tensor<128x16xf32> into tensor<128x1x16xf32>
    %reduced = linalg.reduce ins(%expanded : tensor<128x1x16xf32>) outs(%5 : tensor<1x16xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %10 = arith.maxnumf %in, %init : f32
        linalg.yield %10 : f32
      }
    %6 = tensor.empty() : tensor<128x1x16xf32>
    %7 = "pytorchsim.convert_layout"(%reduced, %6) <{added_dims = array<i64: 0>, dst_perm = array<i64: 0, 1, 2>, src_perm = array<i64: 0, 1>}> : (tensor<1x16xf32>, tensor<128x1x16xf32>) -> tensor<128x1x16xf32>
    %collapsed = tensor.collapse_shape %7 [[0, 1], [2]] : tensor<128x1x16xf32> into tensor<128x16xf32>
    %8 = tensor.empty() : tensor<128x16xbf16>
    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %collapsed : tensor<128x16xbf16>, tensor<128x16xf32>) outs(%8 : tensor<128x16xbf16>) {
    ^bb0(%in: bf16, %in_1: f32, %out: bf16):
      %10 = arith.extf %in : bf16 to f32
      %11 = arith.subf %10, %in_1 : f32
      %12 = arith.truncf %11 : f32 to bf16
      linalg.yield %12 : bf16
    } -> tensor<128x16xbf16>
    "pytorchsim.transfer"(%reinterpret_cast, %c0, %9, %c0, %c0, %c0, %c8) <{async = true, direction = "store", dram_arg = 1 : i64, dram_stride = [8, 1], masked_axes = [1], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<1024xbf16>, index, tensor<128x16xbf16>, index, index, index, index) -> ()
    return
  }
}"""


def a_broadcast_copy_is_cut_through():
    """Breaks if drop-unread-padding takes a broadcast copy as a reader of all of its source: the 1x16 max and
    every tile cut with it keep 16 columns where the store keeps 8."""
    ctx, m = _parsed(_BROADCAST)
    with ctx, ir.Location.unknown():
        fit.run(m, grid=())
        assert m.operation.verify()
        shapes = _fresh_shapes(m)
    assert all(s[-1] == 8 for s in shapes), shapes


# the conv template's dot at drop-unread-padding's input, cut down: A is a collapse of a [K, 4, 16] tile (its
# M merged), K on the lanes; B's load fills 0 past K = 8 of its block of 16
_DOT_MERGED = """#map = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d0)>
module attributes {"ttg.global-access-any-layout", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "pytorchsim:0", "ttg.threads-per-warp" = 128 : i32} {
  func.func @dot_merged_operand_kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [384], strides: [1] : memref<*xf32> to memref<384xf32>
    %0 = tensor.empty() : tensor<16x4x16xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16x4x16xf32>) -> tensor<16x4x16xf32>
    %collapsed = tensor.collapse_shape %1 [[0], [1, 2]] : tensor<16x4x16xf32> into tensor<16x64xf32>
    %2 = bufferization.alloc_tensor() : tensor<16x16xf32>
    %3 = "pytorchsim.transfer"(%reinterpret_cast_2, %c0, %2, %c0, %c0, %c0, %c8) <{direction = "load", dram_arg = 0 : i64, dram_stride = [24, 1], masked_axes = [1], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<384xf32>, index, tensor<16x16xf32>, index, index, index, index) -> tensor<16x16xf32>
    %4 = tensor.empty() : tensor<16x64xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %6 = "pytorchsim.dot"(%collapsed, %3, %5) <{indexing_maps = [#map, #map1, #map2]}> : (tensor<16x64xf32>, tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
    "pytorchsim.transfer"(%reinterpret_cast, %c0, %6, %c0, %c0, %c0, %c16) <{direction = "store", dram_arg = 1 : i64, dram_stride = [64, 1], masked_axes = [0], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<1024xf32>, index, tensor<16x64xf32>, index, index, index, index) -> ()
    return
  }
}"""


def a_dot_operand_through_a_merging_collapse_keeps_k_with_the_other():
    """Breaks if drop-unread-padding loses a dot operand viewed through a collapse that merges axes: B alone
    is cut to
    the 8 of K its load fills, A keeps 16, and the matmul reads past B's slot (a conv1d template)."""
    ctx, m = _parsed(_DOT_MERGED)
    with ctx, ir.Location.unknown():
        fit.run(m, grid=())
        assert m.operation.verify()
        (dot,) = [o for o in walk(m.operation) if isinstance(o.opview, pytorchsim.DotOp)]
        a, b, _ = [list(ir.RankedTensorType(v.type).shape) for v in dot.operands]
    assert a[0] == b[1], (a, b)


# a layer-norm epilogue's stats gathered into a [lanes, 8, 2, 2] tile the generic reads through a
# collapse to the block of 4096; the store keeps the first N
_GATHER_MERGED = """#map = affine_map<(d0) -> (d0)>
module attributes {"ttg.global-access-any-layout", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "pytorchsim:0", "ttg.threads-per-warp" = 128 : i32} {
  func.func @gather_merged_kernel(%arg0: memref<*xf32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c0 = arith.constant 0 : index
    %cN = arith.constant N : index
    %cst = arith.constant 1.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [N], strides: [1] : memref<*xf32> to memref<Nxf32>
    %0 = bufferization.alloc_tensor() : tensor<4096xf32>
    %1 = "pytorchsim.transfer"(%reinterpret_cast, %c0, %0, %c0, %c0, %c0, %cN) <{direction = "load", dram_arg = 0 : i64, dram_stride = [1], masked_axes = [0], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<Nxf32>, index, tensor<4096xf32>, index, index, index, index) -> tensor<4096xf32>
    %2 = tensor.empty() : tensor<128x8x2x2xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x8x2x2xf32>) -> tensor<128x8x2x2xf32>
    %collapsed = tensor.collapse_shape %3 [[0, 1, 2, 3]] : tensor<128x8x2x2xf32> into tensor<4096xf32>
    %4 = tensor.empty() : tensor<4096xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%1, %collapsed : tensor<4096xf32>, tensor<4096xf32>) outs(%4 : tensor<4096xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.addf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<4096xf32>
    "pytorchsim.transfer"(%reinterpret_cast, %c0, %5, %c0, %c0, %c0, %cN) <{direction = "store", dram_arg = 0 : i64, dram_stride = [1], masked_axes = [0], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<Nxf32>, index, tensor<4096xf32>, index, index, index, index) -> ()
    return
  }
}"""


def _gather_merged_fit(n):
    """(the tiles' shapes after the pass, what it logged) with the store keeping the first `n`."""
    import logging
    logged = []
    handler = logging.Handler()
    handler.emit = lambda r: logged.append(r.getMessage())
    fit.LOG.addHandler(handler)
    try:
        ctx, m = _parsed(_GATHER_MERGED.replace("N", str(n)))
        with ctx, ir.Location.unknown():
            fit.run(m, grid=())
            assert m.operation.verify()
            shapes = _fresh_shapes(m)
    finally:
        fit.LOG.removeHandler(handler)
    return shapes, logged


def a_merged_view_is_cut_on_its_outermost_axis_below_the_lanes():
    """Breaks if drop-unread-padding cuts the 1-D operands of a generic and not the merged view beside it
    (verify refuses the generic, ConvNeXtV2's layer norm), or cuts the merged tile's lane axis (its rows land
    in other lanes)."""
    shapes, logged = _gather_merged_fit(3072)
    assert shapes == [[3072], [128, 6, 2, 2], [3072]], shapes
    assert not logged, logged


def a_merged_view_with_no_box_keeps_its_class_and_says_so():
    """Breaks if a class whose numel is no whole number of the merged view's rows is cut, or kept silently."""
    shapes, logged = _gather_merged_fit(3600)
    assert shapes == [[4096], [128, 8, 2, 2], [4096]], shapes
    assert any("no box" in m for m in logged), logged


# the mm of mllama's vision attention (M 96 on a 128-row block) at drop-unread-padding's input: its A, the head
# transpose fused into the load, is a pointer-tile gather whose index tile a generic makes
_GATHER_A = """#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0)>
module attributes {"ttg.global-access-any-layout", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "pytorchsim:0", "ttg.threads-per-warp" = 128 : i32} {
  func.func @pytorchsim_triton_compiler_fused_clone_mm_t_transpose_view_id4mnrro(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c96_i32 = arith.constant 96 : i32
    %c64_i32 = arith.constant 64 : i32
    %c96 = arith.constant 96 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %c512_i32 = arith.constant 512 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c6144_i32 = arith.constant 6144 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [49152], strides: [1] : memref<*xf32> to memref<49152xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [49152], strides: [1] : memref<*xf32> to memref<49152xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [262144], strides: [1] : memref<*xf32> to memref<262144xf32>
    %0 = tensor.empty() : tensor<512x128xi32>
    %1 = "pytorchsim.program_id"() <{axis = 0 : i32}> : () -> i32
    %2 = arith.divsi %1, %c8_i32 : i32
    %3 = arith.muli %2, %c8_i32 : i32
    %4 = arith.subi %c1_i32, %3 : i32
    %5 = arith.minsi %4, %c8_i32 : i32
    %6 = arith.remsi %1, %5 : i32
    %7 = arith.addi %3, %6 : i32
    %8 = arith.remsi %1, %c8_i32 : i32
    %9 = arith.divsi %8, %5 : i32
    %10 = arith.muli %7, %c128_i32 : i32
    %11 = arith.index_cast %10 : i32 to index
    %12 = arith.muli %9, %c512_i32 : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<512x128xi32>) {
    ^bb0(%out: i32):
      %37 = linalg.index 0 : index
      %38 = arith.index_cast %37 : index to i32
      %39 = arith.addi %10, %38 : i32
      %40 = arith.cmpi slt, %39, %c96_i32 : i32
      %41 = linalg.index 1 : index
      %42 = arith.muli %39, %c512_i32 : i32
      %43 = arith.index_cast %41 : index to i32
      %44 = arith.addi %43, %42 : i32
      %45 = arith.divsi %44, %c64_i32 : i32
      %46 = arith.remsi %45, %c8_i32 : i32
      %47 = arith.muli %46, %c6144_i32 : i32
      %48 = arith.muli %39, %c64_i32 : i32
      %49 = arith.remsi %44, %c64_i32 : i32
      %50 = arith.addi %49, %48 : i32
      %51 = arith.addi %50, %47 : i32
      %52 = arith.select %40, %51, %c0_i32 : i32
      linalg.yield %52 : i32
    } -> tensor<512x128xi32>
    %15 = bufferization.alloc_tensor() copy(%14) : tensor<512x128xi32>
    %16 = bufferization.alloc_tensor() : tensor<512x128xf32>
    %17 = "pytorchsim.transfer"(%reinterpret_cast_0, %c0, %16, %c0, %c0, %15) <{async = true, direction = "load", dram_arg = 1 : i64, dram_stride = [0, 0], offset_stride = 1 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> : (memref<49152xf32>, index, tensor<512x128xf32>, index, index, tensor<512x128xi32>) -> tensor<512x128xf32>
    %18 = tensor.empty() : tensor<512x128xf32>
    %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<512x128xf32>) -> tensor<512x128xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%17 : tensor<512x128xf32>) outs(%18 : tensor<512x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %37 = linalg.index 0 : index
      %38 = arith.index_cast %37 : index to i32
      %39 = arith.addi %10, %38 : i32
      %40 = arith.cmpi slt, %39, %c96_i32 : i32
      %41 = arith.select %40, %in, %cst : f32
      linalg.yield %41 : f32
    } -> tensor<512x128xf32>
    %21 = arith.muli %13, %c512 : index
    %22 = arith.addi %13, %c512 : index
    %23 = arith.minsi %22, %c512 : index
    %24 = arith.maxsi %23, %13 : index
    %25 = arith.subi %24, %13 : index
    %26 = bufferization.alloc_tensor() : tensor<512x512xf32>
    %27 = "pytorchsim.transfer"(%reinterpret_cast_1, %21, %26, %c0, %c0, %c0, %25) <{async = true, direction = "load", dram_arg = 0 : i64, dram_stride = [512, 1], masked_axes = [0], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}> : (memref<262144xf32>, index, tensor<512x512xf32>, index, index, index, index) -> tensor<512x512xf32>
    %28 = "pytorchsim.dot"(%20, %27, %19) <{indexing_maps = [#map1, #map2, #map3]}> : (tensor<512x128xf32>, tensor<512x512xf32>, tensor<512x128xf32>) -> tensor<512x128xf32>
    %29 = arith.muli %11, %c512 : index
    %30 = arith.addi %13, %29 : index
    %31 = arith.addi %11, %c128 : index
    %32 = arith.minsi %31, %c96 : index
    %33 = arith.maxsi %32, %11 : index
    %34 = arith.subi %33, %11 : index
    %35 = arith.minsi %34, %c128 : index
    %36 = arith.minsi %25, %c512 : index
    "pytorchsim.transfer"(%reinterpret_cast, %30, %28, %c0, %c0, %c0, %35, %c0, %36) <{async = true, direction = "store", dram_arg = 2 : i64, dram_stride = [1, 512], masked_axes = [1, 0], masked_fill = 0 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 4, 0, 0, 0>}> : (memref<49152xf32>, index, tensor<512x128xf32>, index, index, index, index, index, index) -> ()
    return
  }
}
"""


def a_gather_keeps_its_index_tile_with_its_tile():
    """Breaks if drop-unread-padding cuts a pointer-tile gather's tile apart from its index tile: the gather
    reads index element i for tile element i, so a 512x96 tile on a 512x128 index reads every row but the first
    through another's index (mllama's mm_3, wrong values)."""
    ctx, m = _parsed(_GATHER_A)
    with ctx, ir.Location.unknown():
        fit.run(m, grid=(1, 1, 1))
        assert m.operation.verify()
        slot = pytorchsim.operand_number
        gathers = [o for o in walk(m.operation) if isinstance(o.opview, pytorchsim.TransferOp)
                   and slot(o, "clamp") > slot(o, "index")]
        shapes = [(list(ir.ShapedType(o.operands[slot(o, "sram")].type).shape),
                   list(ir.ShapedType(o.operands[slot(o, "index")].type).shape)) for o in gathers]
    assert shapes and all(a == b for a, b in shapes), shapes


CASES = [bufferize_allocates_the_cut_tiles, a_copy_whose_input_is_written_again_before_a_reader_is_kept,
         a_dot_is_cut_to_what_its_store_keeps_and_its_product_reaches, a_dot_keeps_k_where_no_padding_is_0,
         a_broadcast_copy_is_cut_through, a_dot_operand_through_a_merging_collapse_keeps_k_with_the_other,
         a_merged_view_is_cut_on_its_outermost_axis_below_the_lanes,
         a_merged_view_with_no_box_keeps_its_class_and_says_so, a_gather_keeps_its_index_tile_with_its_tile]


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
