"""A masked load's TOG dma moves what its clamp leaves live, synchronous or not. Transfers became
synchronous unless subtiled (9dea7ec) and a sync load's `sem` is a plain index, so the barrier check
refused every one: a 1x3072x768 GEMV fetched 4096 of B's rows in R0_BLOCK 4096 for K = 3072.
    python3 tests/passes/live_extent_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.lowering.tog.passes import build_skeleton as bs  # noqa: E402
from pytorchsim_triton_compiler.lowering.tog.passes import togsim_ops as ts  # noqa: E402

_MOD = """
module {{
  memref.global @buf0 : memref<8x64xbf16, 1>
  memref.global @buf1 : memref<8x1xbf16, 1>
  func.func @k(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32,
               %arg5: i32, %arg6: i32, %arg7: i32) {{
    %c0 = arith.constant 0 : index
    %c48 = arith.constant 48 : index
    %s = memref.get_global @buf0 : memref<8x64xbf16, 1>
    %o = memref.get_global @buf1 : memref<8x1xbf16, 1>
    %x = memref.reinterpret_cast %arg0 to offset: [0], sizes: [384], strides: [1]
        : memref<*xbf16> to memref<384xbf16>
    %y = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8], strides: [1]
        : memref<*xbf16> to memref<8xbf16>
    "torchsim.transfer"(%x, %c0, %s, %c0, %c0, %c0, %c48) <{{dma_kind = "MVIN", dram_arg = 0 : i64,
        dram_stride = [1, 8], masked_axes = [1], masked_fill = 0 : i64{asyn},
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2, 0, 0, 0>}}>
        : (memref<384xbf16>, index, memref<8x64xbf16, 1>, index, index, index, index) -> ()
    {wait}
    llvm.inline_asm has_side_effects asm_dialect = att {{compute_type = "VectorCompute"}}
        ".insn r CUSTOM_3, 0, 0x40, x0, x0, x0", "~{{a0}},~{{memory}}"  : () -> ()
    affine.for %i = 0 to 8 {{
      %v = affine.load %s[%i, 0] : memref<8x64xbf16, 1>
      affine.store %v, %o[%i, 0] : memref<8x1xbf16, 1>
    }}
    llvm.inline_asm has_side_effects asm_dialect = att
        ".insn r CUSTOM_3, 0, 0x41, x0, x0, x0", "~{{a0}},~{{memory}}"  : () -> ()
    "torchsim.transfer"(%y, %c0, %o, %c0, %c0) <{{dma_kind = "MVOUT", dram_arg = 1 : i64,
        dram_stride = [1, 1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}>
        : (memref<8xbf16>, index, memref<8x1xbf16, 1>, index, index) -> ()
    return
  }}
}}
"""


def _dmas(asyn):
    from mlir.ir import IntegerAttr, Module
    from pytorchsim_triton_compiler.lowering.passes import lib_ir
    module = Module.parse(_MOD.format(
        asyn=", async = true" if asyn else "",
        wait=('%n = arith.constant 512 : index\n    "torchsim.wait"(%c0, %n) : (index, index) -> ()'
              if asyn else "")))
    bs.build_skeleton(module)
    out = []
    for o in lib_ir.walk(module.operation):
        if o.name == ts.DMA and IntegerAttr(o.attributes[ts.ATTR_DIR]).value == ts.DIR_LOAD:
            out.append(o)
    assert len(out) == 1, out
    return out[0]


def a_synchronous_masked_load_is_sized_by_its_clamp():
    from pytorchsim_triton_compiler.lowering.passes.lib_ir import int_array
    d = _dmas(False)
    assert ts.ATTR_DYN_DIMS in d.attributes, d
    assert list(int_array(d.attributes[ts.ATTR_DYN_DIMS])) == [1], d


def an_async_masked_load_whose_wait_expects_its_tile_is_sized_too():
    from pytorchsim_triton_compiler.lowering.passes.lib_ir import int_array
    d = _dmas(True)
    assert list(int_array(d.attributes[ts.ATTR_DYN_DIMS])) == [1], d


CASES = [a_synchronous_masked_load_is_sized_by_its_clamp,
         an_async_masked_load_whose_wait_expects_its_tile_is_sized_too]


def main():
    from mlir.ir import Context, Location
    failed = 0
    for case in CASES:
        try:
            with Context() as ctx, Location.unknown():
                ctx.allow_unregistered_dialects = True
                case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
