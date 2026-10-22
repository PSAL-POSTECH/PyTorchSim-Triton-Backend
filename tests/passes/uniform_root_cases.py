"""Placement is the root's: an MVIN into a one-lane view of a buffer spread over the lanes writes that
lane's bank alone, and only a view of a UNIFORM buffer is replicated into every lane.
    python3 tests/passes/uniform_root_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects import torchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import lower_dma  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

torchsim.register()
LANES = 128
DRAM = "memref<65536xf32>"
VIEW = "memref<1x16xf32, strided<[16, 1]>, 1>"


def _module(root_t):
    """An MVIN of one [16] row into a [1, 16] bank_view of a buffer of type `root_t`."""
    return (f"module {{\n  memref.global \"private\" @spad : {root_t}\n"
            f"  func.func @f(%d: {DRAM}) {{\n"
            f"    %c0 = arith.constant 0 : index\n"
            f"    %s = memref.get_global @spad : {root_t}\n"
            f"    %v = \"torchsim.bank_view\"(%s) {{offset = 0 : i64, sizes = array<i64: 1, 16>,\n"
            f"        strides = array<i64: 16, 1>}} : ({root_t}) -> {VIEW}\n"
            f"    \"torchsim.transfer\"(%d, %c0, %v, %c0, %c0) <{{dma_kind = \"MVIN\", dram_arg = 0 : i64, "
            f"dram_stride = [0, 1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}> "
            f": ({DRAM}, index, {VIEW}, index, index) -> ()\n    return\n  }}\n}}\n")


def _descriptor(text):
    """(shape4, vsa) of the one DMA descriptor lower-dma packs for `text`."""
    m = ir.Module.parse(text)
    assert m.operation.verify()
    lower_dma.run(m, vectorlane=LANES)
    (g,) = [o for o in walk(m.operation) if o.name == "memref.global"
            and ir.StringAttr(o.attributes["sym_name"]).value.startswith("dma_desc_")]
    dense = ir.DenseIntElementsAttr(g.attributes["initial_value"])
    words = [dense[i] & 0xFFFFFFFFFFFFFFFF for i in range(len(dense))]
    shape4 = [words[0] & 0xFFFFFFFF, words[0] >> 32, words[1] & 0xFFFFFFFF, words[1] >> 32]
    return shape4, (words[lower_dma.S_FLAGS] >> 32) & 0xFF


def a_one_lane_view_of_a_spread_buffer_is_not_copied_into_every_lane():
    """Breaks if placement is read off the view's own [1, 16] type: the MVIN then widens a free dim
    to 128 lanes at DRAM stride 0 and writes the row over every other lane's rows of the buffer."""
    shape4, vsa = _descriptor(_module("memref<128x16xf32, 1>"))
    assert shape4 == [1, 1, 1, 16] and vsa == 2, (shape4, vsa)


def a_view_of_a_uniform_buffer_is_still_copied_into_every_lane():
    shape4, vsa = _descriptor(_module("memref<1x16xf32, 1>"))
    assert shape4[vsa] == LANES and vsa != 2, (shape4, vsa)


CASES = [a_one_lane_view_of_a_spread_buffer_is_not_copied_into_every_lane,
         a_view_of_a_uniform_buffer_is_still_copied_into_every_lane]

if __name__ == "__main__":
    bad = 0
    for c in CASES:
        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        with ctx, ir.Location.unknown():
            try:
                c()
                print(f"ok   {c.__name__}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAIL {c.__name__}: {type(e).__name__}: {str(e)[:300]}")
    sys.exit(1 if bad else 0)
