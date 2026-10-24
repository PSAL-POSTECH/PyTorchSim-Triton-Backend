"""lower_dma: an MVIN into a one-bank tile spreads its copies over every lane
through a free extent-1 descriptor dim (DRAM stride 0, the largest spad stride), or is refused by name.
    python3 tests/passes/mvin_replicate_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects import torchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family import lower_dma  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

torchsim.register()
LANES = 128
DRAM = "memref<65536xf32>"


def _module(kind, sram_t, dram_stride, masked=None, gather=None):
    n_clamp = 2 * len(masked or ())
    operands = "%d, %c0, %s, %c0, %c0" + (", %ix" if gather is not None else "") + ", %lo, %hi" * len(masked or ())
    types = f"{DRAM}, index, {sram_t}, index, index" + (", memref<16xi32, 1>" if gather is not None else "") \
        + ", index, index" * len(masked or ())
    attrs = f'dma_kind = "{kind}", dram_arg = 0 : i64, dram_stride = {list(dram_stride)}'
    if masked:
        attrs += f", masked_axes = {list(masked)}, masked_fill = 0 : i64"
    if gather is not None:
        attrs += f", indirect_dim = {gather} : i64, offset_stride = 1 : i64"
    seg = f"1, 1, 1, 1, 1, {int(gather is not None)}, {n_clamp}, 0, 0, 0"
    return (f"module {{\n  memref.global \"private\" @spad : {sram_t}\n"
            f"  memref.global \"private\" @ix : memref<16xi32, 1>\n"
            f"  func.func @f(%d: {DRAM}, %lo: index, %hi: index) {{\n"
            f"    %c0 = arith.constant 0 : index\n"
            f"    %s = memref.get_global @spad : {sram_t}\n"
            f"    %ix = memref.get_global @ix : memref<16xi32, 1>\n"
            f"    \"torchsim.transfer\"({operands}) <{{{attrs}, operandSegmentSizes = array<i32: {seg}>}}> "
            f": ({types}) -> ()\n    return\n  }}\n}}\n")


def _lowered(text):
    m = ir.Module.parse(text)
    assert m.operation.verify()
    lower_dma.run(m, vectorlane=LANES)
    return m


def _descriptor(m):
    """(shape4, dram4, spad4, vsa) of the one DMA descriptor global `m` holds, decoded from its
    initial value (the i32 view, when a runtime field made one, read back as 64-bit slots)."""
    (g,) = [o for o in walk(m.operation) if o.name == "memref.global"
            and ir.StringAttr(o.attributes["sym_name"]).value.startswith("dma_desc_")]
    dense = ir.DenseIntElementsAttr(g.attributes["initial_value"])
    words = [dense[i] for i in range(len(dense))]
    if len(words) == 2 * lower_dma.DESC_SLOTS:
        words = [(lo & 0xFFFFFFFF) | ((hi & 0xFFFFFFFF) << 32) for lo, hi in zip(words[::2], words[1::2])]
    words = [w & 0xFFFFFFFFFFFFFFFF for w in words]
    shape4 = [words[0] & 0xFFFFFFFF, words[0] >> 32, words[1] & 0xFFFFFFFF, words[1] >> 32]
    return shape4, words[6:10], words[10:14], (words[lower_dma.S_FLAGS] >> 32) & 0xFF


def _refusal(text):
    try:
        _lowered(text)
    except NotImplementedError as e:
        return str(e)
    raise AssertionError("not refused")


def a_one_bank_row_is_copied_into_every_lane():
    shape4, dram4, spad4, vsa = _descriptor(_lowered(_module("MVIN", "memref<1x16xf32, 1>", [0, 1])))
    assert shape4[vsa] == LANES and dram4[vsa] == 0 and spad4[vsa] == max(spad4), (shape4, dram4, spad4, vsa)
    assert [e for d, e in enumerate(shape4) if d != vsa] == [1, 1, 16], shape4


def a_scalar_is_copied_into_every_lane():
    shape4, dram4, spad4, vsa = _descriptor(_lowered(_module("MVIN", "memref<f32, 1>", [1])))
    assert shape4[vsa] == LANES and dram4[vsa] == 0 and spad4[vsa] == max(spad4), (shape4, dram4, spad4, vsa)


def a_clamped_extent_1_dim_is_not_the_one_the_copies_ride():
    shape4, dram4, spad4, vsa = _descriptor(_lowered(
        _module("MVIN", "memref<1x1x4x16xf32, 1>", [0, 0, 16, 1], masked=[0])))
    assert vsa == 1 and shape4[vsa] == LANES and dram4[vsa] == 0, (shape4, dram4, vsa)
    assert shape4[0] == 1, shape4


def a_one_bank_tile_with_no_free_extent_1_dim_is_refused_by_name():
    got = _refusal(_module("MVIN", "memref<1x2x4x16xf32, 1>", [0, 64, 16, 1], masked=[0]))
    assert "no extent-1 dim" in got and "named: [0]" in got, got


def a_gather_whose_gathered_dim_is_the_only_free_one_is_refused():
    got = _refusal(_module("MVIN", "memref<1x2x4x16xf32, 1>", [0, 64, 16, 1], gather=0))
    assert "no extent-1 dim" in got and "named: [0]" in got, got


def an_mvout_from_one_bank_still_reads_lane_0_alone():
    shape4, dram4, spad4, vsa = _descriptor(_lowered(_module("MVOUT", "memref<1x16xf32, 1>", [0, 1])))
    assert shape4[vsa] == 1 and shape4 == [1, 1, 1, 16], (shape4, vsa)


def a_spread_tile_rides_its_axis_0_untouched():
    shape4, dram4, spad4, vsa = _descriptor(_lowered(_module("MVIN", "memref<128x16xf32, 1>", [16, 1])))
    assert vsa == 2 and shape4 == [1, 1, 128, 16] and dram4[2:] == [16, 1], (shape4, dram4, vsa)


def the_footprint_of_a_copied_row_is_one_lanes_row():
    m = ir.Module.parse(_module("MVIN", "memref<1x16xf32, 1>", [0, 1]))
    (tx,) = [o for o in walk(m.operation) if o.name == "torchsim.transfer"]
    (got,) = lower_dma.sram_footprints(tx, LANES, lambda v: (0, 0))
    assert got[1] == (0, 16), got


def the_walk_a_tog_states_is_the_descriptors_lane_copies_included():
    """walk() states the descriptor's dims and DRAM strides, its leading unit pads dropped."""
    for kind, t, stride, masked, want in (("MVIN", "memref<1x16xf32, 1>", [0, 1], None, [LANES, 1, 1, 16]),
                                          ("MVOUT", "memref<1x16xf32, 1>", [0, 1], None, [1, 16]),
                                          ("MVIN", "memref<128x16xf32, 1>", [16, 1], None, [128, 16]),
                                          ("MVIN", "memref<1x1x4x16xf32, 1>", [0, 0, 16, 1], [0],
                                           [1, LANES, 4, 16])):
        text = _module(kind, t, stride, masked=masked)
        m = ir.Module.parse(text)
        (tx,) = [o for o in walk(m.operation) if o.name == "torchsim.transfer"]
        dims, strides, lead = lower_dma.walk(tx, LANES)
        shape4, dram4, _spad4, _vsa = _descriptor(_lowered(text))
        cut = len(shape4) - len(dims)
        assert dims == want == shape4[cut:] and strides == dram4[cut:], (kind, t, dims, strides, shape4, dram4)
        assert lead == len(dims) - len(stride), (kind, t, lead)


CASES = [a_one_bank_row_is_copied_into_every_lane, a_scalar_is_copied_into_every_lane,
         a_clamped_extent_1_dim_is_not_the_one_the_copies_ride,
         a_one_bank_tile_with_no_free_extent_1_dim_is_refused_by_name,
         a_gather_whose_gathered_dim_is_the_only_free_one_is_refused,
         an_mvout_from_one_bank_still_reads_lane_0_alone, a_spread_tile_rides_its_axis_0_untouched,
         the_footprint_of_a_copied_row_is_one_lanes_row,
         the_walk_a_tog_states_is_the_descriptors_lane_copies_included]

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
