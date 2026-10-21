"""tritonpytorchsim-place-spad-buffers, TO_CUSTOM's last: every scratchpad alloc a per-lane byte offset
(slots end to end in entry order, each reserve_per_lane x element bytes from a register boundary, shared
only as torchsim-share-spad-buffers says, none when only freed); materialization views one arena there.
    python3 tests/passes/spad_placement_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects import torchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lower  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

torchsim.register()
T = machine.target()
ALIGN = T.vlen_bits // 8

_THREE = """module {
  func.func @f(%x: f32, %h: bf16) {
    %c0 = arith.constant 0 : index
    %a = memref.alloc() : memref<4xf32, 1>
    %b = memref.alloc() : memref<1x40xf32, 1>
    %c = memref.alloc() : memref<8x3xbf16, 1>
    %e = memref.alloc() : memref<4xf32, 1>
    memref.store %x, %a[%c0] : memref<4xf32, 1>
    memref.store %x, %b[%c0, %c0] : memref<1x40xf32, 1>
    memref.store %h, %c[%c0, %c0] : memref<8x3xbf16, 1>
    %ra = memref.load %a[%c0] : memref<4xf32, 1>
    %rb = memref.load %b[%c0, %c0] : memref<1x40xf32, 1>
    %rc = memref.load %c[%c0, %c0] : memref<8x3xbf16, 1>
    memref.dealloc %e : memref<4xf32, 1>
    memref.dealloc %c : memref<8x3xbf16, 1>
    memref.dealloc %b : memref<1x40xf32, 1>
    memref.dealloc %a : memref<4xf32, 1>
    return
  }
}"""

_ONE_AFTER_THE_OTHER = """module {
  func.func @f(%x: f32) {
    %c0 = arith.constant 0 : index
    %a = memref.alloc() : memref<8x16xf32, 1>
    %b = memref.alloc() : memref<8x16xf32, 1>
    memref.store %x, %a[%c0, %c0] : memref<8x16xf32, 1>
    %ra = memref.load %a[%c0, %c0] : memref<8x16xf32, 1>
    memref.store %ra, %b[%c0, %c0] : memref<8x16xf32, 1>
    %rb = memref.load %b[%c0, %c0] : memref<8x16xf32, 1>
    memref.dealloc %b : memref<8x16xf32, 1>
    memref.dealloc %a : memref<8x16xf32, 1>
    return
  }
}"""


def _ctx():
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    return ctx


def _placed(m):
    for p in lower.SPAD_PLACEMENT:
        p.run(m, target=T)
    return m


def _offsets(m):
    from pytorchsim_triton_compiler.lowering.passes import p11b_place_spad_buffers as place
    return [ir.IntegerAttr(o.attributes[place.OFFSET]).value if place.OFFSET in o.attributes else None
            for o in walk(m.operation) if o.name == "memref.alloc"]


def _peak(m):
    from pytorchsim_triton_compiler.lowering.passes import p11b_place_spad_buffers as place
    fn = [o for o in walk(m.operation) if o.name == "func.func"][0]
    return ir.IntegerAttr(fn.attributes[place.PEAK]).value


def slots_lie_end_to_end_in_entry_order_each_from_a_register_boundary():
    """4xf32 over 4 lanes is 1 a lane, 2 at MIN_VEC: 8 bytes. 1x40xf32 sits in one bank: 160 bytes,
    from the next register boundary. 8x3xbf16 is 3 a lane, 4 at MIN_VEC: 8 bytes. Breaks if a slot
    is sized by need, not reserve, or starts off a boundary, or if the order is not the entry's."""
    with _ctx(), ir.Location.unknown():
        m = _placed(ir.Module.parse(_THREE))
        b = ALIGN
        c = -(-(b + 160) // ALIGN) * ALIGN
        assert _offsets(m) == [0, b, c, None], _offsets(m)
        assert _peak(m) == c + 8, _peak(m)


def a_buffer_only_freed_gets_no_offset():
    """%e is allocated and freed and nothing else: no slot, no bytes in the peak."""
    with _ctx(), ir.Location.unknown():
        assert _offsets(_placed(ir.Module.parse(_THREE)))[3] is None


def buffers_that_share_a_slot_share_its_offset():
    """b's life starts after a's last read: the share pass puts it on a's slot, and both take one offset;
    the peak is one slot's, 128 f32 over 8 lanes."""
    with _ctx(), ir.Location.unknown():
        m = _placed(ir.Module.parse(_ONE_AFTER_THE_OTHER))
        assert _offsets(m) == [0, 0], _offsets(m)
        assert _peak(m) == 16 * 4, _peak(m)


def a_slot_naming_no_keeper_is_refused():
    from pytorchsim_triton_compiler.lowering.passes import p11a_share_spad_buffers as share
    from pytorchsim_triton_compiler.lowering.passes import p11b_place_spad_buffers as place
    with _ctx(), ir.Location.unknown():
        m = ir.Module.parse(_ONE_AFTER_THE_OTHER)
        allocs = [o for o in walk(m.operation) if o.name == "memref.alloc"]
        allocs[0].attributes[share.SLOT_ATTR] = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 7)
        try:
            place.run(m, target=T)
        except place.SpadPlacementRefused as e:
            assert share.SLOT_ATTR in str(e), str(e)
        else:
            raise AssertionError("a slot that names no alloc was placed")


def materialize_puts_every_slot_in_one_arena_the_module_names():
    """_THREE's slots are views of one i8 arena as long as the peak, each at its offset, the only-freed
    buffer gone; the module's SPAD_ARENA names the arena."""
    from pytorchsim_triton_compiler.lowering.passes import p12ab_materialize_spad_allocs as mat
    from pytorchsim_triton_compiler.lowering.passes.lib_bank_layout import SPAD_ARENA
    with _ctx(), ir.Location.unknown():
        m = _placed(ir.Module.parse(_THREE))
        offsets, peak = [o for o in _offsets(m) if o is not None], _peak(m)
        mat.run(m, vectorlane=T.vector_lanes)
        assert m.operation.verify()
        ops = list(walk(m.operation))
        g = [o for o in ops if o.name == "memref.global"]
        assert len(g) == 1 and str(ir.TypeAttr(g[0].attributes["type"]).value) == f"memref<{peak}xi8, 1>", g
        assert ir.FlatSymbolRefAttr(m.operation.attributes[SPAD_ARENA]).value == \
            ir.StringAttr(g[0].attributes["sym_name"]).value
        shifts = [ir.IntegerAttr(o.operands[1].owner.attributes["value"]).value
                  for o in ops if o.name == "memref.view"]
        assert shifts == offsets, (shifts, offsets)
        assert not [o for o in ops if o.name in ("memref.alloc", "memref.dealloc")]


def _lowered(text):
    """`text` placed and through TO_LLVM: the LLVM-dialect module, parsed."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "in.mlir"), os.path.join(d, "out.mlir")
        with open(src, "w") as f:
            f.write(text)
        lower.run_module_passes(src, src, lower.SPAD_PLACEMENT, target=T)
        lower.to_llvm(src, out, target=T)
        with open(out) as f:
            return ir.Module.parse(f.read())


def _byte_offset(v, sizes):
    """(symbol, bytes) `v` points at: through llvm.getelementptr (constant indices, or llvm.mlir.constant
    / llvm.add / llvm.mul of them) to an llvm.mlir.addressof; None where not static."""
    def const(x):
        o = x.owner
        if o.name == "llvm.mlir.constant":
            return ir.IntegerAttr(o.attributes["value"]).value
        if o.name in ("llvm.add", "llvm.mul"):
            a, b = const(o.operands[0]), const(o.operands[1])
            return None if a is None or b is None else (a + b if o.name == "llvm.add" else a * b)
        return None
    total = 0
    while True:
        o = v.owner
        if o.name == "llvm.mlir.addressof":
            return ir.FlatSymbolRefAttr(o.attributes["global_name"]).value, total
        if o.name != "llvm.getelementptr":
            return None
        raw = list(ir.DenseI32ArrayAttr(o.attributes["rawConstantIndices"]))
        dyn = list(o.operands)[1:]
        idx = [const(dyn.pop(0)) if r == -2 ** 31 else r for r in raw]
        if None in idx:
            return None
        elem = ir.TypeAttr(o.attributes["elem_type"]).value
        total += idx[0] * sizes.get(str(elem), 0)
        v = o.operands[0]


def a_second_buffer_through_a_bank_view_sits_at_its_own_offset():
    """Two buffers alive at once, each written through a torchsim.bank_view: the bank view's lowering
    sets an absolute offset, which must add to the view's shifted aligned pointer, so the second
    buffer's stores land 32 bytes into the arena, not on the first's. Breaks if the placement were put
    in a memref offset a reinterpret_cast overrides."""
    text = """module {
  func.func @f(%x: vector<2xf32>, %y: vector<2xf32>) {
    %c0 = arith.constant 0 : index
    %a = memref.alloc() : memref<128x2xf32, 1>
    %b = memref.alloc() : memref<128x2xf32, 1>
    %va = "torchsim.bank_view"(%a) {offset = 0 : i64, sizes = array<i64: 1, 2>, strides = array<i64: 2, 1>} : (memref<128x2xf32, 1>) -> memref<1x2xf32, strided<[2, 1]>, 1>
    %vb = "torchsim.bank_view"(%b) {offset = 0 : i64, sizes = array<i64: 1, 2>, strides = array<i64: 2, 1>} : (memref<128x2xf32, 1>) -> memref<1x2xf32, strided<[2, 1]>, 1>
    vector.store %x, %va[%c0, %c0] : memref<1x2xf32, strided<[2, 1]>, 1>, vector<2xf32>
    vector.store %y, %vb[%c0, %c0] : memref<1x2xf32, strided<[2, 1]>, 1>, vector<2xf32>
    %ra = vector.load %va[%c0, %c0] : memref<1x2xf32, strided<[2, 1]>, 1>, vector<2xf32>
    vector.store %ra, %vb[%c0, %c0] : memref<1x2xf32, strided<[2, 1]>, 1>, vector<2xf32>
    memref.dealloc %b : memref<128x2xf32, 1>
    memref.dealloc %a : memref<128x2xf32, 1>
    return
  }
}"""
    from mlir.passmanager import PassManager
    from pytorchsim_triton_compiler.lowering.passes.lib_bank_layout import SPAD_ARENA
    with _ctx(), ir.Location.unknown():
        m = _lowered(text)
        PassManager.parse("builtin.module(canonicalize,cse)").run(m.operation)
        arena = ir.FlatSymbolRefAttr(m.operation.attributes[SPAD_ARENA]).value
        sizes = {"i8": 1, "f32": 4, f"!llvm.array<{ALIGN + 8} x i8>": ALIGN + 8}
        stores = [o for o in walk(m.operation) if o.name == "llvm.store"]
        got = [_byte_offset(o.operands[1], sizes) for o in stores]
        assert got == [(arena, 0), (arena, ALIGN), (arena, ALIGN)], got


def removing_the_arena_marker_is_refused_not_reserved_smaller():
    """p13 reserves the arena as one lane's bytes, the peak, because the module's SPAD_ARENA names it.
    Without the marker the same global is refused, never reserved at another reading of its type (as
    c6c2abe's p13 read !llvm.array<N x i8> as N lanes and kept MIN_VEC bytes)."""
    import tempfile
    from pytorchsim_triton_compiler.lowering.passes import p13_lower_to_llvm as p13
    from pytorchsim_triton_compiler.lowering.passes.lib_bank_layout import SPAD_ARENA
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "in.mlir"), os.path.join(d, "out.mlir")
        with open(src, "w") as f:
            f.write(_THREE)
        lower.run_module_passes(src, src, lower.SPAD_PLACEMENT, target=T)
        with _ctx(), ir.Location.unknown():
            with open(src) as f:
                m = ir.Module.parse(f.read())
            peak = _peak(m)
            for p in lower.TO_LLVM[:lower.TO_LLVM.index(lower.LOWER_DMA)]:
                p.run(m, vectorlane=T.vector_lanes, target=T, timing=False)
            reserved = p13.run(ir.Module.parse(str(m)), vectorlane=T.vector_lanes, target=T)
            assert [(n, c) for n, c, _ in reserved] == [("spad_arena", peak)], reserved
            unmarked = ir.Module.parse(str(m))
            del unmarked.operation.attributes[SPAD_ARENA]
            try:
                p13.run(unmarked, vectorlane=T.vector_lanes, target=T)
            except p13.SpadReservationRefused as e:
                assert SPAD_ARENA in str(e), str(e)
            else:
                raise AssertionError("an arena no marker names was reserved")


def materialize_refuses_an_alloc_no_placement_reached():
    from pytorchsim_triton_compiler.lowering.passes import p12ab_materialize_spad_allocs as mat
    with _ctx(), ir.Location.unknown():
        try:
            mat.run(ir.Module.parse(_THREE), vectorlane=T.vector_lanes)
        except mat.SpadAllocRefused as e:
            assert "offset" in str(e), str(e)
        else:
            raise AssertionError("an unplaced alloc was materialized")


def materialize_refuses_two_buffers_on_one_slot_alive_at_once():
    """a and b of _THREE given one offset by hand: both are stored before either is read, so they live
    at once, and materialization refuses rather than let one overwrite the other."""
    from pytorchsim_triton_compiler.lowering.passes import p11b_place_spad_buffers as place
    from pytorchsim_triton_compiler.lowering.passes import p12ab_materialize_spad_allocs as mat
    with _ctx(), ir.Location.unknown():
        m = _placed(ir.Module.parse(_THREE))
        allocs = [o for o in walk(m.operation) if o.name == "memref.alloc"]
        allocs[1].attributes[place.OFFSET] = allocs[0].attributes[place.OFFSET]
        try:
            mat.run(m, vectorlane=T.vector_lanes)
        except mat.SpadAllocRefused as e:
            assert "live at once" in str(e), str(e)
        else:
            raise AssertionError("two buffers alive at once were put on one slot")


def placement_ends_to_custom_and_materialization_follows_the_containment_check():
    """The offsets are in 09-custom.mlir, so the TOG, the ELF and the cycle binary all read them; the
    storage is made in TO_LLVM once the last access is checked, before the DMA is lowered."""
    from pytorchsim_triton_compiler.lowering.passes import (p11b_place_spad_buffers,
                                                            p12aa_check_spad_containment,
                                                            p12ab_materialize_spad_allocs)
    assert lower.TO_CUSTOM[-1] is p11b_place_spad_buffers, lower.TO_CUSTOM
    assert p11b_place_spad_buffers.MARKERS == ("memref.alloc",)
    i = lower.TO_LLVM.index(p12ab_materialize_spad_allocs)
    assert lower.TO_LLVM.index(p12aa_check_spad_containment) == i - 1, lower.TO_LLVM
    assert lower.TO_LLVM[i + 1] is lower.LOWER_DMA, lower.TO_LLVM


CASES = [slots_lie_end_to_end_in_entry_order_each_from_a_register_boundary,
         a_buffer_only_freed_gets_no_offset,
         buffers_that_share_a_slot_share_its_offset,
         a_slot_naming_no_keeper_is_refused,
         materialize_puts_every_slot_in_one_arena_the_module_names,
         a_second_buffer_through_a_bank_view_sits_at_its_own_offset,
         removing_the_arena_marker_is_refused_not_reserved_smaller,
         materialize_refuses_an_alloc_no_placement_reached,
         materialize_refuses_two_buffers_on_one_slot_alive_at_once,
         placement_ends_to_custom_and_materialization_follows_the_containment_check]


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
