"""tritonpytorchsim-place-spad-buffers, TO_CUSTOM's last: every scratchpad alloc a per-lane byte offset
(slots end to end in entry order, each reserve_per_lane x element bytes from a register boundary, shared
only as torchsim-share-spad-buffers says, none when only freed); materialization takes those slots as is.
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


def materialize_makes_one_global_a_slot_in_offset_order():
    """_THREE's three slots are three globals in their offsets' order, the only-freed buffer none; the
    two buffers of _ONE_AFTER_THE_OTHER on one slot read one global."""
    from pytorchsim_triton_compiler.lowering.passes import p12ab_materialize_spad_allocs as mat
    with _ctx(), ir.Location.unknown():
        m = _placed(ir.Module.parse(_THREE))
        mat.run(m, vectorlane=T.vector_lanes)
        assert m.operation.verify()
        ops = list(walk(m.operation))
        types = [str(ir.TypeAttr(o.attributes["type"]).value) for o in ops if o.name == "memref.global"]
        assert types == ["memref<4xf32, 1>", "memref<1x40xf32, 1>", "memref<8x3xbf16, 1>"], types
        assert not [o for o in ops if o.name in ("memref.alloc", "memref.dealloc")]
        m = _placed(ir.Module.parse(_ONE_AFTER_THE_OTHER))
        mat.run(m, vectorlane=T.vector_lanes)
        ops = list(walk(m.operation))
        assert sum(o.name == "memref.global" for o in ops) == 1, m
        named = {ir.FlatSymbolRefAttr(o.attributes["name"]).value for o in ops if o.name == "memref.get_global"}
        assert len(named) == 1, named


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
         materialize_makes_one_global_a_slot_in_offset_order,
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
