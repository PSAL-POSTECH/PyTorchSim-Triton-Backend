"""tritonpytorchsim-check-spad-containment: an access past its own buffer's slot (reserve_per_lane, one
lane's share) is refused whatever sits beside it; one whose range is not static is counted unknown.
    python3 tests/passes/spad_containment_cases.py        exit 0 iff every case holds"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects import torchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lower  # noqa: E402

torchsim.register()
T = machine.target()


def _ctx():
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    return ctx


def _func(args, body, allocs, extra=""):
    decl = "\n".join(f"    %{n} = memref.alloc() : {t}" for n, t in allocs)
    free = "\n".join(f"    memref.dealloc %{n} : {t}" for n, t in reversed(allocs))
    return (f"module {{\n{extra}  func.func @f({args}) {{\n    %c0 = arith.constant 0 : index\n"
            f"    %c1 = arith.constant 1 : index\n{decl}\n{body}\n{free}\n    return\n  }}\n}}")


def _lowered(text):
    """`text` through TO_LLVM; the refusal's message, or None when it lowered."""
    from pytorchsim_triton_compiler.lowering.passes import p12aa_check_spad_containment as chk
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "in.mlir"), os.path.join(d, "out.mlir")
        with open(src, "w") as f:
            f.write(text)
        try:
            lower.run_module_passes(src, src, lower.SPAD_PLACEMENT, target=T)
            lower.to_llvm(src, out, target=T)
        except chk.SpadOverrun as e:
            return str(e)
    return None


def _audit(text):
    from pytorchsim_triton_compiler.lowering.passes import p12aa_check_spad_containment as chk
    with _ctx(), ir.Location.unknown():
        m = ir.Module.parse(text)
        assert m.operation.verify()
        a = chk.audit(m, T.vector_lanes, T)
        return a.checked, len(a.unknown), [(w, lo, hi, s) for _, w, lo, hi, s in a.outside]


_TILE = "memref<32x16xf32, 1>"


def a_vector_past_its_lanes_share_is_refused_whatever_follows_it():
    """32 lanes of 16: a lane's slot is 64 bytes. A vector<32> from its start writes 128, into whichever
    buffer .spad puts next -- refused with that neighbour first, last, or absent alike."""
    store = f"    vector.store %v, %a[%c0, %c0] : {_TILE}, vector<32xf32>"
    for allocs in ([("a", _TILE), ("b", _TILE)], [("b", _TILE), ("a", _TILE)], [("a", _TILE)]):
        got = _lowered(_func("%v: vector<32xf32>", store, allocs))
        assert got is not None and "[0, 128)" in got and "slot of 64" in got, (allocs, got)


def a_vector_inside_its_lanes_share_lowers():
    store = f"    vector.store %v, %a[%c0, %c0] : {_TILE}, vector<16xf32>"
    assert _lowered(_func("%v: vector<16xf32>", store, [("a", _TILE), ("b", _TILE)])) is None


def the_min_vec_slack_a_one_element_buffer_reserves_holds_a_vector_of_two():
    """p06 widens a one-element store to vector<2>, relying on reserve_per_lane rounding a lane's need up
    to MIN_VEC: that store is inside the slot, and one more element is not."""
    one = "memref<1xf32, 1>"
    two = f"    vector.store %v, %a[%c0] : {one}, vector<2xf32>"
    assert _lowered(_func("%v: vector<2xf32>", two, [("a", one)])) is None
    four = f"    vector.store %v, %a[%c0] : {one}, vector<4xf32>"
    got = _lowered(_func("%v: vector<4xf32>", four, [("a", one)]))
    assert got is not None and "slot of 8" in got, got


def a_loop_counter_puts_the_access_where_its_last_trip_reaches():
    """256 rows on 128 lanes: two rows of 16 a lane. Two trips stay inside; a third is the next slot."""
    deep = "memref<256x16xf32, 1>"
    for trips, inside in ((2, True), (3, False)):
        body = (f"    %n = arith.constant {trips} : index\n"
                f"    scf.for %i = %c0 to %n step %c1 {{\n"
                f"      vector.store %v, %a[%i, %c0] : {deep}, vector<16xf32>\n    }}")
        checked, unknown, outside = _audit(_func("%v: vector<16xf32>", body, [("a", deep)]))
        assert (not outside) == inside and checked + len(outside) == 1, (trips, outside)


def an_index_known_only_at_run_time_is_unknown_never_refused():
    body = f"    vector.store %v, %a[%c0, %i] : {_TILE}, vector<16xf32>"
    text = _func("%v: vector<16xf32>, %i: index", body, [("a", _TILE)])
    assert _audit(text) == (0, 1, []), _audit(text)
    assert _lowered(text) is None


def a_bank_view_describing_bytes_past_the_slot_is_refused_before_anything_reads_it():
    def view(o):
        return (f'    %w = "torchsim.bank_view"(%a) {{offset = {o} : i64, sizes = array<i64: 1, 16>, '
                f'strides = array<i64: 16, 1>}} : ({_TILE}) -> memref<1x16xf32, strided<[16, 1], '
                f'offset: {o}>, 1>')
    assert _audit(_func("", view(0), [("a", _TILE)]))[2] == []
    got = _audit(_func("", view(8), [("a", _TILE)]))[2]
    assert got == [("view", 32, 96, 64)], got


def a_transfers_footprint_is_the_dmas_own_walk_of_its_descriptor():
    """256 rows on 128 lanes, MVIN'd at element `at` of a lane's bank: the DMA walks two laps of 16
    from there. At 0 it fills the slot; at 16 its second lap is the next buffer's."""
    deep = "memref<256x16xf32, 1>"
    tx = ('    %at = arith.constant {at} : index\n'
          '    "torchsim.transfer"(%d, %c0, %a, %at, %c0) <{{dma_kind = "MVIN", dram_arg = 0 : i64, '
          'dram_stride = [16, 1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}> : '
          f'(memref<4096xf32>, index, {deep}, index, index) -> ()')
    assert _audit(_func("%d: memref<4096xf32>", tx.format(at=0), [("a", deep)])) == (1, 0, [])
    got = _audit(_func("%d: memref<4096xf32>", tx.format(at=16), [("a", deep)]))[2]
    assert got == [("transfer", 64, 192, 128)], got


def an_op_that_states_nothing_or_lets_the_address_go_is_unknown_never_dropped():
    """A call, an unregistered op (no MemoryEffectOpInterface), and a pointer taken out of the buffer
    (Pure, no effect on it, no view) each reach the buffer in a way no index states: counted unknown.
    Before, effects_on's "unknown" and an empty effect set fell out of both counts: (0, 0, [])."""
    one = "memref<1x16xf32, 1>"
    call = _func("", f"    func.call @ext(%a) : ({_TILE}) -> ()", [("a", _TILE)],
                 extra=f"  func.func private @ext({_TILE})\n")
    poke = _func("%v: vector<32xf32>", f'    "test.poke"(%v, %a, %c0, %c0) : (vector<32xf32>, '
                 f'{_TILE}, index, index) -> ()', [("a", _TILE)])
    leak = _func("%x: f32", f"""    %p = memref.extract_aligned_pointer_as_index %a : {one} -> index
    %i = arith.index_cast %p : index to i64
    %c400 = arith.constant 400 : i64
    %q = arith.addi %i, %c400 : i64
    %ptr = llvm.inttoptr %q : i64 to !llvm.ptr
    llvm.store %x, %ptr : f32, !llvm.ptr""", [("a", one)])
    for text in (call, poke, leak):
        assert _audit(text) == (0, 1, []), (_audit(text), text)
    assert _lowered(call) is None and _lowered(leak) is None


def a_masked_access_moves_what_its_mask_says_so_it_is_unknown_not_refused():
    """vector<32> on a 64-byte slot with 16 of 32 lanes on reaches 64 bytes, inside: refusing it as if
    every lane were on (the reviewer's maskedstore_legal_mask, transfer_explicit_mask_legal) is wrong.
    Which lanes are on is the mask's run-time value: unknown. An all-true constant mask is no mask."""
    mask = "    %c16 = arith.constant 16 : index\n    %m = vector.create_mask %c16 : vector<32xi1>\n"
    ones = "    %m = arith.constant dense<true> : vector<32xi1>\n"
    store = f"    vector.maskedstore %a[%c0, %c0], %m, %v : {_TILE}, vector<32xi1>, vector<32xf32>"
    write = f"    vector.transfer_write %v, %a[%c0, %c0], %m {{in_bounds = [true]}} : vector<32xf32>, {_TILE}"
    for body in (store, write):
        text = _func("%v: vector<32xf32>", mask + body, [("a", _TILE)])
        assert _audit(text) == (0, 1, []), (_audit(text), body)
        assert _lowered(text) is None, body
        got = _audit(_func("%v: vector<32xf32>", ones + body, [("a", _TILE)]))[2]
        assert [w[1:] for w in got] == [(0, 128, 64)], (got, body)


def the_footprint_walks_the_descriptor_the_encoder_packs():
    """sram_footprints and _emit_direct take a transfer's descriptor and sram indices from one helper,
    lower_dma._direct: a change there reaches the packed slots and the footprint alike. Before, the
    footprint re-spelled _four_dims' arguments and the address beside the encoder's."""
    from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import lower_dma as L
    from pytorchsim_triton_compiler.lowering.passes import p12aa_check_spad_containment as chk
    deep = "memref<256x16xf32, 1>"
    text = _func("%d: memref<4096xf32>",
                 '    "torchsim.transfer"(%d, %c0, %a, %c0, %c0) <{dma_kind = "MVIN", dram_arg = 0 : '
                 'i64, dram_stride = [16, 1], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, '
                 f'0, 0>}}> : (memref<4096xf32>, index, {deep}, index, index) -> ()', [("a", deep)])
    direct, pack = L._direct, L._pack_desc
    packed, seen = [], []

    def wide(tx):
        d4, sidx = direct(tx)
        return d4._replace(spad=[2 * x for x in d4.spad]), sidx

    def record(shape4, dram4, spad4, *a, **k):
        packed.append(list(spad4))
        return pack(shape4, dram4, spad4, *a, **k)
    try:
        L._pack_desc = record
        for use in (direct, wide):
            L._direct = use
            with _ctx(), ir.Location.unknown():
                m = ir.Module.parse(text)
                seen.append([w[1:] for w in chk.audit(m, T.vector_lanes, T).outside])
                L.run(m, vectorlane=T.vector_lanes)
    finally:
        L._direct, L._pack_desc = direct, pack
    assert packed[1] == [2 * x for x in packed[0]] and packed[0] != packed[1], packed
    assert seen[0] == [] and seen[1] and seen[1][0][3] == 128 < seen[1][0][2], seen


def the_counts_reach_the_stage_log():
    """lower.run hands to_llvm its ctx.log: the checked and unknown counts are kept with the run,
    not dropped by run(). Before, to_llvm took no log and the unknowns were invisible."""
    known = f"    vector.store %v, %a[%c0, %c0] : {_TILE}, vector<16xf32>\n"
    late = f"    vector.store %v, %a[%c0, %i] : {_TILE}, vector<16xf32>"
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "in.mlir"), os.path.join(d, "out.mlir")
        with open(src, "w") as f:
            f.write(_func("%v: vector<16xf32>, %i: index", known + late, [("a", _TILE)]))
        lower.run_module_passes(src, src, lower.SPAD_PLACEMENT, target=T)
        lines = []
        lower.to_llvm(src, out, target=T, log=lines.append)
    said = [x for x in lines if "check-spad-containment" in x]
    assert said == ["[lower] tritonpytorchsim-check-spad-containment: 1 access(es) inside their "
                    "slot, 1 unknown, 0 outside"], lines


def the_check_runs_in_to_llvm_after_the_last_access_is_made_and_before_buffers_are_placed():
    from pytorchsim_triton_compiler.lowering.passes import (p12a_lower_lane_reductions,
                                                            p12aa_check_spad_containment,
                                                            p12ab_materialize_spad_allocs)
    at = lower.TO_LLVM.index(p12aa_check_spad_containment)
    assert lower.TO_LLVM.index(p12a_lower_lane_reductions) < at, lower.TO_LLVM
    assert at < lower.TO_LLVM.index(p12ab_materialize_spad_allocs), lower.TO_LLVM


CASES = [a_vector_past_its_lanes_share_is_refused_whatever_follows_it,
         a_vector_inside_its_lanes_share_lowers,
         the_min_vec_slack_a_one_element_buffer_reserves_holds_a_vector_of_two,
         a_loop_counter_puts_the_access_where_its_last_trip_reaches,
         an_index_known_only_at_run_time_is_unknown_never_refused,
         a_bank_view_describing_bytes_past_the_slot_is_refused_before_anything_reads_it,
         a_transfers_footprint_is_the_dmas_own_walk_of_its_descriptor,
         an_op_that_states_nothing_or_lets_the_address_go_is_unknown_never_dropped,
         a_masked_access_moves_what_its_mask_says_so_it_is_unknown_not_refused,
         the_footprint_walks_the_descriptor_the_encoder_packs,
         the_counts_reach_the_stage_log,
         the_check_runs_in_to_llvm_after_the_last_access_is_made_and_before_buffers_are_placed]


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
