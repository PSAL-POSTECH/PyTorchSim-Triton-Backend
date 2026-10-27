"""What pytorchsim-mark-runtime-attributes records, and what it refuses by name, on IR no kernel needs
to reach; what pytorchsim-lower-runtime-attributes makes of a record; the loop and branch queries both read.
    python3 tests/passes/runtime_attribute_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.contract.stop import Reason, Refused  # noqa: E402
from pytorchsim_triton_compiler.dialect import pytorchsim  # noqa: E402
from pytorchsim_triton_compiler.passes.transforms import (
    mark_runtime_attributes as mark, lower_runtime_attributes as lower)
from pytorchsim_triton_compiler.passes.utils.ir_utils import walk  # noqa: E402

pytorchsim.register()

SEG = "operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>"


def _mvin(dram, dram_ty, idx, buf, buf_ty, arg):
    return (f'"pytorchsim.transfer"({dram}, {idx}, {buf}, %c0, %c0) <{{direction = #pytorchsim<direction load>, dram_arg = {arg} : '
            f'i64, dram_stride = [1], {SEG}}}> : ({dram_ty}, index, {buf_ty}, index, index) -> ()')


def _module(body, args="%ids: memref<32xi32>, %x: memref<4096xf32>, %pid: i32"):
    return ir.Module.parse(
        f"module {{\n func.func @k({args}) {{\n  %c0 = arith.constant 0 : index\n"
        f"  %c1 = arith.constant 1 : index\n  %c4 = arith.constant 4 : index\n"
        f"  %c16 = arith.constant 16 : i32\n  %buf = memref.alloc() : memref<i32, 1>\n"
        f"  %t = memref.alloc() : memref<16xf32, 1>\n  %p = arith.index_cast %pid : i32 to index\n"
        f"{body}\n  return\n }}\n}}\n")


def _scalar_body(feeds=True):
    use = "%o = arith.muli %e, %c16 : i32\n  %oi = arith.index_cast %o : i32 to index" if feeds else \
        "%oi = arith.index_cast %pid : i32 to index"
    return (f"  {_mvin('%ids', 'memref<32xi32>', '%p', '%buf', 'memref<i32, 1>', 0)}\n"
            f"  %e = memref.load %buf[] : memref<i32, 1>\n  {use}\n"
            f"  {_mvin('%x', 'memref<4096xf32>', '%oi', '%t', 'memref<16xf32, 1>', 1)}")


def _loop_body(ub="%c4"):
    return (f"  scf.for %j = %c0 to {ub} step %c1 {{\n"
            f"    %a = arith.muli %p, %c4 : index\n    %b = arith.addi %a, %j : index\n"
            f"    {_mvin('%ids', 'memref<32xi32>', '%b', '%buf', 'memref<i32, 1>', 0)}\n"
            f"    %e = memref.load %buf[] : memref<i32, 1>\n    %o = arith.muli %e, %c16 : i32\n"
            f"    %oi = arith.index_cast %o : i32 to index\n"
            f"    {_mvin('%x', 'memref<4096xf32>', '%oi', '%t', 'memref<16xf32, 1>', 1)}\n  }}")


def _recs(m):
    return [(o.opview.form.value, o.opview.id.value, o.opview.rank.value) for o in mark.records(m)]


def _refused(body, args=None):
    m = _module(body, **({"args": args} if args else {}))
    try:
        mark.run(m, grid=(4,))
    except Refused as e:
        assert e.reason is Reason.RUNTIME_ATTRIBUTE, e.reason
        return str(e)
    return "no refusal"


def a_read_reaching_an_address_outside_loops_is_one_scalar_record():
    """Breaks if the read feeding a DMA address is not recorded as S where it is read, or the address
    stops resting on what S yields (the TOG would read the static 0)."""
    m = _module(_scalar_body())
    mark.run(m, grid=(4,))
    assert _recs(m) == [("S", 0, 0)], _recs(m)
    s = mark.records(m)[0]
    assert any(u.owner.name == "arith.muli" for u in s.results[0].uses)


def a_function_no_read_reaches_is_left_as_it_was():
    """Breaks if marking touches a kernel whose TOG reads no data: its 09-custom.mlir would move."""
    m = _module(_scalar_body(feeds=False))
    before = str(m)
    mark.run(m, grid=(4,))
    assert str(m) == before


def an_in_loop_read_is_an_array_record_before_its_loop_and_a_read_at_it():
    """Breaks if an in-loop read is not one A before the loop (stride 1, the loop's 4 trips as capacity)
    and R at the read indexed by the trip."""
    m = _module(_loop_body())
    mark.run(m, grid=(4,))
    assert _recs(m) == [("A", 0, 1), ("R", 0, 1)], _recs(m)
    a, r = mark.records(m)
    assert list(a.opview.strides) == [1] and list(a.opview.capacity) == [4], (a.opview.strides, a.opview.capacity)
    assert a.parent.name == "func.func" and r.parent.name == "scf.for"


def a_data_trip_count_takes_its_capacity_from_the_tensor():
    """Breaks if a loop bound read from data leaves the array record with no static capacity: the
    32 elements of the table its stride 1 reaches."""
    m = _module(f"  {_mvin('%ids', 'memref<32xi32>', '%p', '%buf', 'memref<i32, 1>', 0)}\n"
                f"  %n = memref.load %buf[] : memref<i32, 1>\n  %ni = arith.index_cast %n : i32 to index\n"
                f"  %u = memref.alloc() : memref<i32, 1>\n"
                + _loop_body("%ni").replace("%buf", "%u"))
    mark.run(m, grid=(4,))
    assert _recs(m) == [("S", 0, 0), ("A", 1, 1), ("R", 1, 1)], _recs(m)
    assert list(mark.records(m)[1].opview.capacity) == [32]


def a_large_launch_of_records_is_not_refused():
    """Breaks if marking refuses a launch by the size of its records: 65536 programs of a 4096-trip
    array record write 2 GiB of records, which the wrapper streams to its record file."""
    m = _module("  %c4096 = arith.constant 4096 : index\n" + _loop_body("%c4096"))
    mark.run(m, grid=(1 << 16,))
    assert _recs(m) == [("A", 0, 1), ("R", 0, 1)], _recs(m)
    assert list(mark.records(m)[0].opview.capacity) == [4096]


def an_array_with_no_static_capacity_is_left_unrecorded():
    """Breaks if a record whose trip count is data and whose table has no static size is recorded (its
    row could not be sized) or stops lower (the ELF needs no record); only the TOG is refused."""
    m = _module(
        f"  {_mvin('%ids', 'memref<?xi32>', '%p', '%buf', 'memref<i32, 1>', 0)}\n"
        "  %n = memref.load %buf[] : memref<i32, 1>\n  %ni = arith.index_cast %n : i32 to index\n"
        "  %u = memref.alloc() : memref<i32, 1>\n"
        + _loop_body("%ni").replace("%buf", "%u").replace("memref<32xi32>", "memref<?xi32>"),
        args="%ids: memref<?xi32>, %x: memref<4096xf32>, %pid: i32")
    mark.run(m, grid=(4,))
    assert not any(r[0] == "A" for r in _recs(m)), _recs(m)
    try:
        mark.refuse_unrecorded(m)
    except Refused as e:
        assert e.reason is Reason.RUNTIME_ATTRIBUTE and "no static capacity on dim 0" in str(e), str(e)
    else:
        raise AssertionError("no refusal at tog")


def a_chased_pointer_is_refused():
    """Breaks if an address resting on what an earlier trip read is recorded: no record of the run
    states where the next trip goes before the program has read it."""
    got = _refused(
        "  %h = arith.index_cast %pid : i32 to index\n"
        "  %r = scf.for %j = %c0 to %c4 step %c1 iter_args(%cur = %h) -> (index) {\n"
        f"    {_mvin('%ids', 'memref<32xi32>', '%cur', '%buf', 'memref<i32, 1>', 0)}\n"
        "    %e = memref.load %buf[] : memref<i32, 1>\n    %ei = arith.index_cast %e : i32 to index\n"
        f"    {_mvin('%x', 'memref<4096xf32>', '%ei', '%t', 'memref<16xf32, 1>', 1)}\n"
        "    scf.yield %ei : index\n  }")
    assert "pointer chasing" in got, got


def a_two_level_read_in_one_trip_is_refused():
    """Breaks if `remap[bt[j]]` is recorded as one array: its second level depends on the first in the
    same trip."""
    got = _refused(
        "  %v = memref.alloc() : memref<i32, 1>\n"
        "  scf.for %j = %c0 to %c4 step %c1 {\n"
        f"    {_mvin('%ids', 'memref<32xi32>', '%j', '%buf', 'memref<i32, 1>', 0)}\n"
        "    %e = memref.load %buf[] : memref<i32, 1>\n    %ei = arith.index_cast %e : i32 to index\n"
        f"    {_mvin('%rm', 'memref<32xi32>', '%ei', '%v', 'memref<i32, 1>', 2)}\n"
        "    %f = memref.load %v[] : memref<i32, 1>\n    %fi = arith.index_cast %f : i32 to index\n"
        f"    {_mvin('%x', 'memref<4096xf32>', '%fi', '%t', 'memref<16xf32, 1>', 1)}\n  }}",
        args="%ids: memref<32xi32>, %x: memref<4096xf32>, %rm: memref<32xi32>, %pid: i32")
    assert "two-level indirection" in got, got


def a_float_offset_is_refused():
    """Breaks if a float read turned into an address is recorded: records hold integers alone."""
    got = _refused(
        "  %fb = memref.alloc() : memref<f32, 1>\n"
        f"  {_mvin('%x', 'memref<4096xf32>', '%p', '%fb', 'memref<f32, 1>', 1)}\n"
        "  %f = memref.load %fb[] : memref<f32, 1>\n  %e = arith.fptosi %f : f32 to i32\n"
        "  %ei = arith.index_cast %e : i32 to index\n"
        f"  {_mvin('%x', 'memref<4096xf32>', '%ei', '%t', 'memref<16xf32, 1>', 1)}")
    assert "float value" in got, got


def _cut(m):
    from pytorchsim_triton_compiler.togsim import build_tog as bt
    builder = bt.TogBuilder()
    bt._build(m, builder)
    return builder


def a_compute_loop_whose_trip_count_is_data_is_replayed():
    """Breaks if a loop holding no DMA, whose trip count is data, is refused or left inside a segment:
    its bound is recorded (S) and the TOG replays it around its compute, the sample timing one trip."""
    m = _module(f"  {_mvin('%ids', 'memref<32xi32>', '%p', '%buf', 'memref<i32, 1>', 0)}\n"
                "  %n = memref.load %buf[] : memref<i32, 1>\n  %ni = arith.index_cast %n : i32 to index\n"
                "  scf.for %j = %c0 to %ni step %c1 {\n    %z = arith.addi %j, %c1 : index\n  }")
    mark.run(m, grid=(4,))
    assert _recs(m) == [("S", 0, 0)], _recs(m)
    b = _cut(m)
    assert [n.op.operation.name for n in b.loop_nodes] == ["scf.for"], b.loop_nodes
    inner = [c for c in b.compute_nodes if c.operations[0].operation.parent.name == "scf.for"]
    assert [[o.operation.name for o in c.operations] for c in inner] == [["arith.addi"]]


def _branch(cond):
    return (f"{cond}\n  scf.if %c {{\n    %q = arith.addi %p, %c1 : index\n  }} else {{\n"
            "    %q = arith.muli %p, %c4 : index\n    %r = arith.addi %q, %c1 : index\n  }")


def a_branch_on_a_float_read_records_its_condition():
    """Breaks if a compute-only branch whose condition compares a float read is refused: the condition,
    computed outside every loop, is recorded as itself (S, an i1), which the TOG branches on."""
    m = _module("  %fb = memref.alloc() : memref<f32, 1>\n"
                f"  {_mvin('%x', 'memref<4096xf32>', '%p', '%fb', 'memref<f32, 1>', 1)}\n"
                + _branch("  %f = memref.load %fb[] : memref<f32, 1>\n  %z = arith.constant 0.0 : f32\n"
                          "  %c = arith.cmpf ogt, %f, %z : f32"))
    mark.run(m, grid=(4,))
    assert _recs(m) == [("S", 0, 0)], _recs(m)
    rec = mark.records(m)[0]
    assert rec.operands[0].owner.name == "arith.cmpf" and list(ir.DenseI64ArrayAttr(rec.attributes[mark.SOURCES])) == [1]
    assert any(u.owner.name == "scf.if" for u in rec.results[0].uses)


def a_compute_only_branch_cuts_a_segment_per_arm():
    """Breaks if a branch whose arms only compute is folded into the segment around it: the TOG must
    hold it, each arm its own segment, so that it picks the arm a program runs (here by its id)."""
    m = _module(_branch("  %c = arith.cmpi eq, %pid, %c16 : i32"))
    mark.run(m, grid=(4,))
    b = _cut(m)
    assert [n.op.operation.name for n in b.branch_nodes] == ["scf.if"], b.branch_nodes
    arms = [[o.operation.name for o in c.operations] for c in b.compute_nodes]
    assert arms[-2:] == [["arith.addi"], ["arith.muli", "arith.addi"]], arms


def a_record_names_the_arguments_it_is_read_from():
    """Breaks if a record does not state the function arguments its value comes from (S, A and R read
    %ids, argument 0; a buffer copied from one %ids filled reads it too): a run on other values there
    records another routing than the launch's."""
    got = []
    for body in (_scalar_body(), _loop_body(),
                 _scalar_body().replace("%e = memref.load %buf[]", "%cp = memref.alloc() : memref<i32, 1>\n"
                                        "  memref.copy %buf, %cp : memref<i32, 1> to memref<i32, 1>\n"
                                        "  %e = memref.load %cp[]")):
        m = _module(body)
        mark.run(m, grid=(4,))
        got.append([(o.opview.form.value, list(ir.DenseI64ArrayAttr(o.attributes[mark.SOURCES])))
                    for o in mark.records(m)])
    assert got == [[("S", [0])], [("A", [0]), ("R", [0])], [("S", [0])]], got


def _calls(m):
    return [o for o in walk(m.operation) if o.name == "func.call"]


def a_scalar_record_lowers_to_one_write_from_a_stack_temporary():
    """Breaks if the ELF lowering leaves a record, writes S from anything but an entry-block stack
    temporary of 8 bytes, or states a table with a row's layout or without its hash."""
    from pytorchsim_triton_compiler.contract import kernel_object as ko
    m = _module(_scalar_body())
    mark.run(m, grid=(4,))
    with ir.Location.unknown():
        lower.run(m)
    assert not mark.records(m)
    calls = _calls(m)
    assert len(calls) == 1 and ir.FlatSymbolRefAttr(calls[0].attributes["callee"]).value == ko.RUNTIME_ATTRIBUTE_WRITE
    assert ir.IntegerAttr(calls[0].operands[2].owner.attributes["value"]).value == 8
    alloca = [o for o in walk(m.operation) if o.name == "memref.alloca"]
    assert len(alloca) == 1 and alloca[0].parent.name == "func.func", alloca
    import json
    table = json.loads(ir.StringAttr(m.operation.attributes[lower.TABLE]).value)
    attributes = [{"id": 0, "rank": 0, "capacity": [], "fill": None, "elem_bits": 64, "sources": [0]}]
    assert table == {"table_hash": lower.table_hash(attributes), "attributes": attributes}, table


def an_array_record_writes_its_extents_and_values_from_the_staging_buffer():
    """Breaks if A stops writing from the one staging buffer, sized for its rank and capacity of i64,
    or writes a byte count that does not follow its extents."""
    m = _module(_loop_body())
    mark.run(m, grid=(4,))
    with ir.Location.unknown():
        lower.run(m)
    calls = _calls(m)
    assert len(calls) == 1 and calls[0].parent.name == "func.func", calls
    g = [o for o in walk(m.operation) if o.name == "memref.global"]
    assert len(g) == 1 and ir.StringAttr(g[0].attributes["sym_name"]).value == lower.STAGING
    assert ir.MemRefType(ir.TypeAttr(g[0].attributes["type"]).value).shape == [8 * (1 + 4)]
    assert calls[0].operands[2].owner.name == "arith.index_cast", calls[0].operands[2].owner


def a_snapshot_record_is_one_trip_row():
    """Breaks if a G record is sized for more than its extents and one trip's row of its element width:
    a 2-trip snapshot of 8 i32 entries writes 2 extents and 32 bytes."""
    assert lower.record_bytes({"rank": 2, "capacity": [2, 8], "fill": 1, "elem_bits": 32}) == 16 + 32
    assert lower.record_bytes({"rank": 1, "capacity": [3], "fill": None, "elem_bits": 64}) == 8 + 24
    assert lower.record_bytes({"rank": 0, "capacity": [], "fill": None, "elem_bits": 64}) == 8


def the_timed_binary_erases_every_record():
    """Breaks if the cycle binary keeps a record or what was made for one: gem5 would time stores the
    TOG never states."""
    m = _module(_loop_body())
    before = str(m)
    mark.run(m, grid=(4,))
    with ir.Location.unknown():
        lower.run(m, timing=True)
    assert str(m) == before


def the_skeleton_keeps_an_iter_arg_a_dma_address_runs_by():
    """Breaks if build_skeleton reads an accumulated DMA offset as its init: a prefix sum the TOG must
    recompute would place every trip at the first trip's address."""
    from pytorchsim_triton_compiler.togsim import build_skeleton as bs
    from pytorchsim_triton_compiler.togsim import togsim_ops as ts
    m = _module(
        "  %r = scf.for %j = %c0 to %c4 step %c1 iter_args(%off = %c0) -> (index) {\n"
        f"    {_mvin('%x', 'memref<4096xf32>', '%off', '%t', 'memref<16xf32, 1>', 1)}\n"
        "    %n = arith.addi %off, %j : index\n    scf.yield %n : index\n  }")
    bs.build_skeleton(m)
    loop = next(o for o in walk(m.operation) if o.name == "scf.for")
    dma = next(o for o in walk(m.operation) if o.name == ts.DMA)
    assert len(loop.results) == 1, str(loop)
    assert dma.operands[0] == loop.regions[0].blocks[0].arguments[1], str(dma)


def loops_and_branches_are_read_off_their_interfaces():
    """Breaks if loop_like or branch_controls stop answering from LoopLikeOpInterface and
    RegionBranchOpInterface: the marking pass would see no loop and no branch."""
    m = ir.Module.parse(
        "func.func @f(%c: i1, %n: index) {\n  %c0 = arith.constant 0 : index\n  %c1 = arith.constant 1 : index\n"
        "  scf.for %i = %c0 to %n step %c1 {\n  }\n  affine.for %k = 0 to 8 step 2 {\n  }\n"
        "  scf.if %c {\n  }\n  return\n}\n")
    ops = {o.name: o for o in walk(m.operation)}
    loop = pytorchsim.loop_like(ops["scf.for"])
    assert loop.ubs[0] == list(ops["scf.for"].operands)[1] and len(loop.ivs) == 1
    aff = pytorchsim.loop_like(ops["affine.for"])
    assert (aff.lbs, aff.ubs, aff.steps) == ([0], [8], [2]), aff
    assert pytorchsim.branch_controls(ops["scf.if"]) == [list(ops["scf.if"].operands)[0]]
    assert pytorchsim.branch_controls(ops["scf.for"]) is None and pytorchsim.loop_like(ops["scf.if"]) is None


def a_branch_that_forwards_nothing_is_asked_without_reading_past_its_operands():
    """Breaks (SIGSEGV) if branch_controls asks an empty entry range for its operand index: that range
    sits at the operands' end, and its index is read off memory past the op."""
    m = ir.Module.parse("func.func @f(%c: i1) {\n" + "  scf.if %c {\n  }\n" * 50 + "  return\n}\n")
    ifs = [o for o in walk(m.operation) if o.name == "scf.if"]
    for _ in range(20):
        for o in ifs:
            assert pytorchsim.branch_controls(o) == [list(o.operands)[0]]


def a_while_is_read_off_its_interfaces():
    """Breaks if is_loop, region_inputs or terminator_controls stop answering from LoopLikeOpInterface and
    RegionBranchOpInterface: the TOG would not see a while as a loop, nor what decides it or feeds its args."""
    m = ir.Module.parse(
        "func.func @f(%n: i32) {\n  %c0 = arith.constant 0 : i32\n  %c1 = arith.constant 1 : i32\n"
        "  %r = scf.while (%i = %c0) : (i32) -> i32 {\n    %c = arith.cmpi slt, %i, %n : i32\n"
        "    scf.condition(%c) %i : i32\n  } do {\n  ^bb0(%j: i32):\n    %k = arith.addi %j, %c1 : i32\n"
        "    scf.yield %k : i32\n  }\n  return\n}\n")
    ops = {o.name: o for o in walk(m.operation)}
    w, cond, cmp, add = ops["scf.while"], ops["scf.condition"], ops["arith.cmpi"], ops["arith.addi"]
    assert pytorchsim.is_loop(w) and pytorchsim.loop_like(w) is None and not pytorchsim.is_loop(cmp)
    before, after = (r.blocks[0].arguments[0] for r in w.regions)
    assert pytorchsim.region_inputs(before) == [list(w.operands)[0], add.results[0]], pytorchsim.region_inputs(before)
    assert pytorchsim.region_inputs(after) == [before]
    assert pytorchsim.terminator_controls(cond) == [cmp.results[0]]
    assert pytorchsim.terminator_controls(ops["scf.yield"]) == [] and pytorchsim.terminator_controls(cmp) is None


CASES = [a_read_reaching_an_address_outside_loops_is_one_scalar_record,
         a_function_no_read_reaches_is_left_as_it_was,
         an_in_loop_read_is_an_array_record_before_its_loop_and_a_read_at_it,
         a_data_trip_count_takes_its_capacity_from_the_tensor,
         a_large_launch_of_records_is_not_refused,
         an_array_with_no_static_capacity_is_left_unrecorded, a_chased_pointer_is_refused, a_two_level_read_in_one_trip_is_refused, a_float_offset_is_refused,
         a_compute_loop_whose_trip_count_is_data_is_replayed, a_branch_on_a_float_read_records_its_condition,
         a_compute_only_branch_cuts_a_segment_per_arm, a_scalar_record_lowers_to_one_write_from_a_stack_temporary,
         an_array_record_writes_its_extents_and_values_from_the_staging_buffer, a_snapshot_record_is_one_trip_row,
         a_record_names_the_arguments_it_is_read_from, the_timed_binary_erases_every_record, the_skeleton_keeps_an_iter_arg_a_dma_address_runs_by,
         loops_and_branches_are_read_off_their_interfaces, a_while_is_read_off_its_interfaces,
         a_branch_that_forwards_nothing_is_asked_without_reading_past_its_operands]


def main():
    import gc
    failed = 0
    for case in CASES:
        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        try:
            with ctx, ir.Location.unknown():
                case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {e}")
        gc.collect()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
