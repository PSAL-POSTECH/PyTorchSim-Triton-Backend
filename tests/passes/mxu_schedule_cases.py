"""tritonpytorchsim-schedule-mxu-stream issues the matrix unit's stream as the lowering gave it and places
the rest around it: a steady loop is pipelined (next input's reads, writes of two inputs back, the pops
with the last input's adds), a panel's weights step from one base.   python3 tests/passes/mxu_schedule_cases.py"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler import toolchain  # noqa: E402
from pytorchsim_triton_compiler.contract import target as _target  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import lower_matmul as mxu  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import schedule_mxu_stream as sched  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import trip_count  # noqa: E402

MXU = _target.resolve("tpuv6e")
EPI = os.path.join(ROOT, "tests", "cases", "mechanism", "tile", "tile_gemm_epilogue_at_pop.py")
WIDE = os.path.join(ROOT, "tests", "cases", "mechanism", "tile", "tile_gemm_wide_tile_grid.py")
LOWER_DUMP = f"01-{mxu.NAME}.mlir"
SCHED_DUMP = f"02-{sched.NAME}.mlir"
_LOWERED = {}


def _lowered(kernel):
    """(the lowering's dump, the schedule pass's or None, 09-custom.mlir) of `kernel` compiled to
    lower on the v6e, as text."""
    if kernel not in _LOWERED:
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), kernel,
                            "--workdir", d, "--to", "lower"], check=True, capture_output=True,
                           env=dict(os.environ, TORCHSIM_COMPILE_TARGET=MXU,
                                    TORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
            got = [os.path.join(d, "09-passes", LOWER_DUMP), os.path.join(d, "09-passes", SCHED_DUMP),
                   os.path.join(d, "09-custom.mlir")]
            _LOWERED[kernel] = tuple(open(p).read() if os.path.exists(p) else None for p in got)
    return _LOWERED[kernel]


def _parse(text):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        return ctx, ir.Module.parse(text)


def _is_vcix(o):
    return o.name.startswith("vcix.")


def _stream(op, out):
    """The vcix ops under `op` in the order they run, as (name, opcode, imm): a loop of constant
    trips its body that many times, an op of one region that region once, of two or more none."""
    for region in op.regions:
        for block in region.blocks:
            for o in [x.operation for x in block.operations]:
                if _is_vcix(o):
                    out.append((o.name, ir.IntegerAttr(o.attributes["opcode"]).value,
                                ir.IntegerAttr(o.attributes["imm"]).value))
                elif o.name in ("affine.for", "scf.for"):
                    trips = trip_count(o)
                    assert trips is not None, f"a loop of unknown trips drives the unit: {o.name}"
                    for _ in range(trips):
                        _stream(o, out)
                elif len(o.regions) == 1:
                    _stream(o, out)
                else:
                    assert not any(_is_vcix(x) for x in _all(o)), o.name
    return out


def _all(op):
    out = []
    for region in op.regions:
        for block in region.blocks:
            for o in [x.operation for x in block.operations]:
                out.append(o)
                out.extend(_all(o))
    return out


def _steady(module):
    """(loop, body) of every innermost loop that both pushes and pops."""
    out = []
    for o in _all(module.operation):
        if o.name != "affine.for":
            continue
        body = [x.operation for x in o.regions[0].blocks[0].operations]
        if any(x.regions for x in body):
            continue
        units = [x for x in body if _is_vcix(x)]
        if any(len(x.results) for x in units) and any(not len(x.results) for x in units):
            out.append((o, body))
    return out


def _carried(loop, v):
    """How many trips back the value `v` of `loop`'s body was made: 0 when this trip made it, k when
    it is the iter_arg k yields deep; None when it comes from outside the loop."""
    block = loop.regions[0].blocks[0]
    args = list(block.arguments)[1:]
    yielded = list([x.operation for x in block.operations][-1].operands)
    depth = 0
    while isinstance(v, ir.BlockArgument) and v in args:
        v, depth = yielded[args.index(v)], depth + 1
    owner = v.owner if not isinstance(v, ir.BlockArgument) else None
    inside = owner is not None and owner.operation.parent == loop
    return depth if inside or depth else None


def _origin(loop, v):
    """(trips back, op name) of what made `v`, through the values of `loop`'s body and its iter_args."""
    block = loop.regions[0].blocks[0]
    args = list(block.arguments)[1:]
    yielded = list([x.operation for x in block.operations][-1].operands)
    depth = 0
    while True:
        if isinstance(v, ir.BlockArgument):
            v, depth = yielded[args.index(v)], depth + 1
            continue
        o = v.owner.operation
        if o.name in ("arith.bitcast", "vector.extract_strided_slice"):
            v = o.operands[0]
            continue
        return depth, o.name


def the_unit_runs_the_same_stream_the_lowering_emitted():
    """Every vcix op in the order the program issues it, loops expanded: the same sequence before
    and after the pass (bf16 with an epilogue, f32 wider than the lag), so the lowering's bound holds."""
    for kernel in (EPI, WIDE):
        before, after, _ = _lowered(kernel)
        assert after is not None, f"no {SCHED_DUMP}: the compile did not run the pass"
        ctx_a, m_a = _parse(before)
        with ctx_a:
            want = _stream(m_a.operation, [])
        ctx_b, m_b = _parse(after)
        with ctx_b:
            got = _stream(m_b.operation, [])
        assert want and got == want, (kernel, len(want), len(got))


def a_steady_trip_reads_the_next_input_then_writes_two_back_then_pops_behind_the_last_adds():
    """bf16 512^3: a steady loop's trip reads every C and A of the input after its own, then writes
    the C of the input two before, then runs the unit's ops in the lowering's order, each pop behind
    the add of the same pop an input earlier; the C it writes is the C it read three trips back."""
    _, after, _ = _lowered(EPI)
    ctx, m = _parse(after)
    with ctx:
        loops = _steady(m)
        assert loops, "no steady loop"
        checked = 0
        for loop, body in loops:
            at = {o: i for i, o in enumerate(body)}
            reads = [o for o in body if o.name == "vector.transfer_read"]
            writes = [o for o in body if o.name == "vector.transfer_write"]
            units = [o for o in body if _is_vcix(o)]
            adds = [o for o in body if o.name == "arith.addf"]
            assert len(loop.results), "a steady loop carries nothing from trip to trip: not pipelined"
            assert reads and writes, "a steady trip without its reads and writes"
            assert max(at[r] for r in reads) < min(at[w] for w in writes), "a read after a write"
            assert max(at[w] for w in writes) < min(at[u] for u in units), "a write inside the run"
            assert all(_carried(loop, r.results[0]) == 0 for r in reads)
            assert all(_carried(loop, w.operands[0]) >= 1 for w in writes), "a write of this trip's sum"
            pops = [u for u in units if len(u.results)]
            for a in adds:
                made = sorted(_origin(loop, v) for v in a.operands)
                assert made == [(1, "vcix.v.i"), (2, "vector.transfer_read")], (
                    f"an add not of last trip's pop and the C read two trips back: {made}")
                nxt = next(u for u in units if at[u] > at[a])
                assert nxt in pops, "an add not right before a pop"
            checked += bool(adds)
        assert checked, "no steady loop adds into C"


def the_two_reads_of_an_input_issued_together_lie_in_different_banks():
    """bf16 512^3: the vmatprep's A runs ahead of its vmatmul's by the trips that put the two reads a
    trip issues an odd number of 64-byte lines apart, from the address arithmetic alone."""
    _, after, _ = _lowered(EPI)
    ctx, m = _parse(after)
    with ctx:
        loops = _steady(m)
        assert loops, "no steady loop"
        for loop, body in loops:
            units = [o for o in body if _is_vcix(o)]
            prep, push = next((p, q) for p, q in zip(units, units[1:]) if sched._prepares(p))
            deep = {_carried(loop, prep.operands[0]), _carried(loop, push.operands[0])}
            assert len(deep) == 2, f"vmatprep and vmatmul read A as far ahead: {deep}"
            ahead, behind = _source(loop, prep), _source(loop, push)
            step = sched._bank_step(sched._access(ahead), sched._access(behind), {})
            assert step == "other", step


def _source(loop, push):
    """The op in `loop`'s body that made `push`'s register, through the iter_args."""
    block = loop.regions[0].blocks[0]
    args = list(block.arguments)[1:]
    yielded = list([x.operation for x in block.operations][-1].operands)
    v = push.operands[0]
    while isinstance(v, ir.BlockArgument):
        v = yielded[args.index(v)]
    return v.owner.operation


def a_panel_reads_its_weights_from_one_base_a_bank():
    """bf16 512^3: in a block, a weight register is read at an earlier one's offset plus a constant,
    so a panel's 32 have one base, not an offset each (a panel at constant offsets is llc's to fold)."""
    _, after, _ = _lowered(EPI)
    assert after is not None, f"no {SCHED_DUMP}: the compile did not run the pass"
    ctx, m = _parse(after)
    with ctx:
        reads = bases = 0
        for o in _all(m.operation):
            for region in o.regions:
                for block in region.blocks:
                    seen = set()
                    for x in [y.operation for y in block.operations]:
                        if x.name != "vcix.iv" or ir.IntegerAttr(x.attributes["opcode"]).value not in (
                                mxu.VMATPUSH, mxu.VMATPREP) or ir.IntegerAttr(x.attributes["imm"]).value & mxu.SECOND:
                            continue
                        read = x.operands[0].owner
                        if getattr(read, "name", None) != "vector.transfer_read":
                            continue
                        at = read.operands[2]
                        if not sched._linear(at, {})[0]:
                            continue
                        reads += 1
                        steps = getattr(at.owner, "name", None) == "arith.addi" and any(
                            v in seen for v in at.owner.operands)
                        bases += at not in seen and not steps
                        seen.add(at)
        assert reads and bases * 16 <= reads, f"{bases} bases for {reads} weight reads"


def _one_block(src):
    """`src` parsed, the pass run over it, and its func's ops as names with the vector offsets
    they access (the transfer's last index, a constant)."""
    ctx, m = _parse(src)
    with ctx, ir.Location.unknown():
        sched.run(m)
        assert m.operation.verify()
        body = [x.operation for x in m.body.operations[0].regions[0].blocks[0].operations]
        out = []
        for o in body:
            if o.name in ("vector.transfer_read", "vector.transfer_write"):
                at = o.operands[-1 if o.name == "vector.transfer_write" else 2]
                out.append((o.name.split(".")[1], ir.IntegerAttr(at.owner.attributes["value"]).value))
            elif _is_vcix(o):
                out.append(("pop", None))
        return out


_TWO_POPS = """
func.func @f(%c: memref<1x64xf32>) {
  %z = arith.constant 0 : index
  %a = arith.constant 0 : index
  %b = arith.constant OFFSET : index
  %pad = arith.constant 0.0 : f32
  %rvl = arith.constant 8 : i64
  %p0 = "vcix.v.i"(%rvl) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> : (i64) -> vector<8xi32>
  %f0 = arith.bitcast %p0 : vector<8xi32> to vector<8xf32>
  %r0 = vector.transfer_read %c[%z, %a], %pad {in_bounds = [true]} : memref<1x64xf32>, vector<8xf32>
  %s0 = arith.addf %r0, %f0 : vector<8xf32>
  vector.transfer_write %s0, %c[%z, %a] {in_bounds = [true]} : vector<8xf32>, memref<1x64xf32>
  %p1 = "vcix.v.i"(%rvl) <{imm = 4 : i64, opcode = 2 : i64, rs2 = 0 : i64}> : (i64) -> vector<8xi32>
  %f1 = arith.bitcast %p1 : vector<8xi32> to vector<8xf32>
  %r1 = vector.transfer_read %c[%z, %b], %pad {in_bounds = [true]} : memref<1x64xf32>, vector<8xf32>
  %s1 = arith.addf %r1, %f1 : vector<8xf32>
  vector.transfer_write %s1, %c[%z, %b] {in_bounds = [true]} : vector<8xf32>, memref<1x64xf32>
  return
}
"""


def a_read_of_c_moves_ahead_of_a_write_it_cannot_alias():
    """Two pops into C at offsets 0 and 8, eight f32 each: both reads ahead of both pops, both
    writes after."""
    got = _one_block(_TWO_POPS.replace("OFFSET", "8"))
    assert got == [("transfer_read", 0), ("transfer_read", 8), ("pop", None), ("pop", None),
                   ("transfer_write", 0), ("transfer_write", 8)], got


def a_read_of_c_stays_behind_a_write_that_overlaps_it():
    """At offsets 0 and 4 the second read sees the first write's last four elements: it stays
    after that write, whatever the grouping would rather."""
    got = _one_block(_TWO_POPS.replace("OFFSET", "4"))
    assert got.index(("transfer_write", 0)) < got.index(("transfer_read", 4)), got


_LOOP = """
func.func @f() {
  %c = memref.alloc() : memref<1x512xf32>
  %a = memref.alloc() : memref<1x512xi32>
  %z = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %ipad = arith.constant 0 : i32
  %rvl = arith.constant 8 : i64
  %eight = arith.constant 8 : index
  %shift = arith.constant SHIFT : index
  affine.for %m = 0 to 16 {
    %p = "vcix.v.i"(%rvl) <{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}> : (i64) -> vector<8xi32>
    %f = arith.bitcast %p : vector<8xi32> to vector<8xf32>
    %at = arith.muli %m, %eight : index
    %r = vector.transfer_read %c[%z, %at], %pad {in_bounds = [true]} : memref<1x512xf32>, vector<8xf32>
    %s = arith.addf %r, %f : vector<8xf32>
    %to = arith.addi %at, %shift : index
    vector.transfer_write %s, %c[%z, %to] {in_bounds = [true]} : vector<8xf32>, memref<1x512xf32>
    %w = vector.transfer_read %a[%z, %at], %ipad {in_bounds = [true]} : memref<1x512xi32>, vector<8xi32>
    "vcix.iv"(%w, %rvl) <{imm = 3 : i64, opcode = 0 : i64, rd = 0 : i64}> : (vector<8xi32>, i64) -> ()
  }
  return
}
"""


def _pipelined(shift):
    """The loop of _LOOP, its C written `shift` elements past where it read, after the pass: whether
    it carries values from trip to trip, and the stream it runs."""
    ctx, m = _parse(_LOOP.replace("SHIFT", str(shift)))
    with ctx, ir.Location.unknown():
        want = _stream(m.operation, [])
        sched.run(m)
        assert m.operation.verify()
        loops = [o for o in _all(m.operation) if o.name == "affine.for"]
        return any(len(o.results) for o in loops), _stream(m.operation, []) == want


def a_loop_whose_writes_the_next_reads_cannot_see_is_pipelined():
    """C written where it was read: an input's read meets no earlier input's write, so the reads go
    ahead of the writes of the inputs before, and the unit runs what it ran."""
    piped, same = _pipelined(0)
    assert piped and same, (piped, same)


def a_loop_whose_next_read_sees_a_write_stays_as_it_is():
    """C written 8 past where it was read: input m + 1 reads what input m writes, so reading it a
    trip ahead would read it stale -- the loop is left unpipelined."""
    piped, same = _pipelined(8)
    assert not piped and same, (piped, same)


CASES = [the_unit_runs_the_same_stream_the_lowering_emitted,
         a_steady_trip_reads_the_next_input_then_writes_two_back_then_pops_behind_the_last_adds,
         the_two_reads_of_an_input_issued_together_lie_in_different_banks,
         a_panel_reads_its_weights_from_one_base_a_bank,
         a_read_of_c_moves_ahead_of_a_write_it_cannot_alias,
         a_read_of_c_stays_behind_a_write_that_overlaps_it,
         a_loop_whose_writes_the_next_reads_cannot_see_is_pipelined,
         a_loop_whose_next_read_sees_a_write_stays_as_it_is]

if __name__ == "__main__":
    failed = 0
    for case in CASES:
        try:
            case()
            print("PASS ", case.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL ", case.__name__, "--", e)
    sys.exit(1 if failed else 0)
