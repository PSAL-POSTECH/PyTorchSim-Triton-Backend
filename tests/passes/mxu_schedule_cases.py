"""torchsim-schedule-mxu-stream issues the matrix unit's stream as p08 gave it and places the rest
around it: a steady loop reads C ahead of an input's pops, then adds and writes it, each in a run.
    python3 tests/passes/mxu_schedule_cases.py        exit 0 iff every case holds"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p08a_schedule_mxu_stream as sched  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import trip_count  # noqa: E402

MXU = os.path.join(ROOT, "tests", "targets", "tpuv6e.json")
EPI = os.path.join(ROOT, "tests", "cases", "mechanism", "tile", "tile_gemm_epilogue_at_pop.py")
WIDE = os.path.join(ROOT, "tests", "cases", "mechanism", "tile", "tile_gemm_wide_tile_grid.py")
P08_DUMP = "01-torchsim-lower-matmul-mxu.mlir"
SCHED_DUMP = "02-torchsim-schedule-mxu-stream.mlir"
_LOWERED = {}


def _lowered(kernel):
    """(p08's dump, torchsim-schedule-mxu-stream's or None, 09-custom.mlir) of `kernel` compiled
    to lower on the v6e, as text."""
    if kernel not in _LOWERED:
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), kernel,
                            "--workdir", d, "--to", "lower"], check=True, capture_output=True,
                           env=dict(os.environ, TORCHSIM_COMPILE_TARGET=MXU,
                                    TORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
            got = [os.path.join(d, "09-passes", P08_DUMP), os.path.join(d, "09-passes", SCHED_DUMP),
                   os.path.join(d, "09-custom.mlir")]
            _LOWERED[kernel] = tuple(open(p).read() if os.path.exists(p) else None for p in got)
    return _LOWERED[kernel]


def _parse(text):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        return ctx, ir.Module.parse(text)


def _stream(op, out):
    """The vcix ops under `op` in the order they run, as (name, opcode, imm): a loop of constant
    trips its body that many times, an op of one region that region once, of two or more none."""
    for region in op.regions:
        for block in region.blocks:
            for o in [x.operation for x in block.operations]:
                if o.name.startswith("vcix."):
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
                    assert not any(x.name.startswith("vcix.") for x in _all(o)), o.name
    return out


def _all(op):
    out = []
    for region in op.regions:
        for block in region.blocks:
            for o in [x.operation for x in block.operations]:
                out.append(o)
                out.extend(_all(o))
    return out


def _steady_bodies(module):
    """The block of every innermost loop that both pushes and pops."""
    out = []
    for o in _all(module.operation):
        if o.name != "affine.for":
            continue
        body = [x.operation for x in o.regions[0].blocks[0].operations]
        if any(x.regions for x in body):
            continue
        units = [x for x in body if x.name.startswith("vcix.")]
        if any(len(x.results) for x in units) and any(not len(x.results) for x in units):
            out.append(body)
    return out


def _runs(body):
    """Each run of pops of `body` with the ops after it up to the next push, as (pops, rest)."""
    out, i = [], 0
    while i < len(body):
        if body[i].name.startswith("vcix.") and len(body[i].results):
            j = i
            while j < len(body) and body[j].name.startswith("vcix.") and len(body[j].results):
                j += 1
            k = j
            while k < len(body) and not (body[k].name.startswith("vcix.") and not len(body[k].results)):
                k += 1
            out.append((body[i:j], body[j:k]))
            i = k
        else:
            i += 1
    return out


def the_unit_runs_the_same_stream_p08_emitted():
    """Every vcix op in the order the program issues it, loops expanded: the same sequence before
    and after the pass (bf16 with an epilogue, f32 wider than the lag), so p08's bound holds."""
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


def a_steady_loop_reads_c_before_its_pops_then_adds_and_writes_each_together():
    """bf16 512^3 past the lag: before each run of pops of a steady loop every read of C its adds
    take, after it every add before every write -- no chain finished before the next is begun."""
    _, _, after = _lowered(EPI)
    ctx, m = _parse(after)
    with ctx:
        bodies = _steady_bodies(m)
        assert bodies, "no steady loop"
        checked = 0
        for body in bodies:
            at = {o: i for i, o in enumerate(body)}
            for pops, rest in _runs(body):
                assert len(pops) >= 2, f"a run of {len(pops)} pop: its C chain comes before the next pop"
                adds = [o for o in rest if o.name == "arith.addf"]
                sums = {a.results[0] for a in adds}
                writes = [o for o in rest if o.name == "vector.transfer_write" and o.operands[0] in sums]
                if not adds:
                    continue
                checked += 1
                assert len(adds) == len(writes) > 1, (len(adds), len(writes))
                reads = [v.owner for a in adds for v in a.operands
                         if getattr(v.owner, "name", None) == "vector.transfer_read"]
                assert len(reads) == len(adds), "an add of C reads no C"
                assert all(at[r] < at[pops[0]] for r in reads), "a read of C after the pops"
                assert max(at[a] for a in adds) < min(at[w] for w in writes), "a write before an add"
        assert checked, "no steady loop adds into C"


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
            elif o.name.startswith("vcix."):
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


CASES = [the_unit_runs_the_same_stream_p08_emitted,
         a_steady_loop_reads_c_before_its_pops_then_adds_and_writes_each_together,
         a_read_of_c_moves_ahead_of_a_write_it_cannot_alias,
         a_read_of_c_stays_behind_a_write_that_overlaps_it]

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
