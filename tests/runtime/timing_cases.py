"""What the timing devices hand TOGSim: no compute tile overlaps the next (overlap 0 for every type),
and each gem5 dump's committed micro-ops become an Issue row at the CPU's issue width.
    python3 tests/runtime/timing_cases.py        exit 0 iff every case holds"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.device import gem5  # noqa: E402

_DUMP = """
---------- Begin Simulation Statistics ----------
system.cpu.numCycles                             {cycles}                       # Number of cpu cycles simulated (Cycle)
system.cpu.commitStats0.numInsts                 1                       # Number of instructions committed (thread level) (Count)
system.cpu.commitStats0.numOps                   {ops}                       # Number of ops (including micro ops) committed (thread level) (Count)
system.cpu.executeFuncUnits.funcUnits16.vcix.Systolic.admitted {rows}     # rows admitted at port 0 (Count)
system.cpu.executeFuncUnits.funcUnits16.vcix.Systolic.capacity 1     # (Count)
---------- End Simulation Statistics   ----------
"""


def _overlap_none():
    from pytorchsim_triton_compiler.lowering.tog.passes import build_tog as bt
    from pytorchsim_triton_compiler.lowering.tog.passes import cycle_table as ct
    from pytorchsim_triton_compiler.lowering.tog.passes import typed_grouping as tg
    types = (bt.VECTOR_COMPUTE, tg.MATMUL_COMPUTE, tg.MATMUL_PRELOAD, tg.CROSS_LANE_COMPUTE)
    got = {t: ct.overlap_offset(t, 256) for t in types}
    return all(v is None for v in got.values()), got


def _issue_rows():
    stats = _DUMP.format(cycles=1589, ops=1730, rows=1024) + _DUMP.format(cycles=5104, ops=6107, rows=4096)
    width = gem5.issue_width({"system": {"cpu": {"executeIssueLimit": 12}}})
    got = [w["units"].get(gem5.ISSUE) for w in gem5.windows(stats, width)]
    want = [{"admitted": 1730, "capacity": 12, "unit_of_work": "micro-ops", "cycles": 1589},
            {"admitted": 6107, "capacity": 12, "unit_of_work": "micro-ops", "cycles": 5104}]
    return got == want, got


CASES = {"overlap_offset_none_for_every_type": _overlap_none, "issue_row_per_dump": _issue_rows}


def main():
    failed = 0
    for name, case in CASES.items():
        try:
            ok, got = case()
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{name:36} {'PASS' if ok else 'FAIL'}  {got}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
