"""What the timing devices hand TOGSim: no compute tile overlaps the next, each gem5 dump's committed
micro-ops as an Issue row, every call's dump of each tile by the (tile, call) it names at row
tile + tiles * call, the grid in the order of the parallel axes.
    python3 tests/runtime/timing_cases.py        exit 0 iff every case holds"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.device import as_compiled, base, config, gem5  # noqa: E402
from pytorchsim_triton_compiler.contract import gem5_samples as gs  # noqa: E402

_DUMP = """
---------- Begin Simulation Statistics ----------
system.sample.tileId                             {tile}                       # Tile id the resetstats opening this dump passed, or -1 (Unspecified)
system.sample.call                               {call}                       # Call index the resetstats opening this dump passed, or -1 (Unspecified)
system.cpu.numCycles                             {cycles}                       # Number of cpu cycles simulated (Cycle)
system.cpu.commitStats0.numInsts                 1                       # Number of instructions committed (thread level) (Count)
system.cpu.commitStats0.numOps                   {ops}                       # Number of ops (including micro ops) committed (thread level) (Count)
system.cpu.executeFuncUnits.funcUnits16.vcix.TPUV6E_MXU.admitted {rows}     # rows admitted at port 0 (Count)
system.cpu.executeFuncUnits.funcUnits16.vcix.TPUV6E_MXU.capacity 1     # (Count)
---------- End Simulation Statistics   ----------
"""


def _overlap_none():
    from pytorchsim_triton_compiler.lowering.tog.passes import build_tog as bt
    from pytorchsim_triton_compiler.lowering.tog.passes import cycle_table as ct
    got = {t: ct.overlap_offset(t, 256) for t in bt.COMPUTE_TYPE_NAME}
    return all(v is None for v in got.values()), got


def _issue_rows():
    stats = (_DUMP.format(tile=0, call=0, cycles=1589, ops=1730, rows=1024)
             + _DUMP.format(tile=1, call=0, cycles=5104, ops=6107, rows=4096))
    width = gs.issue_width({"system": {"cpu": {"executeIssueLimit": 12}}})
    got = [w["units"].get(gs.ISSUE) for w in gs.dumps(stats, width)]
    want = [{"admitted": 1730, "capacity": 12, "unit_of_work": "micro-ops", "cycles": 1589},
            {"admitted": 6107, "capacity": 12, "unit_of_work": "micro-ops", "cycles": 5104}]
    return got == want, got


def _measure_on(dumps, n_tiles=2):
    """Gem5Device.measure(ports=True) over a stub gem5 whose stats.txt holds `dumps`, each (tile,
    call, cycles), and an end-of-run dump, for a TOG of `n_tiles` tiles."""
    stats = "".join(_DUMP.format(tile=t, call=c, cycles=n, ops=n + 1, rows=1)
                    for t, c, n in dumps + [(-1, -1, 99999)])
    with tempfile.TemporaryDirectory() as d:
        stub = os.path.join(d, "gem5")
        with open(stub, "w") as fh:
            fh.write(f"#!{sys.executable}\nimport os, sys\nout = sys.argv[sys.argv.index('-d') + 1]\n"
                     f"os.makedirs(out, exist_ok=True)\nopen(os.path.join(out, 'stats.txt'), 'w').write({stats!r})\n"
                     f"open(os.path.join(out, 'config.json'), 'w').write('{{\"system\": {{\"cpu\": "
                     f"{{\"executeIssueLimit\": 12}}}}}}')\n")
        os.chmod(stub, 0o755)
        types = os.path.join(d, "tile_types.json")
        with open(types, "w") as fh:
            json.dump({"compute_types": ["VectorCompute"] * n_tiles}, fh)

        class Target:
            vector_lanes, vlen_bits = 1, 128

        class Req(base.Launch):
            def artifact(self, kind):
                return types if kind == "tile_types" else "cycle_elf"

        saved = config.GEM5, as_compiled.target, as_compiled.run_options
        config.GEM5, as_compiled.target = stub, lambda req: Target()
        as_compiled.run_options = lambda req: {"gem5_script": "s", "model": "m", "machine_config": "c"}
        try:
            req = Req(d, {}, d, {}, log=lambda m: None, run=None)
            cycles, windows = gem5.Gem5Device().measure(req, ports=True)
        finally:
            config.GEM5, as_compiled.target, as_compiled.run_options = saved
    return cycles, [w["cycles"] for w in windows]


def _refused(dumps, says, n_tiles=2):
    try:
        got = _measure_on(dumps, n_tiles)
    except base.DeviceError as e:
        return says in str(e), str(e)
    return False, got


def _every_call_in_row_order():
    """Two calls of two tiles each: all four, call c of tile t at row t + 2 * c."""
    cycles, windows = _measure_on([(0, 0, 1055), (1, 0, 2210), (0, 1, 483), (1, 1, 1999)])
    return cycles == windows == [1055, 2210, 483, 1999], (cycles, windows)


def _keyed_not_positional():
    """Dumps that arrive out of program order still land on the row of the tile and call they name."""
    cycles, _ = _measure_on([(1, 1, 1999), (0, 0, 1055), (0, 1, 483), (1, 0, 2210)])
    return cycles == [1055, 2210, 483, 1999], cycles


def _partial_call_refused():
    """Three dumps cannot be two whole calls of two tiles: refused, naming what is missing."""
    return _refused([(0, 0, 1055), (1, 0, 2210), (0, 1, 483)], "missing (tile, call) [(1, 1)]")


def _duplicate_refused():
    """Two dumps naming one (tile, call) are refused, not one of them kept."""
    return _refused([(0, 0, 1), (1, 0, 2), (0, 1, 3), (0, 1, 4), (1, 1, 5)], "two dumps name tile 0 call 1")


def _unexpected_tile_refused():
    """A tile id past the TOG's tiles is refused, even when the count of dumps matches."""
    return _refused([(0, 0, 1), (2, 0, 2), (0, 1, 3), (2, 1, 4)], "unexpected [(2, 0), (2, 1)]")


def _unmarked_refused():
    """A dump no marker opened, before the end of the run, is refused."""
    return _refused([(0, 0, 1), (-1, -1, 2), (0, 1, 3)], "no marker opened it", n_tiles=1)


def _shape_in_axis_order():
    from pytorchsim_triton_compiler.contract import kernel_object as ko
    got = [ko.launch_shape({"grid": grid, "abi": {"parallel_axes": axes}})
           for grid, axes in (([4, 3, 2], ["z", "y", "x"]), ([4, 3, 2], ["x", "y", "z"]),
                              ([96, 1, 1], ["x"]), ([5], ["x"]), ([2, 7], ["y", "x"]))]
    return got == [[2, 3, 4], [4, 3, 2], [96], [5], [7, 2]], got


CASES = {"overlap_offset_none_for_every_type": _overlap_none, "issue_row_per_dump": _issue_rows,
         "gem5_hands_on_every_call_in_row_order": _every_call_in_row_order,
         "gem5_keys_dumps_by_tile_and_call": _keyed_not_positional,
         "gem5_refuses_a_partial_call": _partial_call_refused,
         "gem5_refuses_a_duplicate_dump": _duplicate_refused,
         "gem5_refuses_an_unexpected_tile": _unexpected_tile_refused,
         "gem5_refuses_an_unmarked_dump": _unmarked_refused,
         "shape_in_parallel_axis_order": _shape_in_axis_order}


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
